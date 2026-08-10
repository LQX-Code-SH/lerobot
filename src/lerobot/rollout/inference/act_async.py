# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ACT 异步推理引擎：后台线程产 action chunk 并整块推给桥接。

与 :class:`RTCInferenceEngine` 的区别
-------------------------------------
- 面向 ACT（单次前向预测整段 chunk），不调用 diffusion 专属的 ``denoise_step`` /
  RTC guidance；``predict_action_chunk`` 不传 ``inference_delay`` /
  ``prev_chunk_left_over``。
- 不维护 :class:`ActionQueue`、不在引擎侧 pop 单动作；``get_action`` 恒返回
  ``None``，旁路 base 策略的单动作下发路径（``send_next_action`` 收到 None 即不
  调 ``send_action``）。
- 产出的**整块** chunk 经 ``robot_wrapper.send_action_chunk`` 直接推给桥接，由
  桥接做融合 / 插值 / 滤波 / 300Hz 下发（chunk-to-bridge 架构）。

观测循环由 base 策略保持：每 tick ``robot.get_observation()`` ->
``_process_observation_and_notify`` -> ``engine.notify_observation``。引擎后台线程
消费最新观测，按 ``inference_hz`` 重规划，解耦推理延迟与执行节拍。
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import LatencyTracker
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

logger = logging.getLogger(__name__)

# 后台线程在暂停 / 空闲 / 无观测时的轮询间隔。
_ACT_ASYNC_IDLE_SLEEP_S: float = 0.01
# 连续瞬时错误间的退避间隔（按连续失败次数累加）。
_ACT_ASYNC_ERROR_RETRY_DELAY_S: float = 0.5
# 连续瞬时错误容忍上限，超过即放弃并传播 shutdown。
_ACT_ASYNC_MAX_CONSECUTIVE_ERRORS: int = 10
# stop() 时 join 后台线程的硬超时。
_ACT_ASYNC_JOIN_TIMEOUT_S: float = 3.0


@dataclass
class ACTAsyncChunkConfig:
    """ACT 异步 chunk 引擎参数（由 ``--inference.type=act_async`` 选择）。

    Attributes:
        inference_hz: 重规划频率。每 ~1/inference_hz 秒推一块新 chunk 到桥接。
            过高会与 300Hz PD 线程争 CPU，过低会令轨迹滞后。默认 3Hz。
        execution_horizon: 截断 chunk 到前 N 步（重规划视界），0 = 不截断。
            ACT 训练时 chunk_size=100（~6.7s@15Hz），部署时只需执行近期视界即可，
            截断可减少 ZMQ 传输与桥接融合开销。默认 30（~2s）。
        idle_sleep_s: 空闲轮询间隔。
    """

    inference_hz: float = 3.0
    execution_horizon: int = 30
    idle_sleep_s: float = _ACT_ASYNC_IDLE_SLEEP_S


class ACTAsyncChunkEngine(InferenceEngine):
    """异步产 chunk 并整块推给桥接的 ACT 推理引擎。

    ``get_action`` 恒返回 ``None``：动作不经策略单动作路径下发，而是由本引擎后台
    线程把整块 chunk 经 ``robot_wrapper.send_action_chunk`` 推给桥接。主循环仍应
    每 tick 调 ``notify_observation`` 喂入最新观测，并在人机介入阶段调
    ``pause`` / ``resume``。
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        cfg: ACTAsyncChunkConfig,
        hw_features: dict,
        task: str,
        fps: float,
        device: str | None,
        ordered_action_keys: list[str],
        shutdown_event: Event | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._cfg = cfg
        self._hw_features = hw_features
        self._task = task
        self._fps = float(fps)
        self._device = device or "cpu"
        self._ordered_action_keys = list(ordered_action_keys)

        self._obs_holder: dict[str, Any] = {"obs": None}
        self._obs_lock = Lock()
        self._policy_active = Event()
        self._shutdown_event = Event()
        self._engine_error = Event()
        self._global_shutdown_event = shutdown_event
        self._chunk_thread: Thread | None = None
        self._chunk_id = 0

        if self._cfg.inference_hz <= 0:
            raise ValueError(f"inference_hz must be > 0, got {self._cfg.inference_hz}")
        self._target_period = 1.0 / float(self._cfg.inference_hz)

        logger.info(
            "ACTAsyncChunkEngine initialized (inference_hz=%.2f execution_horizon=%d fps=%.1f)",
            self._cfg.inference_hz, self._cfg.execution_horizon, self._fps,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """本引擎无 torch.compile warmup 需求，立即可用。"""
        return True

    @property
    def failed(self) -> bool:
        """后台线程因不可恢复错误退出时为 True。"""
        return self._engine_error.is_set()

    def start(self) -> None:
        """启动后台 chunk 生产线程。"""
        self._obs_holder = {"obs": None, "robot_type": self._robot.robot_type}
        self._shutdown_event.clear()
        self._chunk_id = 0
        self._chunk_thread = Thread(
            target=self._chunk_loop, daemon=True, name="ACTAsyncChunk"
        )
        self._chunk_thread.start()
        logger.info("ACT async chunk thread started")

    def stop(self) -> None:
        """通知后台线程退出并等待。"""
        logger.info("Stopping ACT async chunk thread...")
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._chunk_thread is not None and self._chunk_thread.is_alive():
            self._chunk_thread.join(timeout=_ACT_ASYNC_JOIN_TIMEOUT_S)
            if self._chunk_thread.is_alive():
                logger.warning("ACT async chunk thread did not join within %.1fs", _ACT_ASYNC_JOIN_TIMEOUT_S)
            else:
                logger.info("ACT async chunk thread stopped")
            self._chunk_thread = None

    def pause(self) -> None:
        """暂停后台推理（人机介入阶段）。"""
        logger.info("Pausing ACT async chunk thread")
        self._policy_active.clear()

    def resume(self) -> None:
        """恢复后台推理。"""
        logger.info("Resuming ACT async chunk thread")
        self._policy_active.set()

    def reset(self) -> None:
        """重置策略与处理器（episode 间）。"""
        logger.info("Resetting ACT async engine state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        with self._obs_lock:
            self._obs_holder["obs"] = None
        self._chunk_id = 0

    # ------------------------------------------------------------------
    # Action production（旁路单动作路径）
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """恒返回 None：动作由后台线程整块推给桥接，不经策略单动作路径。"""
        return None

    def notify_observation(self, obs: dict) -> None:
        """发布最新观测供后台线程消费（latest-wins）。"""
        with self._obs_lock:
            self._obs_holder["obs"] = obs

    # ------------------------------------------------------------------
    # 后台 chunk 生产线程
    # ------------------------------------------------------------------

    def _chunk_loop(self) -> None:
        """按 inference_hz 重规划，把整块 chunk 推给桥接。"""
        try:
            latency_tracker = LatencyTracker()
            policy_device = torch.device(self._device)
            consecutive_errors = 0

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(self._cfg.idle_sleep_s)
                    continue

                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                if obs is None:
                    time.sleep(self._cfg.idle_sleep_s)
                    continue

                # obs 消费时刻（wall clock）：桥接据此算精确推理延迟 L = send_ts - obs_ts，
                # 替代仅用推理耗时 inference_time_sec（后者不含后处理/发送，偏小）。
                obs_ts = time.time()
                t0 = time.perf_counter()
                try:
                    # 1. 构建推理 batch（镜像 RTCInferenceEngine._rtc_loop）
                    obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
                    obs_batch = prepare_observation_for_inference(
                        obs_batch, policy_device, self._task, self._robot.robot_type
                    )
                    obs_batch["task"] = [self._task]
                    preprocessed = self._preprocessor(obs_batch)

                    # 2. 单次前向预测整段 chunk（ACT 不传 inference_delay/prev_chunk_left_over）
                    actions = self._policy.predict_action_chunk(preprocessed)  # [1, C, D]

                    # 3. 后处理 -> 绝对动作空间 [C, D]
                    processed = self._postprocessor(actions).squeeze(0)  # [C, D] tensor
                    if self._cfg.execution_horizon > 0:
                        processed = processed[: self._cfg.execution_horizon]
                    processed_np = processed.detach().cpu().numpy()  # [C, D]

                    # 4. 列 -> feature 名（与 sync/core 的 ordered_action_keys 映射一致）
                    chunk_dict: dict[str, list[float]] = {
                        key: processed_np[:, i].tolist()
                        for i, key in enumerate(self._ordered_action_keys)
                        if i < processed_np.shape[1]
                    }

                    latency = time.perf_counter() - t0
                    latency_tracker.add(latency)
                    self._chunk_id += 1

                    # 5. 整块推给桥接（chunk-to-bridge）
                    # pause 落在 predict 窗口：丢弃在飞 chunk，避免 stop 后仍 push 一块。
                    if not self._policy_active.is_set():
                        continue
                    self._robot.send_action_chunk(
                        chunk_dict,
                        inference_time_sec=latency,
                        obs_time_sec=obs_ts,
                        chunk_id=self._chunk_id,
                        fps=self._fps,
                    )

                    consecutive_errors = 0
                    logger.debug(
                        "ACT chunk pushed: id=%d shape=%s latency=%.3fs p95=%.3fs",
                        self._chunk_id, processed_np.shape, latency, latency_tracker.p95(),
                    )

                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        "ACT async inference error (%d/%d): %s",
                        consecutive_errors, _ACT_ASYNC_MAX_CONSECUTIVE_ERRORS, e,
                    )
                    logger.debug(traceback.format_exc())
                    if consecutive_errors >= _ACT_ASYNC_MAX_CONSECUTIVE_ERRORS:
                        raise
                    time.sleep(_ACT_ASYNC_ERROR_RETRY_DELAY_S * consecutive_errors)

                # 6. 节流到 inference_hz（推理耗时超过周期则不睡）
                elapsed = time.perf_counter() - t0
                remaining = self._target_period - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        except Exception as e:
            logger.error("Fatal error in ACT async chunk thread: %s", e)
            logger.error(traceback.format_exc())
            self._engine_error.set()
            # 通知顶层 shutdown，让策略控制循环退出
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
