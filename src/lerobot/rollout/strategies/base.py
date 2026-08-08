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

"""Base rollout strategy: autonomous policy execution with no data recording."""

from __future__ import annotations

import logging
import time

from lerobot.utils.robot_utils import precise_sleep

from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    All actions flow through the ``robot_action_processor`` pipeline
    before reaching the robot.
    """

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine."""
        self._init_engine(ctx)
        logger.info("Base strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop until shutdown or duration expires."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Base strategy control loop started")

        # 每步耗时窗口：每 5s 汇总一次。
        # - period（墙钟周期，含 sleep）→ 真实达成 Hz 与 jitter（max）。
        # - obs/proc/act（三段工作时间）→ jitter 来源定位。
        # ACT 开环：前向只在 policy action_queue 空时跑（约 1/n_action_steps 步），
        # 故 act 的 max 暴露前向尖峰，avg 反映每步流水线（GPU 搬运+预处理）开销。
        last_report = start_time
        prev_loop_start = None
        n = n_period = 0
        period_sum = period_max = 0.0
        obs_sum = proc_sum = act_sum = 0.0
        obs_max = proc_max = act_max = 0.0
        infer_sum = dispatch_sum = 0.0
        infer_max = dispatch_max = 0.0

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()
            if prev_loop_start is not None:
                period = loop_start - prev_loop_start
                period_sum += period
                n_period += 1
                if period > period_max:
                    period_max = period
            prev_loop_start = loop_start

            if cfg.duration > 0 and (loop_start - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            _t = time.perf_counter()
            obs = robot.get_observation()
            t_obs = time.perf_counter() - _t

            _t = time.perf_counter()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)
            t_proc = time.perf_counter() - _t

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            _timing: dict = {}
            action_dict = send_next_action(obs_processed, obs, ctx, interpolator, timing=_timing)
            t_infer = _timing.get("infer", 0.0)
            t_dispatch = _timing.get("dispatch", 0.0)
            t_act = t_infer + t_dispatch
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)

            n += 1
            obs_sum += t_obs
            proc_sum += t_proc
            act_sum += t_act
            infer_sum += t_infer
            dispatch_sum += t_dispatch
            if t_obs > obs_max:
                obs_max = t_obs
            if t_proc > proc_max:
                proc_max = t_proc
            if t_act > act_max:
                act_max = t_act
            if t_infer > infer_max:
                infer_max = t_infer
            if t_dispatch > dispatch_max:
                dispatch_max = t_dispatch

            now = time.perf_counter()
            if now - last_report >= 5.0:
                if n > 0:
                    avg_period = period_sum / n_period if n_period else 0.0
                    avg_hz = 1.0 / avg_period if avg_period > 0 else 0.0
                    target_hz = 1.0 / control_interval if control_interval > 0 else 0.0
                    slow = avg_period > control_interval
                    msg = (
                        f"loop timing (n={n}, period avg={avg_period * 1e3:.1f}ms/{avg_hz:.1f}Hz "
                        f"max={period_max * 1e3:.1f}ms vs target {target_hz:.0f}Hz/{control_interval * 1e3:.1f}ms): "
                        f"obs avg={obs_sum / n * 1e3:.1f} max={obs_max * 1e3:.1f} | "
                        f"proc avg={proc_sum / n * 1e3:.1f} max={proc_max * 1e3:.1f} | "
                        f"act avg={act_sum / n * 1e3:.1f} max={act_max * 1e3:.1f} ms "
                        f"(infer avg={infer_sum / n * 1e3:.1f} max={infer_max * 1e3:.1f} | "
                        f"dispatch avg={dispatch_sum / n * 1e3:.1f} max={dispatch_max * 1e3:.1f})"
                    )
                    if slow:
                        logger.warning(
                            msg + " | 控制环慢于目标，可能掉帧/抖动。常见原因：1) 相机 FPS 不足 "
                            "2) 预处理/前向耗时 3) CPU 争用（preview/relay）"
                        )
                    else:
                        logger.info(msg)
                last_report = now
                n = n_period = 0
                period_sum = period_max = 0.0
                obs_sum = proc_sum = act_sum = 0.0
                obs_max = proc_max = act_max = 0.0
                infer_sum = dispatch_sum = 0.0
                infer_max = dispatch_max = 0.0

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware and stop inference."""
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Base strategy teardown complete")
