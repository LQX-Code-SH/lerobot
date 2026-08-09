#!/usr/bin/env python
"""Walker S2 推理服务器（常驻预热 + ZMQ 指令/状态 + 远程拉起/热切换）。

本模块是 ``lerobot-rollout`` 的常驻变体：一次性预热加载模型 + 连机器人，之后外部经 ZMQ
指令随时 ``start/stop/home/status/shutdown``，并可 ``load <params>`` 热切换模型（整体重建
context，含机器人重连）。免去每次部署的冷启动（policy 加载到 CUDA + 桥接拉起）。

架构
----
- 预热：复用 ``build_rollout_context``（policy + robot + processor + engine），与
  ``lerobot-rollout`` 完全一致，零重复逻辑。
- 启停：``ACTAsyncChunkEngine.pause()/resume()/reset()`` 天然映射 stop/start/新 episode。
  引擎后台线程整块推 chunk 给桥接（300Hz densify），主循环只喂观测，pause 即停。
- 控制接口：ZMQ REP(5570) 收发指令 + ZMQ PUB(5571) 广播状态（与桥接 5561/5562/5563 同体系）。
- load 热切换：指令线程置 ``LOADING``，**主线程**观测循环执行重建（stop 旧 engine +
  disconnect 旧 robot + ``build_rollout_context`` 新 cfg + ``_init_engine``），避免跨线程
  ctx 失效；``LOADING`` 期间主循环跳过 ``get_observation``（不碰正断开的 robot）。

不改动 act_async 引擎 / ros2_walker_bridge / walker.py 核心，仅 import 复用。

用法
----
::

    # 容器内（由 inference_server.sh 后台拉起，或直接）：
    python -m lerobot.scripts.inference_server \\
        --policy.path=/ubt_IL/model/.../pretrained_model \\
        --robot.type=walker --robot.joint_config=walker_s2_10d \\
        --inference.type=act_async --inference.act_async.inference_hz=2 \\
        --fps=15 --task="pick part" \\
        --server.cmd_port=5570 --server.status_port=5571

    # 外部客户端：inference_client.sh start|stop|home|status|shutdown|load
"""

# 注意：不用 `from __future__ import annotations`。parser.wrap() 经
# inspect.getfullargspec 取 _parse_rollout_config 的注解作为 config_class，
# PEP 563 字符串化注解会让 draccus.parse 收到 "RolloutConfig" 字符串而非类，报错。
# 容器为 Python 3.12，PEP 604 联合类型（X | None）原生可用，无需 future import。

import argparse
import atexit
import logging
import os
import sys
import time
from threading import Event, Lock, Thread
from typing import Any

import zmq

# ── 注册标准 robot/teleop 类型（镜像 lerobot_rollout.py；walker 由插件注册）──────
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.rollout import RolloutConfig, build_rollout_context
from lerobot.rollout.context import RolloutContext
from lerobot.rollout.strategies.core import RolloutStrategy, send_next_action
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────

DEFAULT_CMD_PORT = 5570
DEFAULT_STATUS_PORT = 5571

# 模型状态机取值。
STATE_LOADING = "LOADING"            # 预热 / load 重建中
STATE_READY = "READY"                # engine paused，空闲待命
STATE_RUNNING = "RUNNING"            # engine resumed，机器人自主执行
STATE_RETURNING_HOME = "RETURNING_HOME"  # home 指令执行中
STATE_ERROR = "ERROR"                # engine 致命错误 / load 失败
STATE_SHUTTING_DOWN = "SHUTTING_DOWN"  # 关机中

# 状态广播间隔（秒）。
_STATUS_PUB_INTERVAL_S = 0.5
# 指令线程 poll 超时（秒）-- 周期性检查 shutdown。
_CMD_POLL_TIMEOUT_MS = 200
# READY(paused) 态观测循环降频轮询间隔（秒），省 CPU 仍维持 obs 新鲜。
_IDLE_OBS_INTERVAL_S = 0.2
# home 前 etc. 让在飞 chunk 落桥接的缓冲。
_CHUNK_SETTLE_S = 0.15


# ── 参数处理 ──────────────────────────────────────────────────────────────────


def _consume_server_args() -> tuple[int, int]:
    """从 ``sys.argv`` 剥离 ``--server.*`` 参数，剩余留给 RolloutConfig 解析。

    用 ``parse_known_args`` 只消费 ``--server.cmd_port`` / ``--server.status_port``，
    其余（RolloutConfig 参数）留在 remaining；随后把 ``sys.argv`` 重写为 remaining，
    使后续 ``@parser.wrap()`` 只看到 RolloutConfig 参数（draccus 遇未知参数会报错）。
    端口可由环境变量 ``SERVER_CMD_PORT`` / ``SERVER_STATUS_PORT`` 默认。
    """
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument(
        "--server.cmd_port",
        type=int,
        default=int(os.environ.get("SERVER_CMD_PORT", str(DEFAULT_CMD_PORT))),
        dest="server_cmd_port",
    )
    p.add_argument(
        "--server.status_port",
        type=int,
        default=int(os.environ.get("SERVER_STATUS_PORT", str(DEFAULT_STATUS_PORT))),
        dest="server_status_port",
    )
    ns, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return ns.server_cmd_port, ns.server_status_port


@parser.wrap()
def _parse_rollout_config(cfg: RolloutConfig) -> RolloutConfig:
    """``parser.wrap`` 入口：读 ``sys.argv`` 构造 RolloutConfig 并返回。"""
    return cfg


def build_rollout_config(cli_args: list[str]) -> RolloutConfig:
    """从给定 CLI 参数列表构造 RolloutConfig（复用 ``parser.wrap`` 全部逻辑）。

    ``parser.wrap`` / ``RolloutConfig.__post_init__`` 通过 ``get_path_arg`` 读 ``sys.argv``
    取 ``--policy.path``，故这里临时把 ``sys.argv`` 置为 ``cli_args`` 再调用，完成后还原。
    load 热切换时用此函数从客户端 params 构造新 cfg。
    """
    old_argv = sys.argv
    sys.argv = [old_argv[0] if old_argv else "inference_server"] + list(cli_args)
    try:
        return _parse_rollout_config()
    finally:
        sys.argv = old_argv


# ── load params → CLI args ────────────────────────────────────────────────────

# 桥接从 os.environ 读取的 chunk_consumer 覆盖项（ros2_walker_bridge.py 1019-1034）。
_BRIDGE_ENV_PARAMS = {
    "blend_horizon": ("BLEND_HORIZON", int),
    "body_publish_hz": ("BODY_PUBLISH_HZ", float),
    "body_v_max": ("BODY_V_MAX", float),
    "transition_ramp_pts": ("TRANSITION_RAMP_PTS", int),
    "smoothing_window": ("SMOOTHING_WINDOW", int),
    "chunk_consumer_enabled": ("CHUNK_CONSUMER_ENABLED", None),  # bool，特殊处理
}


def _params_to_cli_args(params: dict[str, Any]) -> list[str]:
    """把 load 指令的 params dict 映射成 ``lerobot-rollout`` 同款 CLI 参数列表。"""
    args: list[str] = ["--strategy.type=base", "--robot.type=walker"]
    if params.get("robot_model"):
        args.append(f"--robot.joint_config={params['robot_model']}")
    if params.get("robot_config"):
        args.append(f"--robot.robot_config_path={params['robot_config']}")
    if params.get("policy_path"):
        args.append(f"--policy.path={params['policy_path']}")

    itype = params.get("inference_type", "act_async")
    args.append(f"--inference.type={itype}")
    if itype == "act_async":
        if "inference_hz" in params:
            args.append(f"--inference.act_async.inference_hz={params['inference_hz']}")
        if "execution_horizon" in params:
            args.append(f"--inference.act_async.execution_horizon={params['execution_horizon']}")

    if "fps" in params:
        args.append(f"--fps={params['fps']}")
    if params.get("task"):
        args.append(f"--task={params['task']}")
    return args


def _apply_bridge_env(params: dict[str, Any]) -> None:
    """把 load params 中的桥接 chunk_consumer 项写入 ``os.environ``。

    桥接子进程在 ``_start_bridge`` 时继承本进程环境，故 load 重建前更新 env，新桥接即生效。
    """
    for key, (env_key, cast) in _BRIDGE_ENV_PARAMS.items():
        if key not in params:
            continue
        val = params[key]
        if env_key == "CHUNK_CONSUMER_ENABLED":
            os.environ[env_key] = "1" if val else "0"
        elif cast is not None:
            os.environ[env_key] = str(cast(val))


# ── 推理服务器 ────────────────────────────────────────────────────────────────


class InferenceServer(RolloutStrategy):
    """常驻推理服务：预热 + ZMQ 指令/状态 + load 热切换。

    继承 ``RolloutStrategy`` 以复用 ``_init_engine`` / ``_process_observation_and_notify``
    / ``_handle_warmup`` / ``_teardown_hardware``。不使用 ``create_strategy`` 调度，由
    ``serve()`` 直接驱动 setup→run→teardown 生命周期。
    """

    def __init__(
        self,
        cfg: RolloutConfig,
        cmd_port: int,
        status_port: int,
        shutdown_event: Event,
    ) -> None:
        super().__init__(cfg.strategy)
        self._cfg = cfg
        self._cmd_port = cmd_port
        self._status_port = status_port
        self._shutdown_event = shutdown_event

        # 共享状态
        self._ctx_lock = Lock()             # 守护 _ctx_holder 交换
        self._ctx_holder: RolloutContext | None = None
        self._state_lock = Lock()           # 守护 _state + telemetry
        self._state = STATE_LOADING
        self._pending_load: dict[str, Any] | None = None
        self._load_lock = Lock()            # 守护 pending_load 置位
        self._load_error: str | None = None

        # telemetry（观测循环更新，状态线程读）
        self._loop_hz: float = 0.0
        self._uptime_start = time.time()

        # ZMQ
        self._zmq_ctx = zmq.Context.instance()
        self._rep_socket: zmq.Socket | None = None
        self._pub_socket: zmq.Socket | None = None
        self._threads: list[Thread] = []

    # ------------------------------------------------------------------
    # 状态 / telemetry 辅助
    # ------------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        with self._state_lock:
            if self._state == state:
                return  # 同态不重复置位/打日志（shutdown: cmd 线程与 teardown 都设 SHUTTING_DOWN）
            self._state = state
        logger.info("state -> %s", state)

    def _policy_path(self) -> str:
        return str(getattr(self._cfg.policy, "pretrained_path", "") or "")

    def _model_name(self) -> str:
        p = self._policy_path()
        if not p:
            return ""
        parts = p.rstrip("/").split("/")
        if "checkpoints" in parts:
            i = parts.index("checkpoints")
            return parts[i - 1] if i > 0 else parts[-1]
        return parts[-2] if len(parts) >= 2 else parts[-1]

    def _robot_connected(self) -> bool:
        ctx = self._ctx_holder
        if ctx is None:
            return False
        try:
            return ctx.hardware.robot_wrapper.inner.is_connected
        except Exception:
            return False

    def _status_dict(self) -> dict[str, Any]:
        with self._state_lock:
            state = self._state
            loop_hz = self._loop_hz
            load_error = self._load_error
        eng = self._engine
        return {
            "state": state,
            "model": self._model_name(),
            "policy_path": self._policy_path(),
            "loop_hz": round(loop_hz, 1),
            "chunk_id": int(getattr(eng, "_chunk_id", 0)) if eng is not None else 0,
            "robot_connected": self._robot_connected(),
            "uptime_s": round(time.time() - self._uptime_start, 1),
            "load_error": load_error,
            "ts": time.time(),
        }

    def _ok(self, cmd: str, **extra: Any) -> dict[str, Any]:
        resp = {"ok": True, "cmd": cmd, **self._status_dict()}
        resp.update(extra)
        return resp

    def _err(self, cmd: str, error: str, **extra: Any) -> dict[str, Any]:
        resp = {"ok": False, "cmd": cmd, "error": error, **self._status_dict()}
        resp.update(extra)
        return resp

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def serve(self) -> None:
        """预热 → setup → run → teardown。"""
        self._set_state(STATE_LOADING)
        logger.info("Preheating: loading policy + connecting robot (build_rollout_context)...")
        try:
            ctx = build_rollout_context(self._cfg, self._shutdown_event)
        except Exception as e:
            self._load_error = str(e)
            self._set_state(STATE_ERROR)
            logger.exception("Preheat failed")
            return
        with self._ctx_lock:
            self._ctx_holder = ctx
        try:
            self.setup(ctx)
            logger.info(
                "Inference server READY. cmd=tcp://*:%d status=tcp://*:%d model=%s",
                self._cmd_port, self._status_port, self._model_name(),
            )
            self.run(ctx)
        finally:
            self.teardown(self._ctx_holder)

    def setup(self, ctx: RolloutContext) -> None:
        """初始化引擎（start 后立即 pause，空闲待命）+ 启动 ZMQ 线程。"""
        self._init_engine(ctx)
        # start() 已起 chunk 线程但 _policy_active 默认未置位（paused）；显式 pause 以明示。
        eng = self._engine
        if eng is not None and hasattr(eng, "pause"):
            eng.pause()
        self._set_state(STATE_READY)
        self._start_zmq()

    def teardown(self, ctx: RolloutContext | None) -> None:
        """停 ZMQ + 停 engine + 断开机器人（disconnect 内部回 home_position）。

        ``return_to_initial_position=False``：常驻服务器不走 rollout「回 episode 起点」语义。
        shutdown 应直接回 home（disconnect 的 _send_home_chunk，与 ``home`` 命令同款），
        而非先绕到预热时捕获的 initial_position 再回 home——否则机器人在非 home 位启动时
        shutdown 会多一段意外绕路动作。
        """
        self._set_state(STATE_SHUTTING_DOWN)
        self._stop_zmq()
        if ctx is not None:
            try:
                self._teardown_hardware(ctx.hardware, return_to_initial_position=False)
            except Exception as e:
                logger.warning("teardown_hardware failed: %s", e)
        logger.info("Inference server stopped")

    # ------------------------------------------------------------------
    # 观测循环（主线程）
    # ------------------------------------------------------------------

    def run(self, ctx: RolloutContext) -> None:
        """常驻观测循环：喂 obs 给 engine；处理 pending_load；telemetry。

        与 ``BaseStrategy.run`` 几乎一致，去 duration 上限、保持 engine paused（等 start），
        并在循环顶部处理 load 重建（主线程持有 ctx 引用，避免跨线程失效）。
        """
        prev_loop_start: float | None = None
        period_sum = 0.0
        n_period = 0
        last_report = time.perf_counter()
        is_act_async = self._engine is not None and hasattr(self._engine, "resume")

        while not self._shutdown_event.is_set():
            # 1. 处理 pending load（主线程执行重建，独占 ctx 交换）
            if self._pending_load is not None:
                with self._load_lock:
                    params = self._pending_load
                    self._pending_load = None
                self._do_load(params)
                is_act_async = self._engine is not None and hasattr(self._engine, "resume")
                continue

            cur_ctx = self._ctx_holder
            if cur_ctx is None or self._state in (STATE_LOADING, STATE_SHUTTING_DOWN, STATE_ERROR):
                time.sleep(0.05)
                continue

            loop_start = time.perf_counter()
            if prev_loop_start is not None:
                period_sum += loop_start - prev_loop_start
                n_period += 1
            prev_loop_start = loop_start

            try:
                obs = cur_ctx.hardware.robot_wrapper.get_observation()
                self._process_observation_and_notify(cur_ctx.processors, obs)
                # 不调 _handle_warmup：它在 torch.compile warmup 完成时会 engine.resume()，
                # 破坏「start 前 engine 保持 paused」语义。服务器面向 ACT（engine.ready 恒 True、
                # 无需 compile warmup），故跳过；--use_torch_compile 不被支持。
                # act_async：动作由引擎后台线程整块推桥接，send_next_action 是 no-op，跳过省开销。
                # 非 act_async（sync）才走单动作路径（但 sync 无 pause/resume，start/stop 不适用）。
                if not is_act_async:
                    send_next_action(
                        self._cached_obs_processed, obs, cur_ctx, self._interpolator
                    )
            except Exception:
                logger.exception("observation loop iteration error (non-fatal)")

            # telemetry：每 ~1s 更新 loop_hz
            now = time.perf_counter()
            if now - last_report >= 1.0:
                if n_period > 0 and period_sum > 0:
                    with self._state_lock:
                        self._loop_hz = n_period / period_sum
                last_report = now
                period_sum = 0.0
                n_period = 0

            # 节流：RUNNING 跑 fps；READY(paused) 降频省 CPU 仍喂 obs
            dt = time.perf_counter() - loop_start
            if self._state == STATE_RUNNING:
                # 每轮重算：load 后 fps/interpolator 可能变化，预计算会过期
                control_interval = self._interpolator.get_control_interval(self._cfg.fps)
                sleep_t = control_interval - dt
            else:
                sleep_t = _IDLE_OBS_INTERVAL_S - dt
            if sleep_t > 0:
                precise_sleep(sleep_t)

    # ------------------------------------------------------------------
    # load 热切换（主线程执行）
    # ------------------------------------------------------------------

    def _do_load(self, params: dict[str, Any]) -> None:
        """整体重建 context：stop 旧 engine + disconnect 旧 robot + build 新 + init 新。

        用户已确认「整体重建（含重连）」：复用 ``build_rollout_context`` 全流程，桥接重启
        ~1-2s 可接受（模型加载才是大头）。
        """
        self._set_state(STATE_LOADING)
        self._load_error = None
        logger.info("load: rebuilding context for %s", params.get("policy_path"))
        try:
            # 1. 停旧 engine
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception as e:
                    logger.warning("old engine stop failed: %s", e)
            # 2. 断开旧 robot（disconnect 内部回 home + kill 旧桥接/relay）
            old_ctx = self._ctx_holder
            if old_ctx is not None:
                old_robot = old_ctx.hardware.robot_wrapper.inner
                if old_robot.is_connected:
                    try:
                        old_robot.disconnect()
                    except Exception as e:
                        logger.warning("old robot disconnect failed: %s", e)
            with self._ctx_lock:
                self._ctx_holder = None

            # 3. 更新桥接 env（chunk_consumer 覆盖项）
            _apply_bridge_env(params)

            # 4. 构造新 cfg + build 新 context（连新 robot、加载新 policy）
            cli_args = _params_to_cli_args(params)
            new_cfg = build_rollout_config(cli_args)
            new_ctx = build_rollout_context(new_cfg, self._shutdown_event)

            # 5. 交换 ctx + 初始化新 engine（start 后 pause）
            with self._ctx_lock:
                self._ctx_holder = new_ctx
            self._cfg = new_cfg
            self._init_engine(new_ctx)
            eng = self._engine
            if eng is not None and hasattr(eng, "pause"):
                eng.pause()

            self._set_state(STATE_READY)
            logger.info("load complete: model=%s", self._model_name())
        except Exception as e:
            self._load_error = str(e)
            self._set_state(STATE_ERROR)
            logger.exception("load failed")

    # ------------------------------------------------------------------
    # ZMQ 线程
    # ------------------------------------------------------------------

    def _start_zmq(self) -> None:
        self._rep_socket = self._zmq_ctx.socket(zmq.REP)
        self._rep_socket.bind(f"tcp://*:{self._cmd_port}")
        self._pub_socket = self._zmq_ctx.socket(zmq.PUB)
        self._pub_socket.bind(f"tcp://*:{self._status_port}")
        t_cmd = Thread(target=self._cmd_loop, name="server_cmd", daemon=True)
        t_status = Thread(target=self._status_loop, name="server_status", daemon=True)
        self._threads = [t_cmd, t_status]
        t_cmd.start()
        t_status.start()

    def _stop_zmq(self) -> None:
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)
        self._threads = []
        for sock in (self._rep_socket, self._pub_socket):
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
        self._rep_socket = None
        self._pub_socket = None

    def _cmd_loop(self) -> None:
        """REP 指令循环：poll 收一条 → handle → 回一条。"""
        poller = zmq.Poller()
        poller.register(self._rep_socket, zmq.POLLIN)
        while not self._shutdown_event.is_set():
            try:
                events = dict(poller.poll(timeout=_CMD_POLL_TIMEOUT_MS))
            except zmq.ZMQError:
                break
            if self._rep_socket not in events:
                continue
            try:
                req = self._rep_socket.recv_json()
            except Exception as e:
                logger.warning("cmd recv failed: %s", e)
                continue
            resp = self._handle_command(req)
            try:
                self._rep_socket.send_json(resp)
            except Exception as e:
                logger.warning("cmd send failed: %s", e)

    def _status_loop(self) -> None:
        """PUB 状态广播。"""
        while not self._shutdown_event.is_set():
            if self._pub_socket is not None:
                try:
                    self._pub_socket.send_json(self._status_dict(), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
                except Exception as e:
                    logger.warning("status pub failed: %s", e)
            time.sleep(_STATUS_PUB_INTERVAL_S)

    # ------------------------------------------------------------------
    # 指令处理
    # ------------------------------------------------------------------

    def _handle_command(self, req: dict[str, Any]) -> dict[str, Any]:
        cmd = str(req.get("cmd", "")).lower()
        try:
            # status / shutdown 始终放行（其余指令在重建/关停期间会触碰 engine 或 robot，
            # 与主线程独占的 ctx 交换竞态，故拒绝让客户端稍后重试）
            if cmd == "status":
                return self._ok("status")
            if cmd == "shutdown":
                return self._cmd_shutdown()
            if self._state in (STATE_LOADING, STATE_SHUTTING_DOWN):
                return self._err(cmd, f"server busy (state={self._state}), retry later")
            if cmd == "start":
                return self._cmd_start()
            if cmd == "stop":
                return self._cmd_stop()
            if cmd == "home":
                return self._cmd_home()
            if cmd == "load":
                return self._cmd_load(req.get("params") or {})
            return self._err(cmd, f"unknown command: {cmd}")
        except Exception as e:
            logger.exception("command '%s' failed", cmd)
            return self._err(cmd, f"handler exception: {e}")

    def _cmd_start(self) -> dict[str, Any]:
        eng = self._engine
        if eng is None:
            return self._err("start", "no engine loaded")
        if not hasattr(eng, "resume"):
            return self._err("start", "engine has no pause/resume (act_async required for start/stop)")
        if self._state == STATE_RUNNING:
            return self._ok("start", note="already running")
        if hasattr(eng, "reset"):
            eng.reset()
        eng.resume()
        self._set_state(STATE_RUNNING)
        return self._ok("start")

    def _cmd_stop(self) -> dict[str, Any]:
        eng = self._engine
        if eng is not None and hasattr(eng, "pause"):
            eng.pause()
        self._set_state(STATE_READY)
        return self._ok("stop")

    def _cmd_home(self) -> dict[str, Any]:
        eng = self._engine
        if eng is not None and hasattr(eng, "pause"):
            eng.pause()  # 停 chunk 产出，避与 home chunk 双源（[[act-async-shutdown-dual-publisher-shake]]）
        ctx = self._ctx_holder
        if ctx is None:
            return self._err("home", "no context")
        robot = ctx.hardware.robot_wrapper.inner
        self._set_state(STATE_RETURNING_HOME)
        try:
            if robot.is_connected:
                precise_sleep(_CHUNK_SETTLE_S)  # 让在飞 chunk 落桥接
                # canonical home_position（关机同款 _send_home_chunk：300Hz densify 单源、chunk_id=-1）
                robot._send_home_chunk(duration_s=2.0, fps=50.0)
                precise_sleep(2.0)  # 等 densify 执行完
            self._set_state(STATE_READY)
            return self._ok("home")
        except Exception as e:
            self._set_state(STATE_ERROR)
            return self._err("home", f"home failed: {e}")

    def _cmd_load(self, params: dict[str, Any]) -> dict[str, Any]:
        if not params.get("policy_path"):
            return self._err("load", "params.policy_path is required")
        with self._load_lock:
            self._pending_load = dict(params)
            self._set_state(STATE_LOADING)
        # 重建由主线程异步执行；即时回复，客户端轮询 status 至 READY/ERROR
        return self._ok("load", note="rebuilding in main thread; poll status")

    def _cmd_shutdown(self) -> dict[str, Any]:
        self._set_state(STATE_SHUTTING_DOWN)
        self._shutdown_event.set()
        return self._ok("shutdown", note="shutting down")


# ── 入口 ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI 入口：剥离 server 参数 → 构造 RolloutConfig → 启动服务。"""
    register_third_party_plugins()
    init_logging()
    atexit.register(_cleanup_pid_file)
    cmd_port, status_port = _consume_server_args()
    cli_args = sys.argv[1:]
    if not cli_args or not any(a.startswith("--policy.path") for a in cli_args):
        print(
            "[ERROR] --policy.path is required.\n"
            "Example: python -m lerobot.scripts.inference_server "
            "--policy.path=/ubt_IL/model/.../pretrained_model "
            "--robot.type=walker --robot.joint_config=walker_s2_10d "
            "--inference.type=act_async --inference.act_async.inference_hz=2 --fps=15",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = build_rollout_config(cli_args)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    server = InferenceServer(cfg, cmd_port, status_port, shutdown_event)
    server.serve()


def _cleanup_pid_file() -> None:
    """Delete the PID file on graceful exit (path passed via WALKER_INFERENCE_PID_FILE env).

    Only removes the file if its content equals this process's PID (don't clobber a
    newer instance). SIGKILL/crash skips atexit -> stale file handled by the .sh
    ``_pid_is_running_server`` guard (cmdline check).
    """
    path = os.environ.get("WALKER_INFERENCE_PID_FILE")
    if not path:
        return
    try:
        with open(path) as f:
            pid = f.read().strip()
        if pid == str(os.getpid()):
            os.unlink(path)
    except FileNotFoundError:
        pass
    except Exception:
        pass


if __name__ == "__main__":
    main()
