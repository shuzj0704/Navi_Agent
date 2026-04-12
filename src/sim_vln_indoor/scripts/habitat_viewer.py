"""
Habitat-Sim 交互式查看器
========================
键盘操作:
  W/S     - 前进/后退
  A/D     - 左移/右移
  Q/E     - 左转/右转
  R/F     - 抬头/低头
  ↑/↓     - 前进/后退 (方向键)
  ←/→     - 左转/右转 (方向键)
  1/2/3/4 - 切换视角: 前/左/右/后
  SPACE   - 保存当前截图
  ESC     - 退出

手柄操作 (检测到手柄时自动启用, Xbox/PS 布局):
  左摇杆 Y    - 前进/后退 (连续, 速度与摇杆幅度成正比)
  左摇杆 X    - 左右平移
  右摇杆 X    - 左/右转向 (连续)
  右摇杆 Y    - 抬头/低头 (连续)
  LB / RB     - 切换到左视图 / 右视图
  LT / RT     - 切换到后视图 / 前视图
  A           - 截图
  B           - 还原到前视图 + 归零俯仰
  Start / Y   - 退出
"""

import os, sys
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC_ROOT)
os.chdir(_PROJECT_ROOT)
# 强制 habitat-sim 使用 EGL device 0（而非按 CUDA ID 查找）
os.environ["HABITAT_SIM_CUSTOM_EGL_DEVICE"] = "0"
os.environ["MAGNUM_LOG"] = "verbose"
# pygame 的 SDL 后端在无显示的情况下也能初始化 joystick
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import habitat_sim
import numpy as np
import cv2
import sys
import math
from _magnum import Vector3

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

# ========== 配置 ==========
SCENE_DIR = "data/scene_data/mp3d"
WIDTH = 800
HEIGHT = 600
HFOV = 90  # 水平视场角

MOVE_STEP = 0.25   # 每步移动距离 (米)
TURN_STEP = 10.0   # 每步旋转角度 (度)
TILT_STEP = 5.0    # 每步俯仰角度 (度)

# ---- 手柄参数 ----
GP_DEADZONE    = 0.15      # 摇杆死区
GP_LIN_SPEED   = 1.5       # 线速度 m/s (满摇杆)
GP_YAW_SPEED   = 90.0      # 角速度 deg/s (满摇杆)
GP_TILT_SPEED  = 60.0      # 俯仰速度 deg/s (满摇杆)
GP_TRIGGER_THR = 0.3       # 扳机按下阈值 (轴值 -1..1)
GP_LOOP_MS     = 30        # 主循环刷新间隔 ms (手柄模式)


def make_cfg(scene_path):
    """创建 habitat-sim 配置"""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0

    # RGB 传感器
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [HEIGHT, WIDTH]
    rgb_sensor.hfov = HFOV
    rgb_sensor.position = [0.0, 1.5, 0.0]  # 眼睛高度 1.5m

    # Agent 配置
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def list_scenes():
    """列出所有可用场景"""
    scenes = []
    for name in sorted(os.listdir(SCENE_DIR)):
        glb = os.path.join(SCENE_DIR, name, f"{name}.glb")
        if os.path.exists(glb):
            scenes.append((name, glb))
    return scenes


def get_agent_state_info(agent):
    """获取 agent 当前状态信息"""
    state = agent.get_state()
    pos = state.position
    rot = state.rotation
    # 从四元数提取 yaw 角度
    # quaternion: w, x, y, z
    siny_cosp = 2.0 * (rot.w * rot.y + rot.x * rot.z)
    cosy_cosp = 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return pos, yaw


def render_info_bar(img, pos, yaw, view_name, scene_name, gamepad_name=None):
    """在图像上叠加信息"""
    h, w = img.shape[:2]
    # 半透明黑色背景
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    info = f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})  Yaw: {yaw:.0f}°  View: {view_name}  Scene: {scene_name}"
    cv2.putText(img, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 底部操作提示
    cv2.rectangle(overlay, (0, h - 25), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    if gamepad_name:
        hint = (f"KB WASD/QE/RF/1-4/SPACE/ESC  |  GP[{gamepad_name[:20]}] "
                f"L-stick:move  R-stick:look  LB/RB/LT/RT:view  A:shot  B:reset  Start/Y:quit")
    else:
        hint = "WASD:Move  QE:Turn  RF:Tilt  1234:View  SPACE:Save  ESC:Quit"
    cv2.putText(img, hint, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return img


# ========== 手柄封装 ==========

class Gamepad:
    """pygame joystick 的一层轻封装, 提供按键边沿检测。"""

    def __init__(self):
        self.joy = None
        self.prev_buttons = []
        self.prev_hat = (0, 0)
        self.prev_trigger_l = False
        self.prev_trigger_r = False
        if not _HAS_PYGAME:
            return
        try:
            pygame.init()
            pygame.joystick.init()
        except Exception as e:
            print(f"[手柄] pygame 初始化失败: {e}")
            return
        n = pygame.joystick.get_count()
        if n == 0:
            return
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        self.prev_buttons = [False] * self.joy.get_numbuttons()
        print(f"[手柄] 已连接: {self.joy.get_name()} "
              f"(axes={self.joy.get_numaxes()} "
              f"buttons={self.joy.get_numbuttons()} "
              f"hats={self.joy.get_numhats()})")

    @property
    def available(self):
        return self.joy is not None

    @property
    def name(self):
        return self.joy.get_name() if self.joy else None

    def _axis(self, idx, default=0.0):
        try:
            v = self.joy.get_axis(idx)
        except Exception:
            return default
        return 0.0 if abs(v) < GP_DEADZONE else v

    def poll(self):
        """返回当前状态 dict, 同时更新边沿缓存。"""
        if self.joy is None:
            return None
        pygame.event.pump()

        # 常见 Xbox/PS 布局: 0=LX, 1=LY, 2=LT, 3=RX, 4=RY, 5=RT
        # (有的驱动会把 LT/RT 放到 2/5, 有的放到其它; 兼容性从简)
        lx = self._axis(0)
        ly = self._axis(1)
        rx = self._axis(3)
        ry = self._axis(4)

        try:
            lt_raw = self.joy.get_axis(2)
        except Exception:
            lt_raw = -1.0
        try:
            rt_raw = self.joy.get_axis(5)
        except Exception:
            rt_raw = -1.0
        # 扳机轴的静止值通常是 -1, 按下接近 +1; 归一到 0..1
        lt = max(0.0, (lt_raw + 1.0) * 0.5)
        rt = max(0.0, (rt_raw + 1.0) * 0.5)

        buttons = [bool(self.joy.get_button(i)) for i in range(self.joy.get_numbuttons())]
        pressed = [b and not p for b, p in zip(buttons, self.prev_buttons)]
        self.prev_buttons = buttons

        hat = self.joy.get_hat(0) if self.joy.get_numhats() > 0 else (0, 0)
        hat_pressed = (hat != self.prev_hat) and hat != (0, 0)
        self.prev_hat = hat

        lt_edge = lt > GP_TRIGGER_THR and not self.prev_trigger_l
        rt_edge = rt > GP_TRIGGER_THR and not self.prev_trigger_r
        self.prev_trigger_l = lt > GP_TRIGGER_THR
        self.prev_trigger_r = rt > GP_TRIGGER_THR

        return {
            "lx": lx, "ly": ly, "rx": rx, "ry": ry,
            "lt": lt, "rt": rt,
            "buttons": buttons, "pressed": pressed,
            "hat": hat, "hat_pressed": hat_pressed,
            "lt_edge": lt_edge, "rt_edge": rt_edge,
        }

    def shutdown(self):
        if self.joy:
            try:
                self.joy.quit()
            except Exception:
                pass
        if _HAS_PYGAME:
            try:
                pygame.joystick.quit()
                pygame.quit()
            except Exception:
                pass


def main():
    # 选择场景
    scenes = list_scenes()
    if not scenes:
        print(f"在 {SCENE_DIR} 下没有找到场景文件")
        sys.exit(1)

    print("\n可用场景:")
    for i, (name, path) in enumerate(scenes):
        print(f"  [{i}] {name}")

    choice = input(f"\n选择场景编号 [0-{len(scenes)-1}], 直接回车选第一个: ").strip()
    idx = int(choice) if choice else 0
    scene_name, scene_path = scenes[idx]
    print(f"\n加载场景: {scene_name} ...")

    # 初始化仿真器
    cfg = make_cfg(scene_path)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.get_agent(0)

    # 设置初始位置（导航网格上的随机点）
    navmesh_path = scene_path.replace(".glb", ".navmesh")
    if os.path.exists(navmesh_path):
        sim.pathfinder.load_nav_mesh(navmesh_path)
        init_pos = sim.pathfinder.get_random_navigable_point()
        agent_state = agent.get_state()
        agent_state.position = init_pos
        agent.set_state(agent_state)
        print(f"初始位置: {init_pos}")

    # 视角偏移: 前/左/右/后
    view_offsets = {
        "front": 0,
        "left": 90,
        "right": -90,
        "back": 180,
    }
    view_cycle = ["front", "left", "back", "right"]
    current_view = "front"
    tilt_angle = 0.0  # 俯仰角
    screenshot_count = 0

    # 手柄
    pad = Gamepad()
    gp_mode = pad.available
    loop_delay = GP_LOOP_MS if gp_mode else 0  # 手柄模式下非阻塞, 键盘模式下阻塞
    dt_sec = loop_delay / 1000.0 if loop_delay > 0 else 0.05

    # 手柄模式下需要 numpy-quaternion 来平滑更新 yaw
    try:
        import quaternion as _npq  # habitat-sim 依赖, 正常都在
    except ImportError:
        _npq = None
        if gp_mode:
            print("[手柄] 未找到 numpy-quaternion, 手柄转向退化为离散 turn_left/turn_right")

    def _apply_yaw_delta(agent, dyaw_deg):
        st = agent.get_state()
        if _npq is not None:
            dq = _npq.from_rotation_vector(
                np.array([0.0, math.radians(dyaw_deg), 0.0], dtype=float)
            )
            st.rotation = dq * st.rotation
            agent.set_state(st)
        else:
            # 退化: 按 TURN_STEP 整数倍离散转
            n = int(abs(dyaw_deg) / TURN_STEP)
            act = "turn_left" if dyaw_deg > 0 else "turn_right"
            for _ in range(n):
                agent.act(act)

    print(f"仿真器已启动 ({'手柄+键盘' if gp_mode else '仅键盘'})...")

    cv2.namedWindow("Habitat Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Habitat Viewer", WIDTH, HEIGHT)

    while True:
        # 获取当前状态
        state = agent.get_state()
        pos, yaw = get_agent_state_info(agent)

        # 计算视角旋转 (yaw + view offset + tilt)
        view_yaw = yaw + view_offsets[current_view]
        view_yaw_rad = math.radians(view_yaw)
        tilt_rad = math.radians(tilt_angle)

        # 设置传感器朝向: orientation = [pitch, yaw, roll] 弧度
        sensor = agent._sensors["rgb"]
        spec = sensor.specification()
        spec.orientation = Vector3(tilt_rad, math.radians(view_offsets[current_view]), 0.0)
        sensor.set_transformation_from_spec()

        # 渲染
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]  # RGBA → RGB
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 叠加信息
        display = render_info_bar(
            rgb_bgr.copy(), pos, yaw, current_view, scene_name,
            gamepad_name=pad.name if gp_mode else None,
        )
        cv2.imshow("Habitat Viewer", display)

        # ===== 手柄轮询 (先处理手柄, 再读键盘) =====
        if gp_mode:
            gp = pad.poll()
            if gp is not None:
                # --- 连续量: 左摇杆平移 + 右摇杆转向/俯仰 ---
                # 注意: 摇杆 Y 向上为 -1
                if gp["ly"] != 0.0:
                    dd = -gp["ly"] * GP_LIN_SPEED * dt_sec
                    forward = np.array([
                        -math.sin(math.radians(yaw)),
                        0.0,
                        -math.cos(math.radians(yaw)),
                    ])
                    state = agent.get_state()
                    state.position = state.position + forward * dd
                    agent.set_state(state)

                if gp["lx"] != 0.0:
                    dd = gp["lx"] * GP_LIN_SPEED * dt_sec
                    right = np.array([
                        math.cos(math.radians(yaw)),
                        0.0,
                        -math.sin(math.radians(yaw)),
                    ])
                    state = agent.get_state()
                    state.position = state.position + right * dd
                    agent.set_state(state)

                if gp["rx"] != 0.0:
                    # 右摇杆右 = 右转 → yaw 减小 (habitat-sim 惯例)
                    dyaw = -gp["rx"] * GP_YAW_SPEED * dt_sec
                    _apply_yaw_delta(agent, dyaw)

                if gp["ry"] != 0.0:
                    # 右摇杆上 (ry<0) = 抬头 (tilt 减小)
                    tilt_angle = max(
                        -60.0,
                        min(60.0, tilt_angle + gp["ry"] * GP_TILT_SPEED * dt_sec),
                    )

                # --- 按键边沿 ---
                btns = gp["pressed"]
                def _btn(i):
                    return i < len(btns) and btns[i]

                # A: 截图
                if _btn(0):
                    save_path = f"output/screenshot_{screenshot_count:03d}.png"
                    cv2.imwrite(save_path, rgb_bgr)
                    print(f"截图已保存: {save_path}")
                    screenshot_count += 1
                # B: 回到前视 + 归零俯仰
                if _btn(1):
                    current_view = "front"
                    tilt_angle = 0.0
                # Y 或 Start 退出
                if _btn(3) or _btn(7):
                    break
                # LB / RB: 切换视角 (按 view_cycle 向左/右走一步)
                if _btn(4):
                    idx = view_cycle.index(current_view)
                    current_view = view_cycle[(idx - 1) % len(view_cycle)]
                    tilt_angle = 0.0
                if _btn(5):
                    idx = view_cycle.index(current_view)
                    current_view = view_cycle[(idx + 1) % len(view_cycle)]
                    tilt_angle = 0.0
                # LT / RT 边沿: 切到后视 / 前视
                if gp["lt_edge"]:
                    current_view = "back"
                    tilt_angle = 0.0
                if gp["rt_edge"]:
                    current_view = "front"
                    tilt_angle = 0.0
                # D-pad: 离散左右转、前/后视
                if gp["hat_pressed"]:
                    hx, hy = gp["hat"]
                    if hx == -1:
                        agent.act("turn_left")
                    elif hx == 1:
                        agent.act("turn_right")
                    if hy == 1:
                        current_view = "front"
                        tilt_angle = 0.0
                    elif hy == -1:
                        current_view = "back"
                        tilt_angle = 0.0

        # ===== 键盘 =====
        key = cv2.waitKey(loop_delay) & 0xFF
        if key == 255:
            # 手柄模式下没有键盘输入时直接进入下一帧
            continue

        if key == 27:  # ESC
            break

        elif key == ord('w') or key == 82:  # W 或 ↑
            agent.act("move_forward")

        elif key == ord('s') or key == 84:  # S 或 ↓
            agent.act("move_backward") if hasattr(agent, 'move_backward') else None
            # move_backward 可能不存在，手动后退
            state = agent.get_state()
            forward = np.array([
                -math.sin(math.radians(yaw)),
                0,
                -math.cos(math.radians(yaw))
            ])
            state.position = state.position - forward * MOVE_STEP
            agent.set_state(state)

        elif key == ord('a'):  # A - 左平移
            state = agent.get_state()
            right = np.array([
                math.cos(math.radians(yaw)),
                0,
                -math.sin(math.radians(yaw))
            ])
            state.position = state.position - right * MOVE_STEP
            agent.set_state(state)

        elif key == ord('d'):  # D - 右平移
            state = agent.get_state()
            right = np.array([
                math.cos(math.radians(yaw)),
                0,
                -math.sin(math.radians(yaw))
            ])
            state.position = state.position + right * MOVE_STEP
            agent.set_state(state)

        elif key == ord('q') or key == 81:  # Q 或 ←
            agent.act("turn_left")

        elif key == ord('e') or key == 83:  # E 或 →
            agent.act("turn_right")

        elif key == ord('r'):  # R - 抬头
            tilt_angle = max(tilt_angle - TILT_STEP, -60)

        elif key == ord('f'):  # F - 低头
            tilt_angle = min(tilt_angle + TILT_STEP, 60)

        elif key == ord('1'):
            current_view = "front"
            tilt_angle = 0
        elif key == ord('2'):
            current_view = "left"
            tilt_angle = 0
        elif key == ord('3'):
            current_view = "right"
            tilt_angle = 0
        elif key == ord('4'):
            current_view = "back"
            tilt_angle = 0

        elif key == ord(' '):  # SPACE - 截图
            save_path = f"output/screenshot_{screenshot_count:03d}.png"
            cv2.imwrite(save_path, rgb_bgr)
            print(f"截图已保存: {save_path}")
            screenshot_count += 1

    pad.shutdown()
    sim.close()
    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()
