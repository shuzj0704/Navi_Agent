"""
DWA (Dynamic Window Approach) 局部路径规划器
==========================================
全 numpy 向量化实现，KDTree 加速障碍物查询。
仿真阶段使用此版本，真机部署替换为 C++ ROS2 节点。

坐标系: 机器人系 X=前 Y=右, omega>0=右转
所有 cost 归一化到 [0,1] 后加权求和。
"""

import numpy as np
from scipy.spatial import KDTree


# ========== 可调参数 (直接改这里) ==========

# 运动学约束
MAX_SPEED       = 0.5       # 最大线速度 m/s
MAX_YAW_RATE    = 1.0       # 最大角速度 rad/s
MAX_ACCEL       = 20.0      # 最大线加速度 m/s² (大值适配 Habitat 离散动作)
MAX_YAW_ACCEL   = 20.0       # 最大角加速度 rad/s²
V_RESO          = 0.02      # 线速度采样分辨率 m/s
W_RESO          = 0.02      # 角速度采样分辨率 rad/s

# 规划参数
DT              = 0.1       # 控制周期 s
PREDICT_TIME    = 2.0       # 前向预测时间 s
ROBOT_RADIUS    = 0.0       # 碰撞半径 m
OBSTACLE_RANGE  = 5.0       # 障碍物影响范围 m
IGNORE_START_DIST = 0.1     # 轨迹起始 N 米内不参与障碍物判断 (忽略机器人自身附近的噪点)

# 代价权重 (归一化后的加权系数, 总和不需要为1)
W_GOAL          = 2.0       # 目标距离代价权重
W_OBSTACLE      = 0.5       # 障碍物代价权重
W_SPEED         = 1.5       # 速度代价权重 (鼓励快速前进)
W_SMOOTH_V      = 0.3       # 线速度平滑代价权重 (抑制加减速抖动)
W_SMOOTH_W      = 0.3       # 角速度平滑代价权重 (抑制转向抖动)

# ===========================================


class DWAPlanner:
    def __init__(self, **kwargs):
        self.max_speed     = kwargs.get("max_speed", MAX_SPEED)
        self.max_yaw_rate  = kwargs.get("max_yaw_rate", MAX_YAW_RATE)
        self.max_accel     = kwargs.get("max_accel", MAX_ACCEL)
        self.max_yaw_accel = kwargs.get("max_yaw_accel", MAX_YAW_ACCEL)
        self.v_reso        = kwargs.get("v_reso", V_RESO)
        self.w_reso        = kwargs.get("w_reso", W_RESO)

        self.dt            = kwargs.get("dt", DT)
        self.predict_time  = kwargs.get("predict_time", PREDICT_TIME)
        self.robot_radius  = kwargs.get("robot_radius", ROBOT_RADIUS)
        self.obstacle_range = kwargs.get("obstacle_range", OBSTACLE_RANGE)
        self.ignore_start_dist = kwargs.get("ignore_start_dist", IGNORE_START_DIST)

        self.w_goal       = kwargs.get("w_goal", W_GOAL)
        self.w_obstacle   = kwargs.get("w_obstacle", W_OBSTACLE)
        self.w_speed      = kwargs.get("w_speed", W_SPEED)
        self.w_smooth_v   = kwargs.get("w_smooth_v", W_SMOOTH_V)
        self.w_smooth_w   = kwargs.get("w_smooth_w", W_SMOOTH_W)

        # 上一次输出的速度指令 (用于平滑代价)
        self.last_v = 0.0
        self.last_w = 0.0

    def plan(self, state, goal, obstacles):
        """
        参数:
            state:     [x, y, yaw, v, omega]
            goal:      [gx, gy]
            obstacles: (N, 2) 障碍物点
        返回:
            (v, omega) 最优速度指令
        """
        v, w, _ = self._plan_core(state, goal, obstacles)
        self.last_v, self.last_w = v, w
        return v, w

    def plan_debug(self, state, goal, obstacles):
        """与 plan() 相同逻辑，额外返回所有中间数据供可视化。"""
        v, w, debug = self._plan_core(state, goal, obstacles)
        self.last_v, self.last_w = v, w
        return v, w, debug

    def _plan_core(self, state, goal, obstacles):
        """核心规划逻辑，返回 (v, w, debug_or_None)"""
        dw = self._dynamic_window(state)
        vs = np.arange(dw[0], dw[1] + 1e-6, self.v_reso)
        ws = np.arange(dw[2], dw[3] + 1e-6, self.w_reso)
        V, W = np.meshgrid(vs, ws)
        candidates = np.stack([V.ravel(), W.ravel()], axis=-1)

        if len(candidates) == 0:
            return 0.0, 0.0, None

        # 模拟轨迹
        trajs = self._simulate_all(state, candidates)

        # --- 原始代价 ---
        goal_costs_raw   = self._goal_cost(trajs, goal)
        obs_costs_raw    = self._obstacle_cost(trajs, obstacles, state)
        speed_costs_raw  = self.max_speed - candidates[:, 0]   # 越快越好 → 剩余速度越小越好
        smooth_v_raw     = np.abs(candidates[:, 0] - self.last_v)
        smooth_w_raw     = np.abs(candidates[:, 1] - self.last_w)

        # 碰撞标记 (obstacle cost = inf 的)
        collision_mask = obs_costs_raw == np.inf

        # --- 归一化到 [0, 1] (仅对非碰撞候选) ---
        goal_costs_n   = self._normalize(goal_costs_raw, collision_mask)
        obs_costs_n    = self._normalize(obs_costs_raw, collision_mask)
        speed_costs_n  = self._normalize(speed_costs_raw, collision_mask)
        smooth_v_n     = self._normalize(smooth_v_raw, collision_mask)
        smooth_w_n     = self._normalize(smooth_w_raw, collision_mask)

        # --- 加权总代价 ---
        total = (self.w_goal     * goal_costs_n +
                 self.w_obstacle * obs_costs_n +
                 self.w_speed    * speed_costs_n +
                 self.w_smooth_v * smooth_v_n +
                 self.w_smooth_w * smooth_w_n)

        # 碰撞候选设为 inf
        total[collision_mask] = np.inf

        best = np.argmin(total)
        all_inf = total[best] == np.inf

        debug = {
            "dw": dw,
            "candidates": candidates,
            "trajs": trajs,
            "goal_costs": goal_costs_n,
            "obs_costs": obs_costs_n,
            "speed_costs": speed_costs_n,
            "smooth_v_costs": smooth_v_n,
            "smooth_w_costs": smooth_w_n,
            "total_costs": total,
            "best_idx": best,
            "collision_mask": collision_mask,
            "best_traj": trajs[best],
            "n_candidates": len(candidates),
            "n_collision": int(collision_mask.sum()),
            "n_valid": int((~collision_mask).sum()),
            "state": state,
            "goal": goal,
            "n_obstacles": len(obstacles),
            "last_v": self.last_v,
            "last_w": self.last_w,
        }

        if all_inf:
            return 0.0, 0.0, debug
        return float(candidates[best, 0]), float(candidates[best, 1]), debug

    @staticmethod
    def _normalize(costs, collision_mask):
        """将代价数组归一化到 [0, 1]，仅基于非碰撞候选的 min/max"""
        valid = ~collision_mask
        if valid.sum() == 0:
            return np.zeros_like(costs)
        valid_costs = costs[valid]
        c_min = valid_costs.min()
        c_max = valid_costs.max()
        if c_max - c_min < 1e-8:
            return np.zeros_like(costs)
        normed = (costs - c_min) / (c_max - c_min)
        normed = np.clip(normed, 0.0, 1.0)
        return normed

    def _dynamic_window(self, state):
        v, w = state[3], state[4]
        return [
            max(0.0, v - self.max_accel * self.dt),
            min(self.max_speed, v + self.max_accel * self.dt),
            max(-self.max_yaw_rate, w - self.max_yaw_accel * self.dt),
            min(self.max_yaw_rate, w + self.max_yaw_accel * self.dt),
        ]

    def _simulate_all(self, state, candidates):
        M = len(candidates)
        T = max(1, int(self.predict_time / self.dt))
        v = candidates[:, 0]
        w = candidates[:, 1]
        x   = np.full(M, state[0])
        y   = np.full(M, state[1])
        yaw = np.full(M, state[2])
        traj = np.zeros((M, T, 3))
        for t in range(T):
            x   = x + v * np.cos(yaw) * self.dt
            y   = y + v * np.sin(yaw) * self.dt
            yaw = yaw + w * self.dt
            traj[:, t, 0] = x
            traj[:, t, 1] = y
            traj[:, t, 2] = yaw
        return traj

    def _goal_cost(self, trajs, goal):
        end_xy = trajs[:, -1, :2]
        return np.hypot(end_xy[:, 0] - goal[0], end_xy[:, 1] - goal[1])

    def _obstacle_cost(self, trajs, obstacles, state):
        """对每条候选轨迹计算障碍物代价。

        起点 `ignore_start_dist` 米范围内的障碍物整体丢弃 —— 贴身的
        噪点 / 机器人自身轮廓不参与碰撞判断, 等价于"轨迹前 N 米不做
        障碍物判断"。剩余障碍物对整条预测轨迹做标准 KDTree 最小距离
        检查。
        """
        M = len(trajs)
        if len(obstacles) == 0:
            return np.zeros(M)

        # 起点附近的障碍物视为噪点/自体轮廓直接忽略
        if self.ignore_start_dist > 0.0:
            start_xy = np.array([state[0], state[1]], dtype=float)
            d_from_start = np.hypot(
                obstacles[:, 0] - start_xy[0],
                obstacles[:, 1] - start_xy[1],
            )
            obstacles = obstacles[d_from_start >= self.ignore_start_dist]
            if len(obstacles) == 0:
                return np.zeros(M)

        T = trajs.shape[1]
        flat_xy = trajs[:, :, :2].reshape(-1, 2)
        tree = KDTree(obstacles)
        min_dists, _ = tree.query(flat_xy)
        min_dists = min_dists.reshape(M, T).min(axis=1)
        return np.where(
            min_dists < self.robot_radius, np.inf,
            np.where(min_dists < self.obstacle_range,
                     1.0 / min_dists, 0.0)
        )
