"""运动模型定义：基类 + CV + CA + CTRA。

每个模型封装状态维度、状态转移矩阵 F、过程噪声矩阵 Q，
以及非线性状态传播函数 propagate()。
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class MotionModel(ABC):
    """运动模型抽象基类。"""

    dim: int
    state_names: List[str]

    def F(self, dt: float) -> np.ndarray:
        """状态转移矩阵（线性模型）。CTRA 覆写返回 None。"""
        raise NotImplementedError

    def Q(self, dt: float, q: float) -> np.ndarray:
        """过程噪声矩阵（连续白噪声模型）。"""
        raise NotImplementedError

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        """非线性状态传播（默认 F@x，CTRA 覆写）。"""
        F = self.F(dt)
        if F is not None:
            return F @ x
        raise NotImplementedError


# ── CV Model ──────────────────────────────────────────────────────────────

class CVModel(MotionModel):
    """匀速模型 (Constant Velocity)。

    状态: [x, y, vx, vy] (dim=4)
    假设: 速度为白噪声驱动的 Wiener 过程
    """

    dim = 4
    state_names = ["x", "y", "vx", "vy"]

    def F(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def Q(self, dt: float, q: float) -> np.ndarray:
        """连续白噪声加速度模型。

        q: 加速度噪声标准差 (m/s²)
        参考: Bar-Shalom, "Estimation with Applications", Eq 6.3.2-2
        """
        dt2 = dt ** 2
        dt3 = dt ** 3
        q2 = q ** 2
        return q2 * np.array([
            [dt3/3, 0,    dt2/2, 0   ],
            [0,    dt3/3, 0,    dt2/2],
            [dt2/2, 0,    dt,   0   ],
            [0,    dt2/2, 0,    dt   ],
        ])


# ── CA Model ──────────────────────────────────────────────────────────────

class CAModel(MotionModel):
    """匀加速模型 (Constant Acceleration)。

    状态: [x, y, vx, vy, ax, ay] (dim=6)
    假设: jerk (加速度导数) 为白噪声驱动的 Wiener 过程
    """

    dim = 6
    state_names = ["x", "y", "vx", "vy", "ax", "ay"]

    def F(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 2] = dt
        F[0, 4] = 0.5 * dt ** 2
        F[1, 3] = dt
        F[1, 5] = 0.5 * dt ** 2
        F[2, 4] = dt
        F[3, 5] = dt
        return F

    def Q(self, dt: float, q: float) -> np.ndarray:
        """连续白噪声 jerk 模型。

        q: jerk 噪声标准差 (m/s³)
        """
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        dt5 = dt ** 5
        q2 = q ** 2
        Q = np.zeros((6, 6))
        # x, vx, ax 块
        Q[0, 0] = dt5 / 20
        Q[0, 2] = dt4 / 8
        Q[0, 4] = dt3 / 6
        Q[2, 2] = dt3 / 3
        Q[2, 4] = dt2 / 2
        Q[4, 4] = dt
        # y, vy, ay 块 (同 x 块)
        Q[1, 1] = dt5 / 20
        Q[1, 3] = dt4 / 8
        Q[1, 5] = dt3 / 6
        Q[3, 3] = dt3 / 3
        Q[3, 5] = dt2 / 2
        Q[5, 5] = dt
        # 对称化
        Q[2, 0] = Q[0, 2]
        Q[4, 0] = Q[0, 4]
        Q[4, 2] = Q[2, 4]
        Q[3, 1] = Q[1, 3]
        Q[5, 1] = Q[1, 5]
        Q[5, 3] = Q[3, 5]
        return q2 * Q


# ── CTRA Model ────────────────────────────────────────────────────────────

class CTRAModel(MotionModel):
    """协同转弯率与加速度模型 (Constant Turn Rate and Acceleration)。

    状态: [x, y, yaw, v, a, ω] (dim=6)
          x, y   : 位置 (m)
          yaw    : 航向角 (deg), 0°=x轴正方向
          v      : 速度 (m/s), 沿航向方向
          a      : 加速度 (m/s²), 沿航向方向
          ω      : 转弯率 (deg/s)

    传播函数 (非线性自行车模型):
        当 ω ≈ 0: 直线运动
            x' = x + v·cos(yaw)·dt
            y' = y + v·sin(yaw)·dt
            yaw' = yaw
        当 ω ≠ 0: 圆弧运动
            x' = x + v/ω·[sin(yaw+ω·dt) - sin(yaw)]
            y' = y + v/ω·[cos(yaw) - cos(yaw+ω·dt)]
            yaw' = yaw + ω·dt
        v' = v + a·dt
        a' = a
        ω' = ω
    """

    dim = 6
    state_names = ["x", "y", "yaw", "v", "a", "omega"]

    def __init__(self, deg=True):
        """deg=True 时角度使用度，False 时使用弧度。"""
        self.deg = deg

    def F(self, dt: float) -> Optional[np.ndarray]:
        """CTRA 无线性转移矩阵，返回 None。"""
        return None

    def Q(self, dt: float, q: float) -> np.ndarray:
        """CTRA 过程噪声（对角近似）。

        分别对 v, a, ω 加独立的 Wiener 过程噪声。
        """
        q2 = q ** 2
        Q = np.zeros((6, 6))
        Q[3, 3] = q2 * dt          # v 噪声
        Q[4, 4] = q2 * dt          # a 噪声
        Q[5, 5] = (q2 * 5.0) * dt  # ω 噪声（允许稍大）
        return Q

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        """非线性 CTRA 状态传播。"""
        x_new = x.copy()
        yaw = np.deg2rad(x[2]) if self.deg else x[2]
        v = x[3]
        a = x[4]
        omega = np.deg2rad(x[5]) if self.deg else x[5]

        if abs(omega) < 1e-6:
            # 直线运动
            x_new[0] = x[0] + v * np.cos(yaw) * dt
            x_new[1] = x[1] + v * np.sin(yaw) * dt
        else:
            # 圆弧运动
            x_new[0] = x[0] + v / omega * (np.sin(yaw + omega * dt) - np.sin(yaw))
            x_new[1] = x[1] + v / omega * (np.cos(yaw) - np.cos(yaw + omega * dt))
            x_new[2] = x[2] + x[5] * dt  # 保持角度单位一致

        x_new[3] = v + a * dt
        # a, ω 不变（假定恒定）
        return x_new
