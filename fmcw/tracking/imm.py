"""交互式多模型滤波器 (Interacting Multiple Model)。

同时运行多个运动模型的滤波器分支，通过马尔可夫概率转移矩阵
实现分支间交互，输出概率加权融合状态。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


# ── 状态空间转换函数 ──────────────────────────────────────────────────

def cv_to_ca(x_cv: np.ndarray) -> np.ndarray:
    """CV [x,y,vx,vy] → CA [x,y,vx,vy,ax,ay]。"""
    x_ca = np.zeros(6)
    x_ca[:4] = x_cv[:4]
    return x_ca


def ca_to_cv(x_ca: np.ndarray) -> np.ndarray:
    """CA [x,y,vx,vy,ax,ay] → CV [x,y,vx,vy]。"""
    return x_ca[:4].copy()


def cv_to_ctra(x_cv: np.ndarray) -> np.ndarray:
    """CV [x,y,vx,vy] → CTRA [x,y,yaw,v,a,ω]。"""
    x_ctra = np.zeros(6)
    x_ctra[0] = x_cv[0]
    x_ctra[1] = x_cv[1]
    vx, vy = x_cv[2], x_cv[3]
    v = np.sqrt(vx**2 + vy**2)
    x_ctra[2] = np.rad2deg(np.arctan2(vy, vx)) if v > 1e-6 else 0.0
    x_ctra[3] = v
    return x_ctra


def ctra_to_cv(x_ctra: np.ndarray) -> np.ndarray:
    """CTRA [x,y,yaw,v,a,ω] → CV [x,y,vx,vy]。"""
    x_cv = np.zeros(4)
    x_cv[0] = x_ctra[0]
    x_cv[1] = x_ctra[1]
    yaw = np.deg2rad(x_ctra[2])
    v = x_ctra[3]
    x_cv[2] = v * np.cos(yaw)
    x_cv[3] = v * np.sin(yaw)
    return x_cv


def ca_to_ctra(x_ca: np.ndarray) -> np.ndarray:
    """CA [x,y,vx,vy,ax,ay] → CTRA [x,y,yaw,v,a,ω]。"""
    x_ctra = np.zeros(6)
    x_ctra[0] = x_ca[0]
    x_ctra[1] = x_ca[1]
    vx, vy = x_ca[2], x_ca[3]
    v = np.sqrt(vx**2 + vy**2)
    x_ctra[2] = np.rad2deg(np.arctan2(vy, vx)) if v > 1e-6 else 0.0
    x_ctra[3] = v
    # 沿航向的加速度
    x_ctra[4] = (vx * x_ca[4] + vy * x_ca[5]) / (v + 1e-10)
    return x_ctra


def ctra_to_ca(x_ctra: np.ndarray) -> np.ndarray:
    """CTRA [x,y,yaw,v,a,ω] → CA [x,y,vx,vy,ax,ay]。"""
    x_ca = np.zeros(6)
    x_ca[0] = x_ctra[0]
    x_ca[1] = x_ctra[1]
    yaw = np.deg2rad(x_ctra[2])
    v = x_ctra[3]
    x_ca[2] = v * np.cos(yaw)
    x_ca[3] = v * np.sin(yaw)
    a = x_ctra[4]
    x_ca[4] = a * np.cos(yaw)
    x_ca[5] = a * np.sin(yaw)
    return x_ca


# ── IMM 核心 ──────────────────────────────────────────────────────────────

_DEFAULT_TRANS_MATRIX = np.array([
    [0.85, 0.10, 0.05],   # CV → CV, CA, CTRA
    [0.10, 0.85, 0.05],   # CA → CV, CA, CTRA
    [0.05, 0.10, 0.85],   # CTRA → CV, CA, CTRA
])


@dataclass
class IMMBranch:
    """IMM 单分支滤波器容器。"""
    name: str
    filter: object


def _convert_cov(cov: np.ndarray, from_dim: int, to_dim: int) -> np.ndarray:
    """协方差矩阵维度转换（扩展或收缩）。

    Args:
        cov: 原始协方差 (from_dim, from_dim)
        from_dim: 源维度
        to_dim: 目标维度

    Returns:
        转换后协方差 (to_dim, to_dim)
    """
    if from_dim == to_dim:
        return cov
    result = np.zeros((to_dim, to_dim))
    min_dim = min(from_dim, to_dim)
    result[:min_dim, :min_dim] = cov[:min_dim, :min_dim]
    if to_dim > from_dim:
        # 扩展：新增维度用大初始不确定性
        for i in range(from_dim, to_dim):
            result[i, i] = 100.0
    return result


class InteractingMultipleModel:
    """交互式多模型滤波器。

    Usage:
        imm = InteractingMultipleModel(
            branches=[
                ("cv", ekf_cv_instance),
                ("ca", ekf_ca_instance),
                ("ctra", ukf_ctra_instance),
            ],
        )
        imm.predict()
        imm.update(np.array([range_m, angle_deg]))
        x_fused = imm.position  # 融合后 [x, y]
    """

    def __init__(
        self,
        branches: List[Tuple[str, object]],
        trans_matrix: Optional[np.ndarray] = None,
        init_probs: Optional[np.ndarray] = None,
    ):
        self.names = [b[0] for b in branches]
        self.branches = {b[0]: IMMBranch(name=b[0], filter=b[1]) for b in branches}
        self.n = len(branches)
        self.trans_matrix = (trans_matrix if trans_matrix is not None
                            else _DEFAULT_TRANS_MATRIX[:self.n, :self.n])
        self.probs = (init_probs[:self.n] if init_probs is not None
                     else np.full(self.n, 1.0 / self.n))

    # ── 公共接口 ──────────────────────────────────────────────────────

    def init(self, z: np.ndarray):
        """从首次测量初始化所有分支。"""
        for br in self.branches.values():
            if hasattr(br.filter, 'init'):
                br.filter.init(z)

    def predict(self):
        """混合 → 各分支预测。"""
        self._mix()
        for br in self.branches.values():
            br.filter.predict()

    def update(self, z: np.ndarray):
        """更新 → 似然 → 概率更新。

        Args:
            z: [range, angle] 测量值。
        """
        likelihoods = np.zeros(self.n)
        for i, name in enumerate(self.names):
            br = self.branches[name]
            br.filter.update(z)

            # 从滤波器读取缓存的创新和 S 矩阵
            if hasattr(br.filter, '_last_innovation'):
                innovation = br.filter._last_innovation
                S = br.filter._last_S
                dim = len(innovation)
                det = np.linalg.det(S)
                if det > 0:
                    try:
                        mahal = innovation @ np.linalg.solve(S, innovation)
                        likelihoods[i] = np.exp(-0.5 * mahal) / np.sqrt(
                            (2 * np.pi)**dim * max(det, 1e-30))
                    except np.linalg.LinAlgError:
                        likelihoods[i] = 1e-10
                else:
                    likelihoods[i] = 1e-10
            else:
                likelihoods[i] = 1.0

        # 概率更新
        mixed_probs = self.probs @ self.trans_matrix.T
        self.probs = likelihoods * mixed_probs
        prob_sum = self.probs.sum()
        self.probs = self.probs / (prob_sum + 1e-30)

    @property
    def position(self) -> np.ndarray:
        """融合后的 [x, y] 位置。"""
        x_fused = np.zeros(2)
        for i, name in enumerate(self.names):
            br = self.branches[name]
            x_fused += self.probs[i] * br.filter.position
        return x_fused

    @property
    def state(self) -> np.ndarray:
        return self.position.copy()

    @property
    def velocity(self) -> np.ndarray:
        """融合后的径向速度估计。"""
        v_fused = 0.0
        for i, name in enumerate(self.names):
            br = self.branches[name]
            x = br.filter.x
            if len(x) >= 4:
                pos = br.filter.position
                r = np.sqrt(pos[0]**2 + pos[1]**2)
                if r > 1e-6:
                    v_radial = (pos[0] * x[2] + pos[1] * x[3]) / r
                    v_fused += self.probs[i] * v_radial
        return v_fused

    # ── 内部方法 ──────────────────────────────────────────────────────

    def _mix(self):
        """IMM 混合步骤。"""
        # 先验概率: μ_j⁻ = Σ_i π_ij · μ_i
        mu_prior = self.probs @ self.trans_matrix.T
        mu_prior = mu_prior / (mu_prior.sum() + 1e-30)

        # 混合状态
        mixed_states = {}
        for i, name_i in enumerate(self.names):
            x_mixed = np.zeros(self.branches[name_i].filter.x.shape)
            for j, name_j in enumerate(self.names):
                br_j = self.branches[name_j]
                x_j_trans = self._convert_state(name_j, name_i, br_j.filter.x)
                weight = self.trans_matrix[j, i] * self.probs[j] / (mu_prior[i] + 1e-30)
                x_mixed += weight * x_j_trans
            mixed_states[name_i] = x_mixed

        # 混合协方差
        for i, name_i in enumerate(self.names):
            x_mixed = mixed_states[name_i]
            dim_i = len(x_mixed)
            P_mixed = np.zeros((dim_i, dim_i))
            for j, name_j in enumerate(self.names):
                br_j = self.branches[name_j]
                x_j_trans = self._convert_state(name_j, name_i, br_j.filter.x)
                # 转换协方差维度
                dim_j = len(br_j.filter.x)
                P_j_trans = _convert_cov(br_j.filter.P, dim_j, dim_i)
                weight = self.trans_matrix[j, i] * self.probs[j] / (mu_prior[i] + 1e-30)
                diff = x_j_trans - x_mixed
                P_mixed += weight * (P_j_trans + np.outer(diff, diff))

            self.branches[name_i].filter.x = x_mixed
            self.branches[name_i].filter.P = P_mixed

        self.probs = mu_prior

    def _convert_state(self, from_name: str, to_name: str, x: np.ndarray) -> np.ndarray:
        """在不同模型间转换状态向量。"""
        if from_name == to_name:
            return x.copy()

        conv_map = {
            ("cv", "ca"):   cv_to_ca,
            ("ca", "cv"):   ca_to_cv,
            ("cv", "ctra"): cv_to_ctra,
            ("ctra", "cv"): ctra_to_cv,
            ("ca", "ctra"): ca_to_ctra,
            ("ctra", "ca"): ctra_to_ca,
        }

        key = (from_name, to_name)
        if key in conv_map:
            return conv_map[key](x)

        # 无转换函数时：只取公共子空间 [x, y]，其余归零
        result = np.zeros(self.branches[to_name].filter.x.shape)
        result[:2] = x[:2]
        return result
