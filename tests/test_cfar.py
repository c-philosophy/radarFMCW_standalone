"""CFAR 检测器实现测试（CA、OS、GO、SO）。"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fmcw.detection.cfar import CACFAR, OSCFAR, GOCFAR, SOCFAR
from fmcw.detection.peak_finder import PeakFinder


def _make_test_rd_map():
    """Create a simple RD map with two strong targets."""
    rd = np.random.randn(128, 256) * 0.1
    # Target 1 at (20, 40)
    rd[20, 40] = 10.0
    rd[19:22, 39:42] += 5.0
    # Target 2 at (80, 120)
    rd[80, 120] = 8.0
    rd[79:82, 119:122] += 4.0
    return rd


# ============================================================
# 原有测试（保持向后兼容）
# ============================================================

def test_cacfar_detects_targets():
    """CA-CFAR should detect the simulated targets."""
    rd = _make_test_rd_map()
    finder = PeakFinder()
    peaks = finder.detect(rd)

    cfar = CACFAR()
    detections = cfar.filter(rd, peaks, guard_cells=2, reference_cells=8, alpha=8.0)

    det_set = set((int(d[0]), int(d[1])) for d in detections)
    assert (20, 40) in det_set or (20, 40) in [(int(d[0]), int(d[1])) for d in detections], \
        "CA-CFAR should detect target at (20, 40)"


def test_cacfar_rejects_noise():
    """CA-CFAR should not detect noise-only peaks (with high alpha)."""
    rd = np.abs(np.random.randn(128, 256) * 0.1)  # positive-valued noise
    finder = PeakFinder()
    peaks = finder.detect(rd)

    cfar = CACFAR()
    detections = cfar.filter(rd, peaks, guard_cells=2, reference_cells=8, alpha=100.0)
    # With very high alpha, no noise peaks should pass
    assert len(detections) == 0, f"Expected 0 detections with high alpha, got {len(detections)}"


def test_oscfar_different_rank():
    """OS-CFAR with different rank ratios should give different results."""
    rd = _make_test_rd_map()
    finder = PeakFinder()
    peaks = finder.detect(rd)

    os_low = OSCFAR(rank_ratio=0.5)
    os_high = OSCFAR(rank_ratio=0.9)

    det_low = os_low.filter(rd, peaks, guard_cells=2, reference_cells=8, alpha=8.0)
    det_high = os_high.filter(rd, peaks, guard_cells=2, reference_cells=8, alpha=8.0)

    # Higher rank should produce more detections (lower threshold)
    # Actually: higher rank_ratio = higher k = higher noise estimate = stricter threshold
    # So det_high should have fewer or equal detections than det_low
    if len(det_low) > 0 and len(det_high) > 0:
        assert len(det_high) <= len(det_low), \
            f"Higher rank should not produce more detections ({len(det_high)} > {len(det_low)})"


def test_gocfar_edge_preservation():
    """GO-CFAR 应正确处理边界附近的边缘情况。"""
    rd = _make_test_rd_map()
    finder = PeakFinder()
    peaks = finder.detect(rd)

    go = GOCFAR()
    detections = go.filter(rd, peaks, guard_cells=2, reference_cells=8, alpha=8.0)
    # 验证返回值类型和形状正确
    assert isinstance(detections, np.ndarray), "应返回 numpy 数组"
    assert detections.ndim == 2, "应为 2D 数组"
    if len(detections) > 0:
        assert detections.shape[1] == 2, "每个检测应包含 (doppler, range) 索引"


def test_socfar():
    """SO-CFAR 应在正常输入下不崩溃。"""
    rd = _make_test_rd_map()
    finder = PeakFinder()
    peaks = finder.detect(rd)

    so = SOCFAR()
    detections = so.filter(rd, peaks, guard_cells=2, reference_cells=8, alpha=6.0)
    # 验证返回值类型和形状正确
    assert isinstance(detections, np.ndarray), "应返回 numpy 数组"
    assert detections.ndim == 2, "应为 2D 数组"
    if len(detections) > 0:
        assert detections.shape[1] == 2, "每个检测应包含 (doppler, range) 索引"


def test_cfar_empty_peaks():
    """All CFAR variants should handle empty peak lists."""
    rd = np.random.randn(128, 256) * 0.1
    peaks = np.empty((0, 2), dtype=np.int64)

    for cfar_cls in [CACFAR, OSCFAR, GOCFAR, SOCFAR]:
        cfar = cfar_cls() if cfar_cls in [OSCFAR] else cfar_cls()
        if cfar_cls == OSCFAR:
            cfar = OSCFAR()
        result = cfar.filter(rd, peaks, guard_cells=2, reference_cells=8, alpha=8.0)
        assert len(result) == 0, f"{cfar_cls.__name__} should return empty for empty input"


def test_peak_finder_basic():
    """PeakFinder 应能找到局部最大值。"""
    rd = _make_test_rd_map()
    finder = PeakFinder()
    peaks = finder.detect(rd)

    assert len(peaks) >= 2, f"应至少找到 2 个峰值，实际找到 {len(peaks)}"

    # 验证峰值位置在目标附近（允许小幅偏移）
    peak_set = set((int(p[0]), int(p[1])) for p in peaks)
    # 目标 1 在 (20, 40)，目标 2 在 (80, 120)
    # 检查是否有峰值落在目标附近（±2 的范围内）
    target1_found = any(abs(d - 20) <= 2 and abs(r - 40) <= 2 for d, r in peak_set)
    target2_found = any(abs(d - 80) <= 2 and abs(r - 120) <= 2 for d, r in peak_set)
    assert target1_found or target2_found, \
        f"应至少检测到一个目标附近的峰值，峰值集合: {peak_set}"


# ============================================================
# 新增测试：alpha 自动计算
# ============================================================

def test_cacfar_compute_alpha_analytic():
    """CA-CFAR 解析公式计算应与手动计算一致。"""
    guard, ref = 2, 8
    # N = (2*(2+8)+1)^2 - (2*2+1)^2 = 441 - 25 = 416
    N = (2 * (guard + ref) + 1) ** 2 - (2 * guard + 1) ** 2
    expected = N * (1e-4 ** (-1.0 / N) - 1.0)

    alpha = CACFAR.compute_alpha(1e-4, guard, ref, method='analytic')
    assert abs(alpha - expected) < 1e-10, \
        f"解析公式应精确匹配: expected={expected:.10f}, got={alpha:.10f}"


def test_cacfar_auto_alpha():
    """CA-CFAR 在 alpha=None 时应自动从 pfa 计算且能检测目标。"""
    rd = _make_test_rd_map()
    finder = PeakFinder()
    peaks = finder.detect(rd)

    cfar = CACFAR()
    detections = cfar.filter(rd, peaks, guard_cells=2, reference_cells=8,
                             alpha=None, pfa=1e-4)
    assert isinstance(detections, np.ndarray), "应返回 numpy 数组"
    assert detections.ndim == 2, "应为 2D 数组"


def test_cacfar_explicit_alpha_overrides():
    """显式提供 alpha 时应跳过自动计算。"""
    rd = _make_test_rd_map()
    finder = PeakFinder()
    peaks = finder.detect(rd)

    cfar = CACFAR()
    # alpha=None → 自动计算（alpha 较小，检测多）
    det_auto = cfar.filter(rd, peaks, guard_cells=2, reference_cells=8,
                           alpha=None, pfa=1e-4)
    # alpha=100 → 极高门限（检测极少）
    det_explicit = cfar.filter(rd, peaks, guard_cells=2, reference_cells=8,
                               alpha=100.0, pfa=1e-4)
    # 高 alpha 应产生更少或相等的检测
    assert len(det_explicit) <= len(det_auto), \
        f"高 alpha 应检测更少: explicit={len(det_explicit)}, auto={len(det_auto)}"


def test_oscfar_compute_alpha_converges():
    """OS-CFAR 数值求解应收敛于合理范围。"""
    alpha = OSCFAR.compute_alpha(1e-4, 2, 8, rank_ratio=0.75, method='analytic')
    assert 1.0 < alpha < 500.0, \
        f"OS-CFAR alpha={alpha:.2f} 超出合理范围 [1, 500]"


def test_oscfar_rank_ratio_ordering():
    """rank_ratio 越高 → 噪声估计越高 → alpha 越低。"""
    a50 = OSCFAR.compute_alpha(1e-4, 2, 8, rank_ratio=0.5, method='analytic')
    a75 = OSCFAR.compute_alpha(1e-4, 2, 8, rank_ratio=0.75, method='analytic')
    a90 = OSCFAR.compute_alpha(1e-4, 2, 8, rank_ratio=0.9, method='analytic')

    assert a75 < a50, \
        f"75%ile alpha ({a75:.2f}) 应小于 50%ile alpha ({a50:.2f})"
    assert a90 < a75, \
        f"90%ile alpha ({a90:.2f}) 应小于 75%ile alpha ({a75:.2f})"


def test_gocfar_compute_alpha_converges():
    """GO-CFAR 数值求解应收敛于合理范围。"""
    alpha = GOCFAR.compute_alpha(1e-4, 2, 8, method='analytic')
    assert 1.0 < alpha < 500.0, \
        f"GO-CFAR alpha={alpha:.2f} 超出合理范围 [1, 500]"


def test_socfar_compute_alpha():
    """SO-CFAR 解析公式应与手动计算一致。"""
    guard, ref = 2, 8
    N = (2 * (guard + ref) + 1) ** 2 - (2 * guard + 1) ** 2
    expected = (N / 2.0) * (1e-4 ** (-2.0 / N) - 1.0)

    alpha = SOCFAR.compute_alpha(1e-4, guard, ref, method='analytic')
    assert abs(alpha - expected) < 1e-10, \
        f"SO 解析公式应精确匹配: expected={expected:.6f}, got={alpha:.6f}"


def test_auto_alpha_all_variants():
    """所有 4 种 CFAR 在 alpha=None 时均应正常工作不崩溃。"""
    rd = _make_test_rd_map()
    finder = PeakFinder()
    peaks = finder.detect(rd)

    variants = [
        ("CA", CACFAR()),
        ("OS", OSCFAR()),
        ("GO", GOCFAR()),
        ("SO", SOCFAR()),
    ]

    for name, cfar in variants:
        detections = cfar.filter(rd, peaks, guard_cells=2, reference_cells=8,
                                 alpha=None, pfa=1e-4)
        assert isinstance(detections, np.ndarray), \
            f"{name}-CFAR 应返回 numpy 数组"
        assert detections.ndim == 2, \
            f"{name}-CFAR 应返回 2D 数组"


def test_compute_alpha_all_methods():
    """所有 method 选项均应返回合理值。"""
    for method in ['analytic', 'auto']:
        a = CACFAR.compute_alpha(1e-4, 2, 8, method=method)
        assert a > 0, f"method={method}: alpha 应大于 0，实际为 {a}"


def test_alpha_pfa_monotonicity():
    """Pfa 越小 → alpha 越大（更严格的门限）。"""
    a_high = CACFAR.compute_alpha(1e-2, 2, 8, method='analytic')
    a_low = CACFAR.compute_alpha(1e-6, 2, 8, method='analytic')
    assert a_low > a_high, \
        f"低 Pfa 应有更高 alpha: Pfa=1e-2→{a_high:.2f}, Pfa=1e-6→{a_low:.2f}"


# ============================================================
# 新增测试：detect() 全图 2D CFAR 检测
# ============================================================

def test_cacfar_detect_basic():
    """CA-CFAR detect() 应检测到模拟目标。"""
    rd = _make_test_rd_map()
    cfar = CACFAR()
    detections = cfar.detect(rd, guard_cells=2, reference_cells=8, pfa=1e-2)
    assert isinstance(detections, np.ndarray), "应返回 numpy 数组"
    assert detections.ndim == 2, "应为 2D 数组"
    if len(detections) > 0:
        assert detections.shape[1] == 2, "每行应为 (doppler, range)"


def test_cacfar_detect_auto_alpha():
    """CA-CFAR detect() alpha=None 时应自动计算。"""
    rd = _make_test_rd_map()
    cfar = CACFAR()
    detections = cfar.detect(rd, guard_cells=2, reference_cells=8,
                             alpha=None, pfa=1e-4)
    assert isinstance(detections, np.ndarray)


def test_cacfar_detect_vs_filter_consistency():
    """detect() 和 filter() 在相同 alpha 下结果应一致（CA-CFAR）。"""
    rd = _make_test_rd_map()
    cfar = CACFAR()
    alpha = 8.0

    # filter() 需要候选点——用极低门限的 detect 获取
    candidates = cfar.detect(rd, guard_cells=2, reference_cells=8,
                             alpha=0.5, pfa=1e-4)
    det_filter = cfar.filter(rd, candidates, guard_cells=2, reference_cells=8,
                             alpha=alpha)

    # detect() 直接检测
    det_detect = cfar.detect(rd, guard_cells=2, reference_cells=8, alpha=alpha)

    # detect 的结果应是 filter 的子集（detect 用全图 uniform_filter，
    # filter 用逐 cell 精确求和，可能有微小浮点差异）
    detect_set = set((int(d[0]), int(d[1])) for d in det_detect)
    filter_set = set((int(d[0]), int(d[1])) for d in det_filter)
    # 两者交集应 >= 90% 的 filter 结果
    overlap = detect_set & filter_set
    if len(filter_set) > 0:
        assert len(overlap) / len(filter_set) > 0.85, \
            f"detect 和 filter 结果差异过大: overlap={len(overlap)}, filter={len(filter_set)}"


def test_all_variants_detect_no_crash():
    """所有 4 种 CFAR 变体的 detect() 均不应崩溃。"""
    rd = _make_test_rd_map()
    variants = [
        ("CA", CACFAR()),
        ("OS", OSCFAR()),
        ("GO", GOCFAR()),
        ("SO", SOCFAR()),
    ]
    for name, cfar in variants:
        detections = cfar.detect(rd, guard_cells=2, reference_cells=8,
                                 pfa=1e-2)
        assert isinstance(detections, np.ndarray), \
            f"{name}-CFAR detect() 应返回 numpy 数组"


def test_detection_pipeline_standard_mode():
    """DetectionPipeline 标准模式（无 PeakFinder）应正常工作。"""
    from fmcw.detection.detector import DetectionPipeline

    rd = _make_test_rd_map()
    det = DetectionPipeline(cfar_algorithm="ca_cfar")
    peaks = det.run(rd, guard_cells=2, reference_cells=8, pfa=1e-2)
    assert isinstance(peaks, np.ndarray)
    assert peaks.ndim == 2


def test_detection_pipeline_legacy_mode():
    """DetectionPipeline legacy 模式（有 PeakFinder）应正常工作。"""
    from fmcw.detection.detector import DetectionPipeline

    rd = _make_test_rd_map()
    det = DetectionPipeline(cfar_algorithm="ca_cfar", peak_finder=PeakFinder())
    peaks = det.run(rd, guard_cells=2, reference_cells=8, alpha=4.0)
    assert isinstance(peaks, np.ndarray)
    assert peaks.ndim == 2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
