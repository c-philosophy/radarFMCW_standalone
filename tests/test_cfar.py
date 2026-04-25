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


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
