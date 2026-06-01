"""FFT 处理链和参数估计测试。"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fmcw.config.schema import RadarParams
from fmcw.processing.range_fft import range_fft
from fmcw.processing.doppler_fft import doppler_fft
from fmcw.processing.angle_fft import angle_fft
from fmcw.processing.processor import FMCWProcessor
from fmcw.estimation.estimator import ParameterEstimator, TargetEstimate, FrameEstimates
from fmcw.estimation.doa import MUSIC, MVDR
from fmcw.processing.super_resolution import TLSESPRIT, BeamspaceMUSIC


def _make_test_signal(radar):
    """Generate a simple test signal with one target."""
    np.random.seed(42)
    signal = np.random.randn(radar.antenna_num, radar.chirp_num, radar.sample_num) * 0.1
    # Inject a target at range_bin=50, doppler_bin=20
    for a in range(radar.antenna_num):
        for d in range(radar.chirp_num):
            signal[a, d, 50] += 5.0 * np.exp(1j * 2 * np.pi * d * 20 / radar.chirp_num)
    return signal


def test_range_fft_shape():
    """Range FFT should produce correct output shape."""
    radar = RadarParams()
    signal = np.random.randn(8, 128, 1024) + 1j * np.random.randn(8, 128, 1024)

    s_r = range_fft(signal, downsample=1)
    # Half spectrum: 1024/2 = 512 bins
    assert s_r.shape == (8, 128, 512), f"Expected (8, 128, 512), got {s_r.shape}"


def test_doppler_fft_shape():
    """Doppler FFT should preserve shape."""
    radar = RadarParams()
    s_r = np.random.randn(8, 128, 512) + 1j * np.random.randn(8, 128, 512)

    s_rd = doppler_fft(s_r)
    assert s_rd.shape == (8, 128, 512)


def test_angle_fft_shape():
    """Angle FFT should produce zero-padded output."""
    radar = RadarParams()
    s_rd = np.random.randn(8, 128, 512) + 1j * np.random.randn(8, 128, 512)

    s_rda = angle_fft(s_rd, radar.angle_num)
    assert s_rda.shape == (radar.angle_num, 128, 512), \
        f"Expected ({radar.angle_num}, 128, 512), got {s_rda.shape}"


def test_processor_output():
    """FMCWProcessor should produce ProcessingResult with all fields."""
    radar = RadarParams()
    processor = FMCWProcessor(radar)
    signal = _make_test_signal(radar)

    result = processor.process(signal)
    assert result.s_r.shape == (radar.antenna_num, radar.chirp_num, radar.sample_num // 2)
    assert result.rd_map.shape == (radar.chirp_num, radar.sample_num // 2)
    assert result.ra_map.shape == (radar.angle_num, radar.sample_num // 2)


def test_parameter_estimator_basic():
    """ParameterEstimator should convert peaks to physical units."""
    radar = RadarParams()
    processor = FMCWProcessor(radar)

    # Create a simple signal with one target
    signal = np.random.randn(radar.antenna_num, radar.chirp_num, radar.sample_num) * 0.1
    signal[:, 20, 50] += 10.0  # peak at doppler=20, range=50

    result = processor.process(signal)
    estimator = ParameterEstimator(radar)

    # Test with a detection at (20, 50)
    peaks = np.array([[20, 50]], dtype=np.int64)
    estimates = estimator.estimate(result.s_rda, peaks, frame_idx=0)

    assert len(estimates.targets) == 1
    est = estimates.targets[0]
    assert est.range > 0, "Range should be positive"
    assert isinstance(est.range, float)


def test_range_bin_conversion():
    """Range bin conversion should be correct."""
    radar = RadarParams()
    estimator = ParameterEstimator(radar)

    r = estimator._range_from_bin(50)
    assert r > 0, f"Range should be positive, got {r}"

    # Bin 0 should give range 0
    r0 = estimator._range_from_bin(0)
    assert abs(r0) < 1e-10, f"Bin 0 should give range 0, got {r0}"


def test_velocity_conversion():
    """fftshift 后，bin D/2 为零速，bin D/2+1 为正速，bin D/2-1 为负速。"""
    radar = RadarParams()
    estimator = ParameterEstimator(radar)
    mid = radar.chirp_num // 2

    # fftshift 后: bin mid 对应 DC (velocity = 0)
    v0 = estimator._velocity_from_bin(mid)
    assert abs(v0) < 1e-10, f"Bin {mid} (DC after shift) should give 0, got {v0}"

    # bin mid + 1 → positive velocity
    v_pos = estimator._velocity_from_bin(mid + 1)
    assert v_pos > 0, f"Bin {mid + 1} should give positive velocity, got {v_pos}"

    # bin mid - 1 → negative velocity
    v_neg = estimator._velocity_from_bin(mid - 1)
    assert v_neg < 0, f"Bin {mid - 1} should give negative velocity, got {v_neg}"


def test_music_doa():
    """MUSIC DOA should estimate angle from array snapshots."""
    radar = RadarParams(antenna_num=8)

    # Create array snapshots with a target at 10 degrees
    np.random.seed(42)
    n_snapshots = 100
    target_angle = 10.0

    # Use the same steering vector convention as MUSIC class
    d_over_lambda = radar.antenna_spacing / radar.wl  # = 0.5
    n = np.arange(radar.antenna_num)
    steering = np.exp(-1j * 2 * np.pi * d_over_lambda * np.sin(np.deg2rad(target_angle)) * n)
    steering = steering[:, np.newaxis]  # (8, 1)

    # Generate snapshots with high SNR
    signals = np.random.randn(1, n_snapshots) + 1j * np.random.randn(1, n_snapshots)
    X = steering @ signals * 3.0
    X += 0.3 * (np.random.randn(8, n_snapshots) + 1j * np.random.randn(8, n_snapshots))

    music = MUSIC(radar, n_sources=1, angle_resolution=0.5)
    est_angles = music.estimate(X, n_peaks=1)

    assert len(est_angles) == 1, f"Expected 1 angle, got {len(est_angles)}"
    # MUSIC can estimate either +angle or -angle due to ULA ambiguity
    assert abs(abs(est_angles[0]) - 10.0) < 5.0, \
        f"Estimated angle {est_angles[0]:.1f} should be near +/-10 deg"


def test_mvdr_doa():
    """MVDR should estimate angle from array snapshots."""
    radar = RadarParams(antenna_num=8)

    np.random.seed(42)
    n_snapshots = 20
    angles_deg = np.array([-5.0])
    steering = np.exp(-1j * 2 * np.pi * 0.5 * np.sin(np.deg2rad(angles_deg)) * np.arange(8)[:, np.newaxis])

    X = steering @ np.random.randn(1, n_snapshots) * 5.0
    X += 0.1 * (np.random.randn(8, n_snapshots) + 1j * np.random.randn(8, n_snapshots))

    mvdr = MVDR(radar, angle_resolution=0.1, diagonal_loading=0.01)
    est_angles = mvdr.estimate(X, n_peaks=1)

    assert len(est_angles) == 1
    assert abs(est_angles[0] - (-5.0)) < 5.0


def test_tls_esprit():
    """TLS-ESPRIT should estimate angle from array snapshots."""
    radar = RadarParams(antenna_num=8)

    np.random.seed(42)
    n_snapshots = 50
    angles_deg = np.array([10.0])
    steering = np.exp(-1j * 2 * np.pi * 0.5 * np.sin(np.deg2rad(angles_deg)) * np.arange(8)[:, np.newaxis])

    X = steering @ np.random.randn(1, n_snapshots) * 10.0
    X += 0.2 * (np.random.randn(8, n_snapshots) + 1j * np.random.randn(8, n_snapshots))

    esprit = TLSESPRIT(radar, n_sources=1)
    est_angles = esprit.estimate(X)

    assert len(est_angles) == 1
    assert abs(est_angles[0] - 10.0) < 5.0, \
        f"ESPRIT estimated {est_angles[0]:.1f} deg, expected ~10 deg"


# ============================================================
# 新增测试：ParameterEstimator DOA 方法切换
# ============================================================

def test_estimator_doa_fft_default():
    """默认 DOA 方法应为 FFT，行为与之前一致。"""
    radar = RadarParams(antenna_num=8, chirp_num=64, sample_num=256)
    from fmcw.config.schema import EstimationConfig

    signal = np.random.randn(8, 64, 128) + 1j * np.random.randn(8, 64, 128)
    signal[:, 20, 50] += 10.0
    from fmcw.processing.processor import FMCWProcessor
    proc = FMCWProcessor(radar)
    result = proc.process(signal)

    est = ParameterEstimator(radar)
    filtered = np.array([[20, 50]], dtype=np.int64)
    frame_est = est.estimate(result.s_rda, filtered)

    assert len(frame_est.targets) == 1
    assert frame_est.targets[0].angle_bin >= 0, "FFT 方法应有有效的 angle_bin"


def test_estimator_doa_esprit():
    """ESPRIT 方法应正常返回角度估计。"""
    radar = RadarParams(antenna_num=8, chirp_num=64, sample_num=256)
    from fmcw.config.schema import EstimationConfig

    signal = np.random.randn(8, 64, 128) + 1j * np.random.randn(8, 64, 128)
    signal[:, 20, 50] += 10.0
    from fmcw.processing.processor import FMCWProcessor
    proc = FMCWProcessor(radar)
    result = proc.process(signal)

    cfg = EstimationConfig(doa_method="esprit")
    est = ParameterEstimator(radar, config=cfg)
    filtered = np.array([[20, 50]], dtype=np.int64)
    frame_est = est.estimate(result.s_rda, filtered, s_rd=result.s_rd)

    assert len(frame_est.targets) == 1
    assert -90 <= frame_est.targets[0].angle <= 90, "角度应在 ±90° 范围内"
    assert frame_est.targets[0].angle_bin == -1, "ESPRIT 方法 angle_bin 应为 -1"


def test_estimator_doa_music():
    """MUSIC 方法应正常返回角度估计。"""
    radar = RadarParams(antenna_num=8, chirp_num=64, sample_num=256)
    from fmcw.config.schema import EstimationConfig

    signal = np.random.randn(8, 64, 128) + 1j * np.random.randn(8, 64, 128)
    signal[:, 20, 50] += 10.0
    from fmcw.processing.processor import FMCWProcessor
    proc = FMCWProcessor(radar)
    result = proc.process(signal)

    cfg = EstimationConfig(doa_method="music", angle_resolution=1.0)
    est = ParameterEstimator(radar, config=cfg)
    filtered = np.array([[20, 50]], dtype=np.int64)
    frame_est = est.estimate(result.s_rda, filtered, s_rd=result.s_rd)

    assert len(frame_est.targets) == 1
    assert -90 <= frame_est.targets[0].angle <= 90
    assert frame_est.targets[0].angle_bin == -1


def test_estimator_doa_mvdr():
    """MVDR 方法应正常返回角度估计。"""
    radar = RadarParams(antenna_num=8, chirp_num=64, sample_num=256)
    from fmcw.config.schema import EstimationConfig

    signal = np.random.randn(8, 64, 128) + 1j * np.random.randn(8, 64, 128)
    signal[:, 20, 50] += 10.0
    from fmcw.processing.processor import FMCWProcessor
    proc = FMCWProcessor(radar)
    result = proc.process(signal)

    cfg = EstimationConfig(doa_method="mvdr", angle_resolution=1.0)
    est = ParameterEstimator(radar, config=cfg)
    filtered = np.array([[20, 50]], dtype=np.int64)
    frame_est = est.estimate(result.s_rda, filtered, s_rd=result.s_rd)

    assert len(frame_est.targets) == 1
    assert -90 <= frame_est.targets[0].angle <= 90
    assert frame_est.targets[0].angle_bin == -1


def test_estimator_unknown_doa_raises():
    """未知 DOA 方法应抛出 ValueError。"""
    radar = RadarParams()
    from fmcw.config.schema import EstimationConfig
    cfg = EstimationConfig(doa_method="invalid_method")
    est = ParameterEstimator(radar, config=cfg)

    import pytest
    with pytest.raises(ValueError, match="Unknown DOA method"):
        est._estimate_angle(np.zeros((8, 10, 10)), 5, 5)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
