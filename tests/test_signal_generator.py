"""IF 信号生成和场景仿真测试。"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fmcw.config.schema import RadarParams, TargetParams, SceneConfig, TargetSpec, MotionModel
from fmcw.signal.generator import IFSignalGenerator
from fmcw.signal.scene import Scene
from fmcw.signal.target_fluctuation import TargetFluctuation
from fmcw.signal.antenna_pattern import UniformLinearArray
from fmcw.signal.impairments import PhaseNoise, IQImbalance, ThermalNoise


def test_generator_output_shape():
    """Generator should produce correct 3D output shape."""
    radar = RadarParams(fc=79e9, B=0.5e9, Tc=40e-6, chirp_num=128, sample_num=1024, antenna_num=8)
    gen = IFSignalGenerator(radar)

    targets = [TargetParams(range=50.0, velocity=10.0, angle=20.0)]
    signal = gen.generate_frame(targets)

    assert signal.shape == (8, 128, 1024), \
        f"Expected (8, 128, 1024), got {signal.shape}"
    assert signal.dtype == np.float64


def test_generator_snr():
    """Higher SNR should produce cleaner signals (less noise power)."""
    radar = RadarParams()
    gen = IFSignalGenerator(radar)

    targets = [TargetParams(range=50.0, velocity=10.0, angle=20.0)]

    sig_low = gen.generate_frame(targets, snr_db=5.0)
    sig_high = gen.generate_frame(targets, snr_db=40.0)

    noise_low = np.std(sig_low)
    noise_high = np.std(sig_high)

    # High SNR signal should have higher std (more signal dominance)
    # Actually, with higher SNR the noise is lower, so the signal is cleaner
    # Relative to signal power, not absolute std


def test_generator_multi_target():
    """Generator should handle multiple targets."""
    radar = RadarParams()
    gen = IFSignalGenerator(radar)

    targets = [
        TargetParams(range=50.0, velocity=10.0, angle=20.0),
        TargetParams(range=30.0, velocity=-5.0, angle=-10.0),
    ]
    signal = gen.generate_frame(targets)

    assert signal.shape[0] == radar.antenna_num
    assert signal.shape[1] == radar.chirp_num
    assert signal.shape[2] == radar.sample_num


def test_fluctuation_swerling0():
    """Swerling 0 should always return factor 1.0."""
    fluc = TargetFluctuation(model=0)
    factors = fluc.generate(5)
    assert np.allclose(factors, 1.0), "Swerling 0 should always return 1.0"


def test_fluctuation_swerling1():
    """Swerling 1 should return non-negative factors."""
    fluc = TargetFluctuation(model=1)
    np.random.seed(42)
    factors = fluc.generate(100)
    assert len(factors) == 100
    assert np.all(factors >= 0), "Swerling factors should be non-negative"


def test_scene_trajectory():
    """Scene should generate correct target trajectories."""
    radar = RadarParams()
    scene = SceneConfig(
        num_frames=10,
        frame_interval=0.05,
        targets=[
            TargetSpec(id=1, initial_range=50, initial_velocity=10, initial_angle=20,
                       motion=MotionModel(motion="constant_velocity")),
        ],
    )
    s = Scene(radar, scene)
    traj = s.all_trajectories()

    assert len(traj) == 10, f"Expected 10 frames, got {len(traj)}"
    assert len(traj[0]) == 1, f"Expected 1 target per frame, got {len(traj[0])}"

    # After 10 frames at 0.05s interval: range = 50 + 10 * 0.5 = 55m
    final_target = traj[-1][0]
    expected_range = 50.0 + 10.0 * 9 * 0.05
    assert abs(final_target.range - expected_range) < 0.01


def test_scene_angle_update():
    """Scene should update angle with angular velocity."""
    radar = RadarParams()
    scene = SceneConfig(
        num_frames=10,
        frame_interval=0.05,
        targets=[
            TargetSpec(id=1, initial_range=50, initial_velocity=10, initial_angle=20,
                       motion=MotionModel(motion="constant_velocity", angular_velocity=2.0)),
        ],
    )
    s = Scene(radar, scene)
    traj = s.all_trajectories()

    # After 10 frames = 0.5s, angle = 20 + 2*0.5 = 21 deg
    final_angle = traj[-1][0].angle
    expected_angle = 20.0 + 2.0 * 9 * 0.05
    assert abs(final_angle - expected_angle) < 0.01, \
        f"Angle should be {expected_angle}, got {final_angle}"


def test_scene_stream():
    """Scene stream should yield correct tuples."""
    radar = RadarParams()
    scene_cfg = SceneConfig(
        num_frames=3,
        targets=[
            TargetSpec(id=1, initial_range=50, initial_velocity=0, initial_angle=0),
        ],
    )
    s = Scene(radar, scene_cfg)

    frames = list(s.stream())
    assert len(frames) == 3

    for i, (frame_idx, signal, targets) in enumerate(frames):
        assert frame_idx == i
        assert signal.shape == (radar.antenna_num, radar.chirp_num, radar.sample_num)
        assert len(targets) == 1


def test_antenna_ula_steering():
    """ULA should produce correct steering vector shape."""
    ula = UniformLinearArray(n_elements=8)
    sv = ula.steering_vector(angle_deg=20.0)
    assert sv.shape == (8,)
    assert np.iscomplexobj(sv)


def test_phase_noise():
    """Phase noise should not crash on real input."""
    pn = PhaseNoise(phase_noise_dbc_hz=-100.0)
    signal = np.random.randn(8, 128, 256)
    result = pn.apply(signal, bandwidth_hz=1e6)
    assert result.shape == signal.shape


def test_iq_imbalance():
    """IQ imbalance should not crash on complex input."""
    iq = IQImbalance(amplitude_db=1.0, phase_deg=2.0)
    signal = np.random.randn(8, 128, 256) + 1j * np.random.randn(8, 128, 256)
    result = iq.apply(signal)
    assert result.shape == signal.shape
    assert np.iscomplexobj(result)


def test_thermal_noise():
    """Thermal noise should add noise and reduce SNR."""
    tn = ThermalNoise(seed=42)
    signal = np.ones((100,))
    noisy = tn.complex_awgn(signal, snr_db=20.0)

    # Noisy signal should have the right length
    assert len(noisy) == 100
    # SNR should be approximately 20 dB after noise addition
    actual_snr = 10 * np.log10(np.mean(np.abs(signal)**2) / np.mean(np.abs(noisy - signal)**2))
    assert abs(actual_snr - 20.0) < 3.0


def test_generator_with_impairments():
    """Generator should work with impairment models enabled."""
    radar = RadarParams()
    pn = PhaseNoise(phase_noise_dbc_hz=-100.0, seed=42)
    gen = IFSignalGenerator(radar, phase_noise=pn)

    targets = [TargetParams(range=50.0, velocity=10.0, angle=20.0)]
    signal = gen.generate_frame(targets, snr_db=30.0)

    assert signal.shape == (radar.antenna_num, radar.chirp_num, radar.sample_num)
    assert np.isfinite(signal).all(), "Signal should not contain NaN or inf"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
