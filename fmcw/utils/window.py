"""雷达信号处理 FFT 窗函数。"""

import numpy as np


_WINDOW_REGISTRY = {}


def register_window(name: str):
    """装饰器工厂：返回注册窗函数的装饰器。

    Args:
        name: 窗函数名称。

    Returns:
        装饰器函数。
    """
    def decorator(fn):
        _WINDOW_REGISTRY[name] = fn
        return fn
    return decorator


@register_window("hanning")
def _hanning(n: int) -> np.ndarray:
    """生成汉宁窗（Hann window）。

    Args:
        n: 窗口长度（采样点数）。

    Returns:
        值在 [0, 1] 范围内的 1D 汉宁窗数组。

    Note:
        包装 numpy.hanning 以兼容注册表机制。
    """
    return np.hanning(n)


@register_window("hamming")
def _hamming(n: int) -> np.ndarray:
    """生成汉明窗（Hamming window）。

    Args:
        n: 窗口长度（采样点数）。

    Returns:
        值在 [0, 1] 范围内的 1D 汉明窗数组。

    Note:
        包装 numpy.hamming 以兼容注册表机制。
    """
    return np.hamming(n)


@register_window("blackman")
def _blackman(n: int) -> np.ndarray:
    """生成布莱克曼窗（Blackman window）。

    Args:
        n: 窗口长度（采样点数）。

    Returns:
        值在 [0, 1] 范围内的 1D 布莱克曼窗数组。

    Note:
        包装 numpy.blackman 以兼容注册表机制。
    """
    return np.blackman(n)


@register_window("bartlett")
def _bartlett(n: int) -> np.ndarray:
    """生成巴特利特窗（Bartlett window，三角窗）。

    Args:
        n: 窗口长度（采样点数）。

    Returns:
        值在 [0, 1] 范围内的 1D 巴特利特窗数组。

    Note:
        包装 numpy.bartlett 以兼容注册表机制。
    """
    return np.bartlett(n)


@register_window("blackmanharris")
def _blackmanharris(n: int) -> np.ndarray:
    """4项 Blackman-Harris 窗，旁瓣抑制约 92 dB。

    Args:
        n: 窗口长度（采样点数）。

    Returns:
        值在 [0, 1] 范围内的 1D Blackman-Harris 窗数组。
    """
    # 4项 Blackman-Harris 系数
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168

    k = np.arange(n)
    w = a0 - a1 * np.cos(2.0 * np.pi * k / (n - 1)) \
        + a2 * np.cos(4.0 * np.pi * k / (n - 1)) \
        - a3 * np.cos(6.0 * np.pi * k / (n - 1))
    return w


@register_window("chebyshev")
def _chebyshev(n: int, attenuation_db: float = 60.0) -> np.ndarray:
    """Dolph-Chebyshev window with configurable sidelobe attenuation."""
    # Convert dB attenuation to Chebyshev parameter
    atten = 10.0 ** (attenuation_db / 20.0)
    beta = np.cosh(np.arccosh(atten) / (n - 1)) if n > 1 else 1.0
    k = np.arange(n)
    w = np.cos(
        n * np.arccos(beta * np.cos(np.pi * k / n))
    ) / np.cosh(n * np.arccosh(beta))
    w = np.fft.fft(w, n=2 * n)
    w = np.abs(w[:n])
    w = w / w.max()
    return w


@register_window("taylor")
def _taylor(n: int, nbar: int = 5, sll: float = -35.0) -> np.ndarray:
    """Taylor window with nearly constant-level sidelobes."""
    # Simplified Taylor implementation
    a = np.arccosh(10.0 ** (-sll / 20.0)) / np.pi
    sigma_sq = nbar**2 / (a**2 + (nbar - 0.5) ** 2)

    w = np.ones(n)
    for m in range(1, nbar):
        fm = 1.0
        for p in range(1, nbar):
            fm *= 1.0 - m**2 / (sigma_sq * (a**2 + (p - 0.5) ** 2))
        fm = fm * (-1) ** (m + 1) * 0.5
        w += 2.0 * fm * np.cos(2.0 * np.pi * m * np.arange(n) / n)
    return w / w.max()


def create_window(n: int, window_type: str = "hanning", **kwargs) -> np.ndarray:
    """Create a window of length n.

    Args:
        n: Window length.
        window_type: Name of the window (hanning, hamming, blackman,
                     chebyshev, taylor).
        **kwargs: Extra arguments passed to the window function.

    Returns:
        1D window array of length n.
    """
    if window_type is None or window_type.lower() == "none":
        return np.ones(n)

    fn = _WINDOW_REGISTRY.get(window_type.lower())
    if fn is None:
        raise ValueError(
            f"Unknown window '{window_type}'. "
            f"Available: {list(_WINDOW_REGISTRY.keys())}"
        )
    return fn(n, **kwargs)


def available_windows() -> list:
    return list(_WINDOW_REGISTRY.keys())
