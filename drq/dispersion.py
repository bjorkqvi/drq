import numpy as np

g = 9.81


def get_w(f, T, w) -> float:
    """Gets angular freqeuncy when one of f, T or w is given"""
    if sum([v is None for v in [f, T, w]]) == 0:
        raise ValueError("Give f [Hz], T [s] or w [rad/s]!")
    if sum([v is not None for v in [f, T, w]]) > 1:
        raise ValueError("Give only one of: f [Hz], T [s] or w [rad/s]!")

    if f is not None:
        w = 2 * np.pi * np.atleast_1d(f)
    if T is not None:
        w = 2 * np.pi / np.atleast_1d(T)

    return np.atleast_1d(w)


def sanitize_input(
    val1: float | np.ndarray, val2: float | np.ndarray
) -> tuple[float | np.ndarray]:
    """ "Return numpy arrays with same sizes (at least 1d)"""
    val1 = np.atleast_1d(val1)
    val2 = np.atleast_1d(val2)

    if val1.shape == val2.shape:
        return np.atleast_1d(val1), np.atleast_1d(val2)

    if (val1.shape != val2.shape) and (val1.size > 1 and val2.size > 1):
        raise Exception(
            "val1 and val2 have to be numpy arrays of SAME sizes (or scalars)!"
        )

    if val2.size == 1 and val1.size > 1:
        val2 = np.full(val1.shape, np.squeeze(val2))

    if val1.size == 1 and val2.size > 1:
        val1 = np.full(val2.shape, np.squeeze(val1))

    return val1, val2


def wavenumber(w: float | np.ndarray, depth: float | np.ndarray = 10_000) -> float:
    """Calculates the wavenumber from angulat wave freqeuncy and depth using linear dispersion"""
    w, depth = sanitize_input(w, depth)
    kd = w**2 / g
    k = kd
    for __ in range(200):
        k = w**2 / g / np.tanh(k * depth)
    return k


def wavelength(T: float, depth: float = 10_000) -> float:
    """Calculates the wavelength from wave period and depth using linear dispersion"""
    T, depth = sanitize_input(T, depth)
    w = 2 * np.pi / T
    k = wavenumber(w=w, depth=depth)
    return 2 * np.pi / k


def inverse_phase_speed(
    f: float = None,
    T: float = None,
    w: float = None,
    k: float = None,
    depth: float = 10_000,
):
    """Calculates the inverse wave phase speed using linear theory"""
    if k is not None:
        kh = k * depth
        kh = np.minimum(kh, 2 * np.pi)
        tanhkh = np.tanh(kh)
        tanhkh[np.abs(tanhkh) < 0.0001] = np.nan
        v = np.sqrt(k / g / tanhkh)
        v = v * np.sign(k)
        v[np.isnan(v)] = 0
        return v
    w = get_w(f, T, w)
    w, depth = sanitize_input(w, depth)
    k = wavenumber(w, depth=depth)

    return k / w


def phase_speed(
    f: float = None,
    T: float = None,
    w: float = None,
    k: float = None,
    depth: float = 10_000,
):
    """Calculates the wave phase speed using linear theory"""
    if k is not None:
        kh = k * depth
        kh = np.minimum(kh, 2 * np.pi)
        return np.sqrt(g / k * np.tanh(kh))

    w = get_w(f, T, w)
    w, depth = sanitize_input(w, depth)
    k = wavenumber(w, depth=depth)

    return w / k


def group_speed(
    f: float = None,
    T: float = None,
    w: float = None,
    depth: float = 10_000,
):
    """Calculates the wave group speed using linear theory"""
    w = get_w(f, T, w)
    w, depth = sanitize_input(w, depth)
    c = phase_speed(w=w, depth=depth)
    k = c / w

    kh = k * depth

    kh = np.minimum(kh, 2 * np.pi)
    n = 0.5 * (1 + 2 * kh / np.sinh(2 * kh))

    return c * n
