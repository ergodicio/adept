"""Central finite-difference stencils of the FD light solver (LPSE ``evolution.solverOrder``).

LPSE's ``SchrodingerSolver3::step_2d`` (``SchrodingerSolver3.cpp:1342-1624``) offers the
2nd-, 4th- and 6th-order central stencils for the vector Laplacian and the grad-div term;
the cross derivative is the product of the matching first-derivative stencils (its
``/144`` and ``/3600`` denominators are ``12^2`` and ``60^2``). This module holds the
coefficients, the stencil symbol (for the stability bound and the grid dispersion) and
the plane-wave injector rows that follow from any of them.
"""

import numpy as np

# second derivative, f''(x) dx^2 = sum_j c_j f(x + j dx), j = -m .. m
_SECOND = {
    2: np.array([1.0, -2.0, 1.0]),
    4: np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0,
    6: np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]) / 180.0,
}
# first derivative, f'(x) dx = sum_j d_j f(x + j dx)
_FIRST = {
    2: np.array([-1.0, 0.0, 1.0]) / 2.0,
    4: np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0,
    6: np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0,
}
ORDERS = tuple(_SECOND)


def check_order(order) -> int:
    order = int(order)
    if order not in _SECOND:
        raise ValueError(f"terms.light.fd_order must be one of {ORDERS}, got {order!r}")
    return order


def second_derivative(order: int) -> np.ndarray:
    """Coefficients ``c_j`` of ``f'' dx^2``, ``j = -order/2 .. order/2``."""
    return _SECOND[check_order(order)]


def first_derivative(order: int) -> np.ndarray:
    """Coefficients ``d_j`` of ``f' dx``, ``j = -order/2 .. order/2``."""
    return _FIRST[check_order(order)]


def symbol(order: int, theta) -> np.ndarray:
    """``sigma(theta) = sum_j c_j exp(i j theta)`` (real): the second-derivative stencil applied
    to ``exp(i k x)`` gives ``sigma(k dx) / dx^2``; ``sigma -> -theta^2`` as ``theta -> 0``."""
    c = second_derivative(order)
    m = len(c) // 2
    theta = np.asarray(theta, dtype=np.float64)
    return sum(c[j + m] * np.cos(j * theta) for j in range(-m, m + 1))


def max_eigenvalue_factor(order: int) -> float:
    """``-sigma(pi) / 4``: the stencil's largest eigenvalue per dimension relative to the
    2nd-order ``4 / dx^2`` (1, 4/3, 68/45), which scales the explicit scheme's dt limit."""
    return float(-symbol(order, np.pi) / 4.0)


def curl_curl_max_eigenvalue(order: int, dx: float, dy: float | None, out_of_plane: bool) -> float:
    """Largest eigenvalue of the discrete ``curl curl`` of ``RamanLight.curl_curl`` (1 / length^2).

    In-plane (E_x, E_y) the symbol is ``[[b, -m], [-m, a]]`` with ``a, b = -sigma(k h) / h^2`` and
    ``m`` the cross-derivative symbol; E_z sees the plain Laplacian ``a + b``. ``dy = None`` is the
    1-D grid (no y derivatives). With E_z never excited (``out_of_plane = False``, LPSE's
    ``is_pPolarizedIn2D``) the in-plane maximum is 4 / h^2 times 1, 1.4047, 1.6860 at orders 2, 4, 6;
    LPSE's empirical ``dimensionFactor * solverOrderFactor * extraFactorThatIDoNotUnderstand`` is
    1, 4/3 / 0.94 = 1.4184, 68/45 / 0.89 = 1.6979 (LightSolver.cpp:2663-2724), about 1 % above it.
    """
    c, d = second_derivative(order), first_derivative(order)
    m = len(c) // 2
    theta = np.linspace(0.0, np.pi, 1025)  # the symbols are even in theta; theta = pi is on the grid
    tx = theta[:, None]
    ty = theta[None, :] if dy is not None else np.zeros((1, 1))
    a = -sum(c[j + m] * np.cos(j * tx) for j in range(-m, m + 1)) / dx**2
    b = -sum(c[j + m] * np.cos(j * ty) for j in range(-m, m + 1)) / (dy**2 if dy is not None else 1.0)
    fx = sum(d[j + m] * np.sin(j * tx) for j in range(-m, m + 1)) / dx
    fy = sum(d[j + m] * np.sin(j * ty) for j in range(-m, m + 1)) / (dy if dy is not None else 1.0)
    in_plane = (a + b) / 2.0 + np.sqrt(((a - b) / 2.0) ** 2 + (fx * fy) ** 2)
    return float(max(in_plane.max(), (a + b).max() if out_of_plane else 0.0))


def grid_wavenumber(k_dx: float, order: int) -> float:
    """The grid wavenumber ``k_g dx`` of the stencil's propagating mode at physical ``k dx``:
    the root of ``sigma(theta) = -(k dx)^2`` in ``(0, pi]``, or ``pi`` when the mode is outside
    the stencil's band (``(k dx)^2 > -sigma(pi)``). Second order: ``cos(k_g dx) = 1 - (k dx)^2/2``."""
    order = check_order(order)
    target = -(float(k_dx) ** 2)
    if target <= float(symbol(order, np.pi)):
        return float(np.pi)
    if order == 2:
        return float(np.arccos(1.0 + target / 2.0))
    lo, hi = 0.0, float(np.pi)
    # sigma is monotone decreasing on (0, pi] for these stencils: bisection to round-off
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(symbol(order, mid)) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def injector_offsets(order: int, direction: int) -> list[int]:
    """Row offsets ``r - i_plane`` that the plane-wave injector of ``order`` writes: with
    ``direction = +1`` the wave fills ``r >= i_plane + 1`` (LPSE's x-min injector), with
    ``-1`` it fills ``r <= i_plane`` (x-max); the offsets are ``-m + 1 .. m`` for
    ``m = order / 2``, i.e. the two rows ``0, 1`` at second order."""
    m = check_order(order) // 2
    return list(range(-m + 1, m + 1))


def injector_weights(order: int, direction: int) -> dict[int, list[tuple[int, float]]]:
    """The total-field / scattered-field injector of the second-derivative stencil: for the
    analytic plane wave ``V`` and the mask ``H`` of the region the wave is to fill, the source
    ``S = D[H V] - H D[V]`` (``D`` the stencil) launches exactly ``V`` into the masked region
    and nothing outside it, for any stencil order. Row by row,

        S_r = sum_j c_j [H(r + j) - H(r)] V(r + j),

    which is non-zero only within ``m`` cells of the plane. Returns, per row offset, the
    ``(j, weight)`` pairs with ``weight = c_j [H(r+j) - H(r)]``, so that
    ``S_r = sum weight * V(r + j)`` and the second-order source is LPSE's / MATLAB's
    two-point injector: ``S_{i0} = +V(i0 + 1)``, ``S_{i0+1} = -V(i0)``."""
    order = check_order(order)
    c = second_derivative(order)
    m = order // 2
    if direction not in (1, -1):
        raise ValueError("direction must be +1 (fills rows above the plane) or -1 (below)")

    def mask(r):
        return 1.0 if (r >= 1 if direction == 1 else r <= 0) else 0.0

    weights: dict[int, list[tuple[int, float]]] = {}
    for r in injector_offsets(order, direction):
        rows = []
        for j in range(-m, m + 1):
            w = c[j + m] * (mask(r + j) - mask(r))
            if w != 0.0:
                rows.append((j, float(w)))
        if rows:
            weights[r] = rows
    return weights


def launched_amplitude_ratio(k_dx: float, order: int) -> float:
    """Amplitude of the plane wave the injector launches relative to the analytic amplitude
    of ``V``: the source's projection on the stencil's propagating mode ``k_g`` over the
    discrete group velocity ``-sigma'(k_g dx)``, both from the same stencil,

        |sum_r S_r e^{-i k_g r dx}| / |sigma'(k_g dx)|  with  V(r) = e^{i k r dx}.

    Second order at small ``k dx``: ``sin(k dx) / sin(k_g dx)`` (the MATLAB calibration);
    the deficit falls with the order as the stencil's dispersion error does."""
    order = check_order(order)
    theta_g = grid_wavenumber(k_dx, order)
    c = second_derivative(order)
    m = order // 2
    projection = 0.0j
    for r, rows in injector_weights(order, +1).items():
        for j, w in rows:
            projection += w * np.exp(1j * float(k_dx) * (r + j)) * np.exp(-1j * theta_g * r)
    d_sigma = sum(-j * c[j + m] * np.sin(j * theta_g) for j in range(-m, m + 1))
    return float(abs(projection) / abs(d_sigma))
