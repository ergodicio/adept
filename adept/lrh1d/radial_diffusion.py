from jax import numpy as jnp


def create_abc(kappa, r, dt):
    r_T = 0.5 * (r[1:] + r[:-1])
    dr_T = r[1:] - r[:-1]

    diag0 = (1.0 + 2.0 * dt * kappa / dr_T**2.0 - 2.0 * dt * (r_T - dr_T) * kappa / (2.0 * r_T * dr_T**2.0))[0]
    diag = 1.0 + 2.0 * dt * kappa / dr_T**2.0
    sub_diag = -2.0 * (r_T - dr_T) * kappa  # + r_T * dr_T * jnp.gradient(kappa, dr_T)
    super_diag = -2.0 * (r_T - dr_T) * kappa  # - r_T * dr_T * jnp.gradient(kappa, dr_T)

    sub_diag /= 2.0 * r_T * dr_T**2.0
    super_diag /= 2.0 * r_T * dr_T**2.0

    return dt * sub_diag[1:], diag, dt * super_diag[:-1]


if __name__ == "__main__":
    from lineax import linear_solve, TridiagonalLinearOperator, Tridiagonal

    kappa = 1e-8
    dt = 0.1

    dr = jnp.concatenate([jnp.array([0.0]), jnp.ones(32) * 0.1, jnp.ones(8) * 0.4])
    r = jnp.cumsum(dr)

    T = jnp.exp(-((r - 2.5) ** 2.0) / 0.8**2.0)

    a, b, c = create_abc(kappa, T, dt=0.1)
    sol = linear_solve(
        TridiagonalLinearOperator(diagonal=b, lower_diagonal=a, upper_diagonal=c), T[1:], solver=Tridiagonal()
    ).value
    print(sol)
