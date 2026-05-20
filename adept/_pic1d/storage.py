"""PIC-1D save quantities.

Mirrors the structure used by :mod:`adept._vlasov1d.storage` so that the
existing input deck's ``save:`` section "just works":

- ``save.fields.t``: spacetime grids of E, dE (driver), and per-species
  moments (n, j, P) computed *via the same particle-shape deposit kernel*
  used by the field solver, so PIC and Vlasov 1:1 comparisons are honest.
- ``save.<species>.<label>.t``: snapshots of particle x, v (and weights are
  fixed-in-time so they are not re-saved).
- ``save.default``: scalar diagnostics (mean E^2, mean dE^2, per-species
  particle energy, momentum, weight).
"""

import numpy as np
from jax import numpy as jnp

from adept._pic1d.solvers.pushers.shape import deposit


def _add_dim_axes(save_config: dict) -> None:
    for dim_key, dim_config in save_config.items():
        if not isinstance(dim_config, dict) or f"n{dim_key}" not in dim_config:
            continue
        dim_min = float(dim_config[f"{dim_key}min"])
        dim_max = float(dim_config[f"{dim_key}max"])
        dim_n = int(dim_config[f"n{dim_key}"])
        if dim_key == "x":
            dx = (dim_max - dim_min) / dim_n
        else:
            dx = 0.0
        dim_config["ax"] = np.linspace(dim_min + dx / 2.0, dim_max - dx / 2.0, dim_n)


def _moments_from_particles(cfg: dict, species_name: str, y: dict, shape: str):
    nx = cfg["grid"]["nx"]
    dx = cfg["grid"]["dx"]
    xmin = cfg["grid"]["xmin"]
    x_p = y[f"x_{species_name}"]
    v_p = y[f"v_{species_name}"]
    w_p = y[f"w_{species_name}"]
    n_g = deposit(x_p, w_p, nx, dx, xmin, shape)
    j_g = deposit(x_p, w_p * v_p, nx, dx, xmin, shape)
    p_g = deposit(x_p, w_p * v_p * v_p, nx, dx, xmin, shape)
    return n_g, j_g, p_g


def get_field_save_func(cfg: dict):
    """Nested field save matching :func:`adept._vlasov1d.storage.get_field_save_func`.

    Returns a dict with one entry per species (nested ``{n, j, P}``) plus the
    shared ``e``, ``de``, ``a``, ``prev_a``, ``pond`` arrays at the top level.
    Vector potential entries are stored with their ghost cells intact so the
    post-process step can compute ``ep``/``em`` consistently with Vlasov-1D.
    """
    species_names = list(cfg["grid"]["species_params"].keys())
    shape = cfg["grid"]["particle_shape"]
    dx = cfg["grid"]["dx"]

    def save(t, y, args):
        result = {}
        for sp in species_names:
            n_g, j_g, p_g = _moments_from_particles(cfg, sp, y, shape)
            result[sp] = {"n": n_g, "j": j_g, "P": p_g}
        result["e"] = y["e"]
        result["de"] = y["de"]
        # Always include a, prev_a, pond — they're zero for pure ES runs but
        # this keeps the post-process schema identical between PIC and Vlasov.
        result["a"] = y["a"]
        result["prev_a"] = y["prev_a"]
        result["pond"] = -0.5 * jnp.gradient(y["a"] ** 2, dx)[1:-1]
        return result

    return save


def get_default_save_func(cfg: dict):
    species_names = list(cfg["grid"]["species_params"].keys())

    def save(t, y, args):
        scalars = {}
        for sp in species_names:
            v_p = y[f"v_{sp}"]
            w_p = y[f"w_{sp}"]
            mass = cfg["grid"]["species_params"][sp]["mass"]
            scalars[f"mean_KE_{sp}"] = 0.5 * mass * jnp.sum(w_p * v_p * v_p)
            scalars[f"mean_p_{sp}"] = mass * jnp.sum(w_p * v_p)
            scalars[f"sum_w_{sp}"] = jnp.sum(w_p)
        scalars["mean_e2"] = jnp.mean(y["e"] ** 2)
        scalars["mean_de2"] = jnp.mean(y["de"] ** 2)
        if "a" in y:
            scalars["mean_a2"] = jnp.mean(y["a"][1:-1] ** 2)
        return scalars

    return save


def get_dist_save_func(species_name: str):
    def save(t, y, args):
        return {"x": y[f"x_{species_name}"], "v": y[f"v_{species_name}"]}

    return save


def get_save_quantities(cfg: dict) -> dict:
    species_names = list(cfg["grid"]["species_params"].keys())
    new_save: dict = {}

    for save_type, save_config in cfg["save"].items():
        if save_type.startswith("fields"):
            _add_dim_axes(save_config)
            save_config["func"] = get_field_save_func(cfg)
            new_save[save_type] = save_config

        elif save_type in species_names:
            # Nested ``{label: {t: {...}}}`` shape, matching Vlasov-1D.
            for label, label_config in save_config.items():
                _add_dim_axes(label_config)
                label_config["func"] = get_dist_save_func(save_type)
                label_config["_species_name"] = save_type
                new_save[f"{save_type}.{label}"] = label_config

        else:
            raise NotImplementedError(f"Unknown save type for PIC-1D: {save_type}")

    cfg["save"] = new_save
    cfg["save"]["default"] = {"t": {"ax": cfg["grid"]["t"]}, "func": get_default_save_func(cfg)}
    return cfg
