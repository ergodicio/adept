from typing import Dict
from functools import partial
from diffrax import ODETerm
import interpax
import numpy as np
import jax.numpy as jnp
from astropy.units import Quantity as _Q

from adept.lpse2d.core.integrator import SplitStep
from adept.vlasov1d.integrator import Stepper


def get_save_quantities(cfg: Dict) -> Dict:
    """
    This function updates the config with the quantities required for the diagnostics and saving routines

    :param cfg:
    :return:
    """

    # cfg["save"]["func"] = {**cfg["save"]["func"], **{"callable": get_save_func(cfg)}}
    tmin = _Q(cfg["save"]["t"]["tmin"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    tmax = _Q(cfg["save"]["t"]["tmax"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    dt = _Q(cfg["save"]["t"]["dt"]).to("s").value / cfg["units"]["derived"]["timeScale"]
    nt = int((tmax - tmin) / dt) + 1

    cfg["save"]["t"]["dt"] = dt
    cfg["save"]["t"]["ax"] = jnp.linspace(tmin, tmax, nt)

    if "x" in cfg["save"]:
        xmin = cfg["grid"]["xmin"]
        xmax = cfg["grid"]["xmax"]
        dx = _Q(cfg["save"]["x"]["dx"]).to("m").value / cfg["units"]["derived"]["spatialScale"] * 100
        nx = int((xmax - xmin) / dx)
        cfg["save"]["x"]["dx"] = dx
        cfg["save"]["x"]["ax"] = jnp.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
        cfg["save"]["kx"] = np.fft.fftfreq(nx, d=dx / 2.0 / np.pi)

        if "y" in cfg["save"]:
            ymin = cfg["grid"]["ymin"]
            ymax = cfg["grid"]["ymax"]
            dy = _Q(cfg["save"]["y"]["dy"]).to("m").value / cfg["units"]["derived"]["spatialScale"] * 100
            ny = int((ymax - ymin) / dy)
            cfg["save"]["y"]["dy"] = dy
            cfg["save"]["y"]["ax"] = jnp.linspace(ymin + dy / 2.0, ymax - dy / 2.0, ny)
            cfg["save"]["ky"] = np.fft.fftfreq(ny, d=dy / 2.0 / np.pi)
        else:
            raise NotImplementedError("Must specify y in save")

        xq, yq = jnp.meshgrid(cfg["save"]["x"]["ax"], cfg["save"]["y"]["ax"], indexing="ij")

        interpolator = partial(
            interpax.interp2d,
            xq=jnp.reshape(xq, (nx * ny), order="F"),
            yq=jnp.reshape(yq, (nx * ny), order="F"),
            x=cfg["grid"]["x"],
            y=cfg["grid"]["y"],
            method="linear",
        )

        def save_func(t, y, args):
            save_y = {}
            for k, v in y.items():
                if k == "E0":
                    cmplx_fld = v.view(jnp.complex128)
                    save_y[k] = jnp.concatenate(
                        [
                            jnp.reshape(interpolator(f=cmplx_fld[..., ivec]), (nx, ny), order="F")[..., None]
                            for ivec in range(2)
                        ],
                        axis=-1,
                    ).view(jnp.float64)
                elif k == "epw":
                    cmplx_fld = v.view(jnp.complex128)
                    save_y[k] = jnp.reshape(interpolator(f=cmplx_fld), (nx, ny), order="F").view(jnp.float64)
                else:
                    save_y[k] = jnp.reshape(interpolator(f=v), (nx, ny), order="F")

            return save_y

    else:
        save_func = lambda t, y, args: y

    cfg["save"]["func"] = save_func

    return cfg


def get_diffeqsolve_quants(cfg):

    cfg = get_save_quantities(cfg)
    return dict(
        terms=ODETerm(SplitStep(cfg)), solver=Stepper(), saveat=dict(ts=cfg["save"]["t"]["ax"], fn=cfg["save"]["func"])
    )
