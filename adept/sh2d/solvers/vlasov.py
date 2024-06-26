from typing import Dict, Tuple
from collections import defaultdict

import equinox as eqx
import jax
from jax import numpy as jnp


def get_envelope(p_wL, p_wR, p_L, p_R, ax):
    return 0.5 * (jnp.tanh((ax - p_L) / p_wL) - jnp.tanh((ax - p_R) / p_wR))


class Driver(eqx.Module):
    xax: jax.Array
    yax: jax.Array

    def __init__(self, xax, yax):
        self.xax = xax
        self.yax = yax

    def __call__(self, this_pulse: Dict, current_time: jnp.float64):
        kk = this_pulse["k0"]
        ww = this_pulse["w0"]
        dw = this_pulse["dw0"]
        t_L = this_pulse["t_c"] - this_pulse["t_w"] * 0.5
        t_R = this_pulse["t_c"] + this_pulse["t_w"] * 0.5
        t_wL = this_pulse["t_r"]
        t_wR = this_pulse["t_r"]
        x_L = this_pulse["x_c"] - this_pulse["x_w"] * 0.5
        x_R = this_pulse["x_c"] + this_pulse["x_w"] * 0.5
        x_wL = this_pulse["x_r"]
        x_wR = this_pulse["x_r"]

        y_L = this_pulse["y_c"] - this_pulse["y_w"] * 0.5
        y_R = this_pulse["y_c"] + this_pulse["y_w"] * 0.5
        y_wL = this_pulse["y_r"]
        y_wR = this_pulse["y_r"]

        envelope_t = get_envelope(t_wL, t_wR, t_L, t_R, current_time)
        envelope_x = get_envelope(x_wL, x_wR, x_L, x_R, self.xax)
        envelope_y = get_envelope(y_wL, y_wR, y_L, y_R, self.yax)

        return (
            envelope_t
            * envelope_x[:, None]
            * envelope_y[None, :]
            * jnp.abs(kk)
            * this_pulse["a0"]
            * jnp.sin(kk * self.xax[:, None] - (ww + dw) * current_time)
        )


class Vlasov(eqx.Module):
    push_e: eqx.Module
    push_b: eqx.Module
    push_a: eqx.Module

    def __init__(self, cfg):
        self.push_e = Edfdv(cfg)
        self.push_b = Bdfdv(cfg)
        self.push_a = Vdfdx(cfg)

    def __call__(self, prev_f, e, b):
        delta_f = jax.tree_map(jnp.zeros_like, prev_f)
        delta_f = self.push_e(prev_f=prev_f, delta_f=delta_f, e=e)
        delta_f = self.push_b(prev_f=prev_f, delta_f=delta_f, b=b)
        delta_f = self.push_a(prev_f=prev_f, delta_f=delta_f)
        return delta_f


class Edfdv(eqx.Module):
    v: jax.Array
    nl: int
    ny: int
    dv: float
    a1: Dict
    c1: Dict
    c2: Dict
    b1: Dict
    a2: Dict
    c3: Dict
    c4: Dict
    b2: Dict

    def __init__(self, cfg: Dict):
        self.v = cfg["grid"]["v"]
        self.nl = cfg["grid"]["nl"]
        self.ny = cfg["grid"]["ny"]
        self.dv = cfg["grid"]["dv"]
        self.a1, self.c2, self.b1, self.a2, self.c3, self.c4, self.b2 = (
            defaultdict(dict),
            defaultdict(dict),
            defaultdict(dict),
            defaultdict(dict),
            defaultdict(dict),
            defaultdict(dict),
            defaultdict(dict),
        )
        self.c1 = {}
        for il in range(self.nl + 1):
            self.c1[il] = 1 / (2 * il + 1) * il / (2 * (il + 1))
            for im in range(il + 1):
                self.a1[il][im] = (il + 1 - im) / (2 * il + 1) * il / (2 * (il + 1))
                self.c2[il][im] = -(il - im + 2) * (il - im + 1) / (2 * il + 1) * il / (2 * (il + 1))
                self.b1[il][im] = -il / (2 * il + 1) * (il / 2)
                self.a2[il][im] = (il + im) / (2 * il + 1)
                self.c3[il][im] = 0.5 / (2 * il + 1)
                self.c4[il][im] = -0.5 * (il + im - 1) * (il + im) / (2 * il + 1)
                self.b2[il][im] = -il * (il + 1) / (2 * il + 1)

        self.a1[0][0] = 1.0
        self.c1[0] = 0.5
        self.a2[1][0] = 1.0 / 3.0
        self.b2[1][0] = 2.0 / 3.0
        # self.a3[1][0] = 0.4
        self.c3[1][1] = -0.1

    def ddv(self, f):
        return (
            jnp.gradient(jnp.concatenate([-f[..., 0:1], f, 0.0 * f[..., 0:1]], axis=-1), axis=-1)[..., 1:-1] / self.dv
        )

    def calc_gh(self, prev_f: Dict) -> Tuple[Dict, Dict]:
        g = {}
        h = {}
        for il, m_dict in prev_f.items():
            g[il] = {}
            h[il] = {}
            for im, flm in m_dict.items():
                g[il][im] = self.ddv(flm) - il / self.v[None, None, :] * flm
                h[il][im] = (il + 1) / self.v[None, None, :] * flm + self.ddv(flm)

        return g, h

    def calc_a1(self, ex, g, il, im):
        return self.a1[il][im] * ex[..., None] * g[il][0]

    def calc_c1(self, em, g, il, im):
        return self.c1[il] * em[..., None] * g[il][im]

    def calc_c2(self, ep, g, il, im):
        return self.c2[il][im] * ep[..., None] * g[il][im]

    def calc_b1(self, ep, g, il, im):
        return self.b1[il][im] * jnp.real(ep[..., None] * g[il][im])

    def calc_a2(self, ex, h, il, im):
        return self.a2[il][im] * ex[..., None] * h[il][im]

    def calc_c3(self, em, h, il, im):
        return self.c3[il][im] * em[..., None] * h[il][im]

    def calc_c4(self, ep, h, il, im):
        return self.c4[il][im] * ep[..., None] * h[il][im]

    def calc_b2(self, ep, h, il, im):
        return self.b2[il][im] * jnp.real(ep[..., None] * h[il][im])

    def __call__(self, prev_f, delta_f, e):
        g, h = self.calc_gh(prev_f)

        ex = e[:, :, 0]
        ep = e[:, :, 1] + 1j * e[:, :, 2]
        em = e[:, :, 1] - 1j * e[:, :, 2]

        for il in range(0, self.nl):
            # a1 l=0:nl-1 m=0:nm
            for im in range(0, il + 1):
                delta_f[il + 1][im] += self.calc_a1(ex=ex, g=g, il=il, im=im)

            if self.ny > 2:
                # c1 l=0:nl-1 m=0:nm-1
                for im in range(0, il):
                    delta_f[il + 1][im + 1] += self.calc_c1(em=em, g=g, il=il, im=im)

        if self.ny > 2:
            for il in range(1, self.nl):
                # c2 l=1:nl-1 m=1:nm
                for im in range(1, il + 1):
                    delta_f[il + 1][im - 1] += self.calc_c2(ep=ep, g=g, il=il, im=im)

                # b1 l=1:nl-1 m=0
                for im in range(0, 1):
                    delta_f[il + 1][im] = self.calc_b1(ep=ep, g=g, il=il, im=im + 1)

        for il in range(1, self.nl + 1):
            # a2 l=1:nl   m=0:m
            for im in range(0, il + 1):
                if il > im:
                    delta_f[il - 1][im] += self.calc_a2(ex=ex, h=h, il=il, im=im)

            if self.ny > 2:
                # c4 l=1:nl   m=1:nl
                for im in range(1, il + 1):
                    delta_f[il - 1][im - 1] += self.calc_c4(ep=ep, h=h, il=il, im=im)

        if self.ny > 2:
            # c3 l=2:nl   m=0:m-1
            for il in range(2, self.nl + 1):
                for im in range(0, il):
                    delta_f[il - 1][im + 1] += self.calc_c3(em=em, h=h, il=il, im=im)

            # b2 l=1:nl   m=0
            for il in range(1, self.nl + 1):
                for im in range(0, 1):
                    delta_f[il - 1][im] += self.calc_c4(ep=ep, h=h, il=il, im=im + 1)

        return delta_f


class Bdfdv(eqx.Module):
    v: jax.Array
    nl: int

    def __init__(self, cfg: Dict):
        self.v = cfg["grid"]["v"]
        self.nl = cfg["grid"]["nl"]

    def __call__(self, prev_f, delta_f, b):
        bx = b[..., 0]
        bm = b[..., 1] + 1j * b[..., 2]
        bp = b[..., 1] - 1j * b[..., 2]

        for il in range(1, self.nl + 1):
            for im in range(0, il):
                delta_f[il][im + 1] = 0.5 * bm[..., None] * prev_f[il][im]  # a3
            for im in range(1, il + 1):
                delta_f[il][im] += -1j * im * bx[..., None] * prev_f[il][im]  # a1
                delta_f[il][im - 1] += -(il - im + 1) * (il + im) / 2.0 * bm[..., None] * prev_f[il][im]  # a2
            delta_f[il][0] += jnp.real(-il * (il + 1) * bp[..., None] * prev_f[il][1])  # b1

        return delta_f


class Vdfdx(eqx.Module):
    v: jax.Array
    nl: int
    ny: int
    dx: float
    dy: float

    def __init__(self, cfg: Dict):
        self.v = cfg["grid"]["v"]
        self.nl = cfg["grid"]["nl"]
        self.ny = cfg["grid"]["ny"]
        self.dx = cfg["grid"]["dx"]
        self.dy = cfg["grid"]["dy"]

    def ddx(self, flm):
        return jnp.gradient(jnp.concatenate([flm[-1:], flm, flm[:1]]), axis=0)[1:-1] / self.dx

    def calc_a1(self, prev_f, il, im):
        return -((il - im + 1) / (2 * il + 1)) * self.v[None, None, :] * self.ddx(prev_f[il][im])

    def calc_a2(self, prev_f, il, im):
        return -((il + im) / (2 * il + 1)) * self.v[None, None, :] * self.ddx(prev_f[il][im])

    def calc_b1(self, prev_f, il, im):
        return (
            (il * (il + 1) / (2 * il + 1)) * self.v[None, None, :] * jnp.gradient(prev_f[il][im + 1], axis=1) / self.dy
        )

    def calc_b2(self, prev_f, il, im):
        return -self.calc_b1(-prev_f, il, im)

    def calc_c1(self, prev_f, il, im):
        return -(1.0 / (2 * il + 1)) * 0.5 * self.v[None, None, :] * jnp.gradient(prev_f[il][im], axis=1) / self.dy

    def calc_c2(self, prev_f, il, im):
        return (
            ((il - im + 2) * (il - im + 1) / (2 * il + 1))
            * 0.5
            * self.v[None, None, :]
            * jnp.gradient(prev_f[il][im], axis=1)
            / self.dy
        )

    def calc_c3(self, prev_f, il, im):
        return -self.calc_c1(prev_f, il, im)

    def calc_c4(self, prev_f, il, im):
        return (
            -((il + im - 1) * (il + im) / (2 * il + 1))
            * 0.5
            * self.v[None, None, :]
            * jnp.gradient(prev_f[il][im], axis=1)
            / self.dy
        )

    def __call__(self, prev_f, delta_f):
        for il in range(0, self.nl):
            for im in range(0, il + 1):
                delta_f[il + 1][im] += self.calc_a1(prev_f, il, im)

            if self.ny > 2:
                for im in range(0, 1):
                    delta_f[il + 1][im] += self.calc_b1(prev_f, il, im)

        for il in range(1, self.nl + 1):
            for im in range(0, il + 1):
                if il > im:
                    delta_f[il - 1][im] += self.calc_a2(prev_f, il, im)

            if self.ny > 2:
                for im in range(0, 1):
                    delta_f[il - 1][im] += self.calc_b2(prev_f, il, im)

        if self.ny > 2:
            for il in range(1, self.nl):
                for im in range(1, il):
                    delta_f[il + 1][im + 1] += self.calc_c1(prev_f, il, im)

                for im in range(1, il + 1):
                    delta_f[il + 1][im - 1] += self.calc_c2(prev_f, il, im)

            for il in range(2, self.nl + 1):
                for im in range(0, il):
                    delta_f[il - 1][im + 1] += self.calc_c3(prev_f, il, im)

                for im in range(1, il + 1):
                    delta_f[il - 1][im - 1] += self.calc_c4(prev_f, il, im)

        return delta_f
