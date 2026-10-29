"""Check rendered cell geometry without launching a simulation."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from adept import patched_mlflow
from adept._vlasov1d.helpers import post_process
from adept._vlasov1d.iaw import IAWTurbulence1D


@pytest.mark.parametrize("save_kind", ["cell_average", "uniform", "interpolated", "fourier"])
def test_phase_space_mesh_geometry(tmp_path, monkeypatch, save_kind):
    plt.switch_backend("Agg")
    # Deliberate jumps in width: inferred midpoints would displace several faces.
    velocity_edges = np.array([-4, -2, -1.5, -1, -0.5, 0, 2, 4], dtype=float)
    velocity_centers = (velocity_edges[:-1] + velocity_edges[1:]) / 2
    x = np.array([-2, 0, 2, 4], dtype=float)
    key = "ion.main"
    save = {"_species_name": "ion", "t": {}}
    species_grid = {"v": velocity_centers, "v_edges": velocity_edges}
    plot_v = velocity_centers
    expected_x_edges = np.array([-3, -1, 1, 3, 5], dtype=float)
    expected_v_edges = velocity_edges
    if save_kind != "cell_average":
        plot_v = np.arange(-3, 4, dtype=float)
        expected_v_edges = np.arange(-3.5, 4, dtype=float)
        if save_kind == "uniform":
            species_grid = {"v": plot_v}
        else:
            spatial_dim = "kx" if save_kind == "fourier" else "x"
            if spatial_dim == "kx":
                x = np.arange(4, dtype=float)
                expected_x_edges = np.arange(-0.5, 4, dtype=float)
            save.update({spatial_dim: {"ax": x}, "v": {"ax": plot_v}})
    cfg = {"grid": {"x": x, "dx": 2.0, "species_grids": {"ion": species_grid}}, "save": {key: save}}
    times = np.arange(3, dtype=float)
    values = (1 + 0.1 * times[:, None, None]) * (1 + 0.1 * x[None, :, None]) * (1 + 0.1 * plot_v[None, None, :])
    result = SimpleNamespace(
        ts={key: times, "default": times},
        ys={key: values, "default": {"mean_n_ion": np.ones(3)}},
    )
    meshes = []
    original = Axes.pcolormesh

    def record_mesh(self, *args, **kwargs):
        mesh = original(self, *args, **kwargs)
        # Exclude colorbar meshes; inspect the actual rendered QuadMesh vertices.
        if len(args) >= 3 and np.shape(args[2]) == (7, 4):
            meshes.append(mesh.get_coordinates().copy())
        return mesh

    monkeypatch.setattr(Axes, "pcolormesh", record_mesh)
    monkeypatch.setattr(patched_mlflow, "log_metrics", lambda *args, **kwargs: None)
    post_process(result, cfg, str(tmp_path), {})
    assert len(meshes) == 2  # Initial distribution and its time difference.
    assert (tmp_path / "plots/dists/ion.main/phase_space.png").is_file()

    if save_kind != "fourier":
        # Exercise the IAW panels with bounds loaded from the saved NetCDF file.
        with xr.open_dataset(tmp_path / "binary/dist-ion.main.nc") as dataset:
            saved = dataset.load()
        IAWTurbulence1D._plot_delta_f_panels(SimpleNamespace(cfg=cfg), {key: saved}, str(tmp_path))
        assert len(meshes) == 4
        assert (tmp_path / "plots/dists/ion.main/phase_space_dfx.png").is_file()

    for coordinates in meshes:
        np.testing.assert_allclose(coordinates[:, :, 0], np.broadcast_to(expected_x_edges, (8, 5)), atol=1e-14)
        np.testing.assert_allclose(coordinates[:, :, 1], np.broadcast_to(expected_v_edges[:, None], (8, 5)), atol=1e-14)
