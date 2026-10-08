"""``adept.utils.log_params`` must log every flattened entry, 100 per call."""

from __future__ import annotations

import pytest

from adept import utils


@pytest.mark.parametrize("n", [1, 99, 100, 101, 199, 200, 201, 202, 250, 300, 301, 303, 450])
def test_log_params_logs_every_entry(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(utils.mlflow, "log_params", lambda d: calls.append(dict(d)))
    cfg = {"group": {f"k{i:04d}": i for i in range(n)}}

    utils.log_params(cfg)

    logged = {k for c in calls for k in c}
    assert logged == {f"group.k{i:04d}" for i in range(n)}
    assert all(0 < len(c) <= 100 for c in calls)
    assert len(calls) == max(1, -(-n // 100))
