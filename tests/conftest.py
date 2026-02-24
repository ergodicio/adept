import os

import git
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
np.seterr(divide="ignore")


@pytest.fixture
def tags():
    repo = git.Repo()
    try:
        branch_name = os.environ["BRANCH_NAME"]
    except KeyError:
        branch_name = repo.active_branch.name
    return dict(
        branch=branch_name,
        commit_id=repo.head.object.hexsha,
        commit_author=repo.head.object.author.name,
        commit_author_email=repo.head.object.author.email,
    )
