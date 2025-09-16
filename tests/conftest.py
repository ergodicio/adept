import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
np.seterr(divide="ignore")
