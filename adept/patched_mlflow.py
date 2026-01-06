"""
Patched MLflow module that changes the REST API path prefix from /api/2.0 to /ajax-api/2.0.

Usage:
    Instead of:
        import mlflow

    Use:
        import patched_mlflow as mlflow

    Or:
        from patched_mlflow import *
"""

import mlflow.utils.rest_utils

# Apply the monkey patch: change REST API path prefix
mlflow.utils.rest_utils._REST_API_PATH_PREFIX = "/ajax-api/2.0"
mlflow.utils.rest_utils._TRACE_REST_API_PATH_PREFIX = "/ajax-api/2.0/mlflow/traces"

# Import necessary modules and regenerate the endpoint mappings
from mlflow.protos.service_pb2 import MlflowService
from mlflow.store.tracking.rest_store import RestStore
from mlflow.utils.rest_utils import extract_api_info_for_service

# Patch the RestStore class to use the new API endpoints
RestStore._METHOD_TO_INFO = extract_api_info_for_service(MlflowService, "/ajax-api/2.0")

# Import the patched mlflow module
import mlflow

# Re-export all mlflow attributes so this module can be used as a drop-in replacement
__all__ = [name for name in dir(mlflow) if not name.startswith("_")]

# Make all mlflow attributes available in this module
for name in __all__:
    globals()[name] = getattr(mlflow, name)

# Also make the mlflow module itself available for cases like mlflow.tracking.MlflowClient()
__version__ = mlflow.__version__
__doc__ = mlflow.__doc__
