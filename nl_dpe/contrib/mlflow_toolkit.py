###
# Copyright (2025) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###
from pathlib import Path

from nl_dpe.error import NlDpeError

__all__ = ["get_artifact_path"]


def get_artifact_path(run_id: str) -> Path:
    """Return the local path to the artifact directory for a given MLflow run.
    Args:
        run_id (str): The MLflow run ID for which to retrieve the artifact path.
    Returns:
        Path: The local path to the artifact directory.
    """
    # sergey: MLflow is imported here just in case I will need to make MLflow optional dependency.
    import mlflow
    import mlflow.exceptions
    from mlflow.utils.file_utils import local_file_uri_to_path

    try:
        artifact_uri: str | None = mlflow.get_run(run_id).info.artifact_uri
        if artifact_uri is None:
            raise NlDpeError(f"No artifact URI found for run ID: {run_id}.")
    except mlflow.exceptions.MlflowException as err:
        raise NlDpeError.from_exception(err)
    return Path(local_file_uri_to_path(artifact_uri))
