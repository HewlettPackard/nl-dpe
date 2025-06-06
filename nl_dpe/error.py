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

__all__ = ["NlDpeError"]


class NlDpeError(Exception):
    """Base class for all exceptions raised by the nl-dpe package."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    @classmethod
    def from_exception(cls, exception: Exception) -> "NlDpeError":
        return cls(str(exception))
