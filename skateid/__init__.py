"""SkateID trick recognition package."""

import sys

__version__ = "0.3.0"

# Keep in sync with [project] requires-python in pyproject.toml.
MIN_PYTHON = (3, 10)
SUPPORTED_PYTHON = ">=3.10"

if sys.version_info < MIN_PYTHON:
    raise RuntimeError(
        "SkateID requires Python {required}, but this interpreter is "
        "{major}.{minor}. Create the project environment with "
        "`py -3.14 -m venv .venv` and use `.venv\\Scripts\\python.exe`.".format(
            required=SUPPORTED_PYTHON,
            major=sys.version_info.major,
            minor=sys.version_info.minor,
        )
    )

