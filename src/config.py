"""Compatibility import surface for project-wide configuration constants.

The canonical configuration module is `simulation.config`.
This module exists only for backward-compatible `import config` usage.
"""

from simulation.config import *  # noqa: F403,F401
from simulation.config import __all__ as __all__  # noqa: F401
