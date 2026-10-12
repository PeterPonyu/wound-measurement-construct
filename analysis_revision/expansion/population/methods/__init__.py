"""Method registry.  Importing this package registers every baseline and flow variant."""
from . import baselines, flows  # noqa: F401  (registration side effects)
from .base import REGISTRY, create, methods_for  # noqa: F401
