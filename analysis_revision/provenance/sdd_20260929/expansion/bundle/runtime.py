"""Vendored Python runtime for offline bundles, with an explicit dependency closure.

The existing population handoff builder (scripts/prepare_population_gpu_bundle.py)
vendors the closure of numpy, pandas, scipy and torch only.  This module resolves
the closure of the packages the expansion code imports and reuses that builder's
tested file-copy and native-library helpers; the builder itself is not modified.
"""
from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

from ..common.io import atomic_write_json, atomic_write_text

ROOT_DISTRIBUTIONS = ("numpy", "scipy", "pandas", "torch", "packaging", "anndata", "h5py", "pot", "scikit-learn",
                      "statsmodels", "pyarrow")


def installed_closure(roots=ROOT_DISTRIBUTIONS) -> dict:
    """Resolve active Requires-Dist markers and extras against installed metadata."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pending = [(name, set()) for name in roots]
    resolved = {}
    while pending:
        name, extras = pending.pop()
        name = canonicalize_name(name)
        previous = resolved.get(name)
        if previous is not None and extras <= previous:
            continue
        extras |= previous or set()
        resolved[name] = extras
        for raw in metadata.distribution(name).requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not any(requirement.marker.evaluate({"extra": e}) for e in extras | {""}):
                continue
            version = metadata.version(requirement.name)
            if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
                raise ValueError(f"installed dependency fails its requirement: {raw}; found {version}")
            if requirement.url:
                raise ValueError(f"direct URL dependency is not supported: {raw}")
            pending.append((requirement.name, set(requirement.extras)))
    return {name: metadata.version(name) for name in sorted(resolved)}


def _population_builder(root: Path):
    spec = importlib.util.spec_from_file_location("population_bundle_builder", root / "scripts/prepare_population_gpu_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vendor_runtime(bundle, root, roots=ROOT_DISTRIBUTIONS) -> dict:
    import torch

    if platform.system() != "Linux" or platform.machine() != "x86_64" or platform.libc_ver()[0] != "glibc":
        raise RuntimeError("the offline runtime supports Linux x86_64 with glibc only")
    if torch.version.cuda is None:
        raise RuntimeError("the installed Torch must include CUDA; nothing is installed or downloaded")
    builder = _population_builder(Path(root))
    bundle = Path(bundle)
    runtime = bundle / "runtime"
    stdlib = Path(sysconfig.get_path("stdlib"))
    target_stdlib = runtime / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    builder.copy_file(Path(sys.executable), runtime / "bin/python")
    shutil.copytree(stdlib, target_stdlib, ignore=shutil.ignore_patterns("site-packages", "__pycache__", "*.pyc"))
    site = target_stdlib / "site-packages"
    site.mkdir()
    distributions = installed_closure(roots)
    copied = set()
    for name in distributions:
        distribution = metadata.distribution(name)
        if distribution.files is None:
            raise ValueError(f"installed package has no file inventory: {name}")
        for relative in distribution.files:
            if ".." in relative.parts or "__pycache__" in relative.parts or relative.suffix == ".pyc":
                continue
            target = site / relative
            if target not in copied:
                builder.copy_file(Path(distribution.locate_file(relative)), target)
                copied.add(target)
    libraries = builder.native_python_libraries(stdlib, runtime / "lib")
    lock = {"schema_version": 1, "python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "glibc": platform.libc_ver()[1], "torch_cuda": torch.version.cuda,
            "roots": list(roots), "distributions": distributions, "native_libraries": libraries,
            "excluded_host_dependencies": ["Linux kernel", "glibc and ELF loader", "NVIDIA kernel/user driver"],
            "installation_required": False, "network_required": False}
    atomic_write_json(bundle / "environment.lock.json", lock)
    atomic_write_text(bundle / "requirements.lock.txt", "# Inventory of vendored distributions; do not pip install.\n"
                      + "".join(f"{name}=={version}\n" for name, version in distributions.items()))
    return lock
