"""Entry point: python -m expansion <command>."""
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'remote/autodl/handoff.json').is_file():
    from remote.autodl.resources import apply_cpu_limit
    apply_cpu_limit(ROOT)

from .runner.cli import main

if __name__ == "__main__":
    sys.exit(main())
