"""Entry point: python -m expansion <command>."""
import sys

from .runner.cli import main

if __name__ == "__main__":
    sys.exit(main())
