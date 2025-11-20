"""ReClaim core package."""

from importlib import metadata


try:
    __version__ = metadata.version("reclaim")
except metadata.PackageNotFoundError:  # pragma: no cover - resolved at runtime
    __version__ = "0.0.0"

__all__ = ["__version__", "main"]


def main() -> None:
    """Default console entry point for basic smoke checks."""
    print("ReClaim is ready to reclaim your time.")
