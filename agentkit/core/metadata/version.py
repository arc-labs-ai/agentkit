"""Installed distribution version metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arc-agentkit")
except PackageNotFoundError:  # pragma: no cover — running from a source tree, not an install
    __version__ = "0.0.0+unknown"
