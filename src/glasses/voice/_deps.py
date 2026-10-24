"""Importing the optional audio stack with an error you can act on.

A bare ``ModuleNotFoundError: No module named 'sherpa_onnx'`` eight frames
deep in a pipeline tells you nothing about which command fixes it. This is
the same pattern the CLI uses for OpenCV.

It is easy to land here without having uninstalled anything: ``uv sync``
installs only the declared default dependencies and prunes everything else,
so it removes every optional extra this project has. ``uv sync
--all-extras`` is the form that keeps them.
"""

from __future__ import annotations

INSTALL_HINT = (
    "    uv pip install -e '.[face,voice]'\n"
    "\n"
    "If you just ran `uv sync`, that is the cause: it prunes optional\n"
    "extras. Use `uv sync --all-extras` instead."
)


def require_sherpa():  # type: ignore[no-untyped-def]
    """Import sherpa_onnx, or explain how to get it."""
    try:
        import sherpa_onnx
    except ImportError as error:
        raise ModuleNotFoundError(
            f"The voice pipeline needs sherpa-onnx.\n\n{INSTALL_HINT}"
        ) from error
    return sherpa_onnx
