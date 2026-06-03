from __future__ import annotations

from pathlib import Path


def require_existing_file(value: str | Path, flag: str) -> str:
    """Validate a required file path and return its absolute string path."""
    if value is None:
        raise FileNotFoundError(f"{flag} is missing.")

    path_str = str(value).strip()
    if not path_str:
        raise FileNotFoundError(
            f"{flag} is empty. If you launched this from a new terminal, source scripts/experiment_env.sh first."
        )

    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{flag} does not exist: {path.resolve()}")

    return str(path.resolve())
