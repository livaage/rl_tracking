from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple


def _ensure_paths_exist(paths: Iterable[Path]) -> List[Path]:
    resolved = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {path}")
        resolved.append(path.resolve())
    if not resolved:
        raise ValueError("No data directories were resolved from configuration")
    return resolved


def resolve_data_directories(
    data_config: dict,
    default_dir: Path,
) -> Tuple[Path, List[Path]]:
    """Resolve dataset base directory and list of event directories to use.

    The function supports several configuration styles:

    - ``base_dir`` + ``include_all_parts`` + ``parts_glob`` to load every part.*
    - ``base_dir`` + explicit ``directories`` listing subdirectories.
    - Single ``data_dir`` pointing to one part.
    - Falling back to ``default_dir`` when nothing is specified.

    Parameters
    ----------
    data_config:
        Configuration dictionary from the training YAML/JSON.
    default_dir:
        Fallback directory that contains TrackML event folders.

    Returns
    -------
    base_dir, directories:
        The base directory (used for the ``EventProcessor``) and a list of
        sub-directories (each containing TrackML events) to iterate over.
    """

    base_dir_cfg = data_config.get("base_dir")
    data_dir_cfg = data_config.get("data_dir")

    if base_dir_cfg is not None:
        base_dir = Path(base_dir_cfg).expanduser().resolve()
    elif data_dir_cfg is not None:
        base_dir = Path(data_dir_cfg).expanduser().resolve()
    else:
        base_dir = Path(default_dir).expanduser().resolve()

    directories_cfg = data_config.get("directories")
    include_all_parts = data_config.get("include_all_parts", False)
    parts_glob = data_config.get("parts_glob", "part_*")

    directories: List[Path]

    if directories_cfg:
        directories = []
        for entry in directories_cfg:
            path = Path(entry).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            directories.append(path)
        directories = _ensure_paths_exist(directories)
    elif include_all_parts:
        directories = sorted(
            p for p in base_dir.glob(parts_glob) if p.is_dir()
        )
        directories = _ensure_paths_exist(directories)
    elif data_dir_cfg is not None:
        directories = _ensure_paths_exist([Path(data_dir_cfg).expanduser()])
    else:
        directories = _ensure_paths_exist([Path(default_dir).expanduser()])

    return base_dir, directories
