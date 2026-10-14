from pathlib import Path


def resolve_image(value: str, csv_dir: Path, image_root: Path | None = None) -> Path:
    """Resolve moved Windows/POSIX logs without recursively guessing duplicate names."""
    normalized = value.strip().strip('"').replace("\\", "/")
    if not normalized:
        raise FileNotFoundError("Empty camera path")
    source = Path(normalized)
    candidates = []
    if image_root is not None:
        candidates += [image_root / source.name, image_root / normalized]
    candidates += [
        csv_dir / normalized,
        csv_dir / "IMG" / source.name,
        csv_dir / source.name,
        source,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find image {value!r} beside {csv_dir}")
