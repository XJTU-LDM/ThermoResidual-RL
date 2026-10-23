"""Artifact hashing and immutable JSON output helpers."""
import hashlib
import json
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, *, frozen=False):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if frozen and path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Frozen artifact changed; create a new revision: {path}")
        return
    path.write_text(text, encoding="utf-8")

