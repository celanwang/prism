"""Verify the frozen inventory with the standard library alone.

Run: python scripts/verify_inventory.py [--root CLONE] [--inventory JSON].
No prism installation, Git process, environment collection or writes are needed.
The inventory's source_commit and git_blob IDs describe the committed raw bytes;
verification compares those SHA-256 values with the checked-out artifact bytes.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def verify(root: Path, inventory: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    data = json.loads(inventory.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if data.get("version") != 1 or not re.fullmatch(r"[0-9a-f]{40}", data["source_commit"]):
        raise ValueError("invalid inventory version/source_commit")
    if not isinstance(data["files"], list) or not data["files"]:
        raise ValueError("empty inventory")
    errors, seen = [], set()
    for entry in data["files"]:
        name = entry["path"]
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or any(p in ("", ".", "..") for p in name.split("/"))
            or name in seen
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or type(entry["size"]) is not int
            or entry["size"] < 0
        ):
            raise ValueError(f"invalid inventory entry: {name}")
        seen.add(name)
        path = root / name
        try:
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("symlink not allowed")
            raw = path.read_bytes()
            if len(raw) != entry["size"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                errors.append(f"hash/size mismatch: {name}")
        except (OSError, ValueError) as error:
            errors.append(f"{name}: {error}")
    return {
        "source_commit": data["source_commit"],
        "checked": len(seen),
        "ok": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.root, args.inventory or args.root / "artifacts/inventory-v1.json")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Inventory failed: {error}\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    parser.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
