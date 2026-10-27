"""Rebase historical project paths in an training archive. Dry-run by default."""
from pathlib import Path
import argparse
import json

OLD = "/Users/bensontan/Desktop/image_detection"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    if not (root / "web/package.json").is_file():
        p.error("root must be the repository directory")
    changed = []
    # Rebase metadata/configuration only. Never modify images, weights, or source code.
    for directory in ("dataset", "runs", "artifacts"):
        for f in (root / directory).rglob("*"):
            if f.is_symlink() or not f.is_file() or f.suffix not in (".json", ".yaml", ".yml"):
                continue
            text = f.read_text()
            protected = text.replace(root.as_posix(), "__CUP_HANDOFF_CURRENT_ROOT__")
            updated = protected.replace(OLD, root.as_posix()).replace("__CUP_HANDOFF_CURRENT_ROOT__", root.as_posix())
            if updated == text:
                continue
            if args.apply:
                backup = f.with_name(f.name + ".original-paths")
                if backup.exists():
                    raise RuntimeError(f"Backup already exists: {backup}")
                backup.write_text(text)
                f.write_text(updated)
            changed.append(f.relative_to(root).as_posix())
    print(json.dumps({"applied": args.apply, "root": str(root), "files": changed}, indent=2))

if __name__ == "__main__":
    main()
