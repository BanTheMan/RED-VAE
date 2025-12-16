#!/usr/bin/env python3
"""
Inspect and print the directory/file structure of the Mendeley EEG dataset.

Usage:
    python inspect_mendeley_structure.py /path/to/mendeley_root

Optional flags:
    --max-files N    Limit number of files printed per directory (default: 10)
    --max-depth D    Limit recursion depth (default: None = unlimited)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def walk_dir(
    root: Path,
    max_files: int = 10,
    max_depth: int | None = None,
    depth: int = 0,
):
    if max_depth is not None and depth > max_depth:
        return

    indent = "  " * depth
    print(f"{indent}{root.name}/")

    files = []
    dirs = []

    for p in sorted(root.iterdir()):
        if p.is_dir():
            dirs.append(p)
        else:
            files.append(p)

    # Print files (limited)
    for f in files[:max_files]:
        print(f"{indent}  {f.name}")

    if len(files) > max_files:
        print(f"{indent}  ... ({len(files) - max_files} more files)")

    # Recurse into subdirs
    for d in dirs:
        walk_dir(d, max_files, max_depth, depth + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=str,
        help="Path to extracted Mendeley dataset root",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Max files to print per directory",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max directory depth to recurse",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    print(f"\nInspecting Mendeley dataset at:\n  {root}\n")
    walk_dir(root, args.max_files, args.max_depth)


if __name__ == "__main__":
    main()
