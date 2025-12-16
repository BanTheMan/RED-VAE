#!/usr/bin/env python3
"""
Concatenate source files in a repository into one file.

Examples:
  python concat_repo.py --out repo_dump.txt
  python concat_repo.py --out repo_dump.py --extensions .py .md .yml
  python concat_repo.py --respect-gitignore --out repo_dump.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules",
    "dist", "build", "out",
    ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".idea", ".vscode",
    "target",  # Rust/Java builds
    ".gradle",
}

# A conservative default set of "source/code/config" extensions.
DEFAULT_EXTENSIONS = {
    # Python
    ".py", ".pyi",
    # JS/TS
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    # Web
    ".html", ".css", ".scss", ".sass",
    # C/C++
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh",
    # Java/Kotlin/Scala
    ".java", ".kt", ".kts", ".scala",
    # C#/F#
    ".cs", ".fs", ".fsx",
    # Go/Rust
    ".go", ".rs",
    # Shell
    ".sh", ".bash", ".zsh", ".ps1",
    # Data / config / docs often useful in codebases
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".xml", ".md", ".rst", ".txt",
    ".dockerfile", "Dockerfile",  # handle later too
    ".gitignore", ".gitattributes",
    "Makefile", "CMakeLists.txt",
    ".sql",
}

# Comment style heuristics for headers
LINE_COMMENT_EXTS = {
    # hash-style
    ".py", ".sh", ".bash", ".zsh", ".ps1", ".yml", ".yaml", ".toml", ".ini", ".cfg",
    ".md", ".txt", ".dockerfile",
    ".gitignore", ".gitattributes",
    "Makefile", "CMakeLists.txt",
}
SLASH_COMMENT_EXTS = {
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh",
    ".java", ".kt", ".kts", ".scala",
    ".cs", ".go", ".rs",
    ".css", ".scss", ".sass",
    ".sql",
}
XML_LIKE_EXTS = {".xml", ".html"}


def is_probably_binary(path: Path, sample_size: int = 8192) -> bool:
    try:
        data = path.read_bytes()[:sample_size]
    except Exception:
        # If we can't read it, skip it.
        return True

    if not data:
        return False

    # If there are null bytes, very likely binary.
    if b"\x00" in data:
        return True

    # Heuristic: if too many non-text bytes, treat as binary
    # (very lenient; avoids skipping UTF-8 code).
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
    nontext = sum(b not in text_chars for b in data)
    return nontext / len(data) > 0.30


def choose_header_comment(relpath: str, filename: str) -> str:
    """
    Returns a header line (or small block) that marks the start of a file.
    """
    name_key = filename if filename in {"Dockerfile", "Makefile", "CMakeLists.txt"} else Path(filename).suffix

    if name_key in XML_LIKE_EXTS:
        return f"<!-- FILE: {relpath} -->\n"
    if name_key in SLASH_COMMENT_EXTS:
        return f"// FILE: {relpath}\n"
    # default to hash comment (works fine in a .txt too)
    return f"# FILE: {relpath}\n"


def matches_extensions(path: Path, extensions: set[str]) -> bool:
    # Handle special filenames like Dockerfile/Makefile
    if path.name in extensions:
        return True
    # Normal extension match
    return path.suffix.lower() in extensions


def load_gitignore_patterns(repo_root: Path) -> list[str]:
    """
    Minimal gitignore support: reads .gitignore lines (no advanced semantics).
    If you need full fidelity, you can install `pathspec` and use that instead.
    """
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        return []
    patterns: list[str] = []
    for line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        patterns.append(s)
    return patterns


def is_ignored_simple(rel_posix: str, patterns: list[str]) -> bool:
    """
    Very simple ignore matcher:
      - supports trailing '/' as directory prefix
      - supports '*' wildcard in a basic way
    Not a full gitignore implementation.
    """
    if not patterns:
        return False

    import fnmatch

    for pat in patterns:
        p = pat.replace("\\", "/")

        # Directory pattern
        if p.endswith("/"):
            if rel_posix.startswith(p):
                return True
            continue

        # Rooted pattern
        if p.startswith("/"):
            p2 = p.lstrip("/")
            if fnmatch.fnmatch(rel_posix, p2):
                return True
            continue

        # General pattern (match anywhere)
        if fnmatch.fnmatch(rel_posix, p) or fnmatch.fnmatch(Path(rel_posix).name, p):
            return True

    return False


def iter_files(repo_root: Path, extensions: set[str], skip_dirs: set[str], gitignore_patterns: list[str]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune skip directories early (explicit + dot-directories)
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs and not d.startswith(".")
        ]

        for fn in filenames:
            # Skip dotfiles
            if fn.startswith("."):
                continue

            if fn.endswith(".json"):
                continue

            # Skip __init__.py
            if fn == "__init__.py":
                continue

            path = Path(dirpath) / fn
            rel = path.relative_to(repo_root).as_posix()

            if gitignore_patterns and is_ignored_simple(rel, gitignore_patterns):
                continue

            if not matches_extensions(path, extensions):
                continue

            yield path



def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repository root directory (default: .)")
    ap.add_argument("--out", default="repo_concat.txt", help="Output file path")
    ap.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="File extensions or special names to include (e.g., .py .js Dockerfile Makefile). "
             "If omitted, uses a sensible default set.",
    )
    ap.add_argument("--skip-dirs", nargs="*", default=None, help="Directory names to skip (default: common build/cache dirs)")
    ap.add_argument("--max-bytes", type=int, default=2_000_000, help="Skip files larger than this many bytes (default: 2,000,000)")
    ap.add_argument("--respect-gitignore", action="store_true", help="Try to respect .gitignore (simple matching)")
    ap.add_argument("--no-header", action="store_true", help="Do not add per-file header comments")
    args = ap.parse_args(argv)

    repo_root = (Path(args.root) / "src").resolve()
    out_path = Path(args.out).resolve()

    extensions = set(DEFAULT_EXTENSIONS) if args.extensions is None else set(args.extensions)
    skip_dirs = set(DEFAULT_SKIP_DIRS) if args.skip_dirs is None else set(args.skip_dirs)

    gitignore_patterns: list[str] = []
    if args.respect_gitignore:
        gitignore_patterns = load_gitignore_patterns(repo_root)

    files = sorted(iter_files(repo_root, extensions, skip_dirs, gitignore_patterns), key=lambda p: p.as_posix())

    # Avoid writing output into itself if it's under the repo root and matches extensions.
    try:
        out_rel = out_path.relative_to(repo_root).as_posix()
    except Exception:
        out_rel = None

    with out_path.open("w", encoding="utf-8", errors="replace", newline="\n") as out:
        out.write(f"### Repository concat from: {repo_root}\n")
        out.write(f"### Files included: {len(files)}\n\n")

        for path in files:
            if out_rel is not None and path.resolve() == out_path:
                continue

            try:
                size = path.stat().st_size
            except Exception:
                continue

            if size > args.max_bytes:
                # Write a stub so you know it existed
                rel = path.relative_to(repo_root).as_posix()
                out.write("\n" + ("#" * 80) + "\n")
                out.write(f"# SKIPPED (too large: {size} bytes): {rel}\n")
                out.write(("#" * 80) + "\n\n")
                continue

            if is_probably_binary(path):
                rel = path.relative_to(repo_root).as_posix()
                out.write("\n" + ("#" * 80) + "\n")
                out.write(f"# SKIPPED (binary or unreadable): {rel}\n")
                out.write(("#" * 80) + "\n\n")
                continue

            rel = path.relative_to(repo_root).as_posix()
            out.write("\n" + ("#" * 80) + "\n")
            if not args.no_header:
                out.write(choose_header_comment(rel, path.name))
            out.write(("#" * 80) + "\n\n")

            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                out.write(f"# ERROR reading file: {rel}\n")
                continue

            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")

    print(f"Wrote {len(files)} file(s) into: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
