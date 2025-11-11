#!/usr/bin/env python3
"""
Generate a clean, formatted view of recent commit messages.

Usage examples:
  - Show last 20 commits in stdout:
        python scripts/fix_commit_messages.py
  - Show last 50 commits:
        python scripts/fix_commit_messages.py --last 50
  - Commits since a date:
        python scripts/fix_commit_messages.py --since 2025-11-01
  - Write to a markdown file:
        python scripts/fix_commit_messages.py -o messages.md
  - Both since and limit can be combined; --since narrows the range before limiting.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import DefaultDict, Dict, List, Optional, Tuple

CONVENTIONAL_TYPES = [
    "feat",
    "fix",
    "perf",
    "refactor",
    "docs",
    "test",
    "build",
    "ci",
    "chore",
    "style",
]

CC_REGEX = re.compile(
    r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?:\s*(?P<subject>.+)$"
)


def run_git_log(
    since: Optional[str], last: int
) -> List[Tuple[str, str, str, str]]:
    """
    Return list of tuples: (hash, author_date_iso, author_name, subject)
    """
    format_str = "%H|%ad|%an|%s"
    cmd = ["git", "log", f"--pretty=format:{format_str}", "--date=iso-strict"]
    if since:
        cmd.append(f"--since={since}")
    # No merges provides a cleaner list; comment out if you want merges included
    cmd.append("--no-merges")
    try:
        out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    except subprocess.CalledProcessError as exc:
        print(f"Failed to run git log: {exc}", file=sys.stderr)
        sys.exit(1)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    entries = []
    for ln in lines[:last]:
        parts = ln.split("|", 3)
        if len(parts) != 4:
            # fallback: skip malformed lines
            continue
        entries.append(tuple(parts))  # type: ignore[arg-type]
    return entries


def normalize_subject(subject: str) -> str:
    s = subject.strip()
    # Collapse consecutive spaces
    s = re.sub(r"\s+", " ", s)
    # Trim trailing dots (avoid aggressive punctuation changes)
    s = s.rstrip(".。")
    return s


def group_by_conventional(entries: List[Tuple[str, str, str, str]]):
    grouped: DefaultDict[str, List[Tuple[str, str, str, str, Dict[str, str]]]] = defaultdict(list)
    for h, date_iso, author, subject in entries:
        subj = normalize_subject(subject)
        m = CC_REGEX.match(subj)
        meta: Dict[str, str] = {}
        if m:
            typ = m.group("type").lower()
            scope = m.group("scope") or ""
            breaking = "!" if m.group("breaking") else ""
            subject_clean = m.group("subject")
            meta = {"scope": scope, "breaking": breaking, "subject": subject_clean}
            key = typ if typ in CONVENTIONAL_TYPES else "other"
        else:
            key = "other"
        grouped[key].append((h, date_iso, author, subj, meta))
    return grouped


def render_markdown(
    grouped, title: str, repo_name: Optional[str], show_hash: bool
) -> str:
    lines: List[str] = []
    header_title = title or "Recent Commits"
    if repo_name:
        header_title = f"{repo_name} - {header_title}"
    lines.append(f"## {header_title}")
    lines.append("")
    order = [t for t in CONVENTIONAL_TYPES if t in grouped] + [
        k for k in grouped.keys() if k not in CONVENTIONAL_TYPES
    ]
    for k in order:
        items = grouped[k]
        if not items:
            continue
        section = k.capitalize() if k != "other" else "Others"
        lines.append(f"### {section}")
        lines.append("")
        for h, date_iso, author, subj, meta in items:
            date_short = date_iso.split("T")[0] if "T" in date_iso else date_iso
            display = subj
            if meta.get("subject"):
                display = meta["subject"]
                if meta.get("scope"):
                    display = f"{meta['subject']} ({meta['scope']})"
                if meta.get("breaking"):
                    display += " [BREAKING]"
            if show_hash:
                short = h[:7]
                lines.append(f"- {display} — {author}, {date_short} (`{short}`)")
            else:
                lines.append(f"- {display} — {author}, {date_short}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def detect_repo_name() -> Optional[str]:
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except subprocess.CalledProcessError:
        return None
    base = url.rsplit("/", 1)[-1]
    base = base[:-4] if base.endswith(".git") else base
    return base or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Format recent commit messages into Markdown.")
    parser.add_argument("--since", help="Only include commits since date (e.g. 2025-11-01).")
    parser.add_argument("--last", type=int, default=20, help="Number of commits to include (default: 20).")
    parser.add_argument("-o", "--output", help="Output file path (e.g. messages.md).")
    parser.add_argument("--no-hash", action="store_true", help="Do not show short hash.")
    parser.add_argument("--title", default="Recent Commits", help="Markdown title.")
    args = parser.parse_args()

    entries = run_git_log(args.since, max(args.last, 1))
    grouped = group_by_conventional(entries)
    repo_name = detect_repo_name()
    md = render_markdown(grouped, args.title, repo_name, show_hash=not args.no_hash)

    if args.output:
        out_path = os.path.abspath(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote {out_path}")
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()

