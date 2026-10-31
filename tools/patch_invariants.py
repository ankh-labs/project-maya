#!/usr/bin/env python3
"""Check that the expressions this fork depends on survived a replay.

    python tools/patch_invariants.py [--rev <sha>] [--manifest <file>]

Exit 0 only if every entry matches its expected occurrence count. FAILS CLOSED: a missing file, a malformed line, a
non-integer count, or an unknown status is a failure - never a skip. This exists because five hunks were silently
dropped by the v1.1.40/v1.1.41 replays and every one of them compiled; see the header of the manifest.
"""

import argparse
import subprocess
import sys

MANIFEST = "tools/patch_invariants.txt"
STATUSES = {"REQUIRED", "ACKNOWLEDGED-DIVERGENT"}


def read_file(path, rev):
    if rev:
        r = subprocess.run(
            ["git", "show", f"{rev}:{path}"], capture_output=True, text=True
        )
        if r.returncode != 0:
            return None
        return r.stdout
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rev", default=None, help="check a git revision instead of the working tree"
    )
    ap.add_argument("--manifest", default=MANIFEST)
    args = ap.parse_args()

    try:
        with open(args.manifest, encoding="utf-8") as f:
            raw = f.read().splitlines()
    except OSError as e:
        print(f"FAIL: cannot read manifest {args.manifest}: {e}")
        return 2

    entries = []
    errors = []
    for n, line in enumerate(raw, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            errors.append(
                f"{args.manifest}:{n}: malformed (expected 4 '|'-separated fields, got {len(parts)})"
            )
            continue
        pattern, path, expected, status = parts
        if not pattern or not path:
            errors.append(f"{args.manifest}:{n}: empty pattern or path")
            continue
        if status not in STATUSES:
            errors.append(f"{args.manifest}:{n}: unknown status {status!r}")
            continue
        try:
            want = int(expected)
        except ValueError:
            errors.append(
                f"{args.manifest}:{n}: expected count {expected!r} is not an integer"
            )
            continue
        entries.append((n, pattern, path, want, status))

    if not entries and not errors:
        print("FAIL: manifest has no entries - an empty manifest proves nothing")
        return 2

    cache = {}
    failures = []
    for n, pattern, path, want, status in entries:
        if path not in cache:
            cache[path] = read_file(path, args.rev)
        text = cache[path]
        if text is None:
            failures.append(
                f"{args.manifest}:{n}: cannot read {path}"
                + (f" at {args.rev}" if args.rev else "")
            )
            continue
        lines = text.splitlines()
        hits = [i + 1 for i, l in enumerate(lines) if pattern in l]
        got = len(hits)
        if got != want:
            where = ", ".join(str(x) for x in hits[:6]) or "none"
            failures.append(
                f"{path}: expected {want} of {pattern!r}, found {got} (lines: {where})   [{status}]"
            )
        else:
            print(f"ok   {status:<23} {got:>2}  {pattern[:64]}")

    if errors:
        print()
        for e in errors:
            print("FAIL " + e)
    if failures:
        print()
        for f in failures:
            print("FAIL " + f)

    if errors or failures:
        print(f"\nFAILED: {len(errors)} malformed, {len(failures)} mismatched")
        return 1
    print(
        f"\nPASS: {len(entries)} invariants hold"
        + (f" at {args.rev}" if args.rev else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
