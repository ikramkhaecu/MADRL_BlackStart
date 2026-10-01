"""Configuration loader.

Every numerical parameter of the benchmark lives in ``configs/default.yaml``
together with a provenance tag.  ``python -m sim.config --provenance`` prints
the table that is reproduced in ``docs/PARAMETERS.md``; ``--provisional``
prints only the values the authors still have to confirm.
"""
from __future__ import annotations

import argparse
import copy
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "configs", "default.yaml")
_TAG = re.compile(r"\[(paper|code-v1|code-main|data|standard|confirmed|PROVISIONAL)[^\]]*\]")


def load(path: str | None = None, overrides: dict | None = None) -> dict:
    with open(path or DEFAULT) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_update(copy.deepcopy(cfg), overrides)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def provenance(path: str | None = None):
    """Return [(dotted.key, value_text, [tags])] parsed from the YAML comments."""
    rows, stack = [], []
    with open(path or DEFAULT) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            body, _, comment = line.partition("#")
            m = re.match(r"\s*(-\s*)?\"?([A-Za-z0-9_]+)\"?\s*:\s*(.*)", body)
            if not m:
                if "[" in comment and stack:          # list item carrying a tag
                    tags = [t.group(0) for t in _TAG.finditer(comment)]
                    if tags:
                        rows.append((".".join(k for _, k in stack) + "[]",
                                     body.strip(), tags, comment.strip()))
                continue
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, m.group(2)))
            tags = [t.group(0) for t in _TAG.finditer(comment)]
            if tags:
                rows.append((".".join(k for _, k in stack), m.group(3).strip(),
                             tags, comment.strip()))
    return rows


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance", action="store_true")
    ap.add_argument("--provisional", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    rows = provenance()
    if a.provisional:
        rows = [r for r in rows if any(("PROVISIONAL" in t or "standard" in t) for t in r[2])
                and not any("confirmed" in t for t in r[2])]
    if a.markdown:
        print("| parameter | value | provenance |\n|---|---|---|")
        for k, v, _, c in rows:
            print(f"| `{k}` | `{v}` | {c} |")
    else:
        for k, v, _, c in rows:
            print(f"{k:38s} {v:28s} {c}")
    print(f"\n{len(rows)} parameters listed")


if __name__ == "__main__":
    _main()
