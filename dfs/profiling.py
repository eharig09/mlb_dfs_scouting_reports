"""Pipeline stage timing, written out in a machine-readable form.

Two rules shaped this:

* **Off costs nothing.** Instrumentation that slows the nightly build would be removed
  within a week. When profiling is off, `stage()` returns a shared no-op context manager
  and the whole thing is a few nanoseconds per call site. Measured overhead when *on* is
  under 1% of a slate build.
* **Nesting is real.** "projection took 1.1s" is useless without knowing it sits inside
  "slate build 2.0s". Stages nest, and the JSON keeps the tree, so a report can show both
  self time and cumulative time.

Usage::

    from .profiling import profiler

    with profiler.session("board", date="2026-08-01", slate="main"):
        with profiler.stage("acquire"):
            ...
        with profiler.stage("project"):
            ...
    # -> docs/benchmarks/profile_board_2026-08-01_main.json

Nothing outside a session records anything, so importing this module into a library that
is used without profiling is free.
"""

import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

BENCHMARK_DIR = os.path.join("docs", "benchmarks")

# The canonical stage names, in pipeline order. Callers are not restricted to these -- an
# unknown name is recorded as-is -- but using them keeps reports comparable across runs.
STAGES = [
    "acquire",          # API / network
    "normalize",        # parsing, alias folding, joins
    "features",         # feature construction
    "project",          # player projections
    "ownership",        # field ownership model
    "candidates",       # candidate lineup generation
    "solve",            # solver execution
    "simulate",         # correlated slate simulation
    "portfolio",        # portfolio selection
    "render",           # markdown / xlsx / csv output
]


class _Node:
    """One timed stage and its children."""

    __slots__ = ("name", "start", "elapsed", "children", "count", "meta")

    def __init__(self, name):
        self.name = name
        self.start = None
        self.elapsed = 0.0
        self.count = 0
        self.children = {}
        self.meta = {}

    def child(self, name):
        node = self.children.get(name)
        if node is None:
            node = self.children[name] = _Node(name)
        return node

    def as_dict(self):
        children = [c.as_dict() for c in self.children.values()]
        # Self time is what this stage spent outside its own children. It is the number
        # that actually points at the code to fix; cumulative time only points at a caller.
        child_total = sum(c["cumulative_s"] for c in children)
        record = {
            "stage": self.name,
            "cumulative_s": round(self.elapsed, 6),
            "self_s": round(max(0.0, self.elapsed - child_total), 6),
            "calls": self.count,
        }
        if self.meta:
            record["meta"] = self.meta
        if children:
            record["children"] = children
        return record


class _NullContext:
    """Returned by stage() when profiling is off. Also accepts .note() so call sites
    do not have to branch on whether a session is active."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def note(self, **_kwargs):
        pass


_NULL = _NullContext()


class Profiler:
    """Nested stage timing for one pipeline run.

    A single module-level instance is shared, because the pipeline is single-threaded per
    run and threading a profiler object through nine modules would be worse than the
    problem it solves. Sessions do not nest; starting one inside another is ignored so a
    library call that opens its own session cannot clobber the caller's.
    """

    def __init__(self):
        self._root = None
        self._stack = []
        self._info = {}

    @property
    def active(self):
        return self._root is not None

    @contextmanager
    def session(self, label, date=None, slate=None, out_dir=BENCHMARK_DIR,
                enabled=True, write=True, **info):
        """Time a whole pipeline run and write the tree to JSON on exit."""
        if not enabled or self.active:
            yield self
            return

        self._root = _Node(label)
        self._stack = [self._root]
        self._info = {
            "label": label,
            "date": date,
            "slate": slate,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **{k: v for k, v in info.items() if v is not None},
        }
        self._root.start = time.perf_counter()
        try:
            yield self
        finally:
            self._root.elapsed = time.perf_counter() - self._root.start
            self._root.count = 1
            report = self.report()
            self._root, self._stack = None, []
            if write:
                try:
                    self._write(report, out_dir, label, date, slate)
                except OSError as error:              # never fail a build over telemetry
                    sys.stderr.write(f"[profiling] could not write report: {error}\n")
            self.last_report = report

    @contextmanager
    def stage(self, name, **meta):
        """Time one stage. Re-entering the same name accumulates rather than replacing."""
        if not self.active:
            yield _NULL
            return
        node = self._stack[-1].child(name)
        node.meta.update({k: v for k, v in meta.items() if v is not None})
        self._stack.append(node)
        start = time.perf_counter()
        try:
            yield node
        finally:
            node.elapsed += time.perf_counter() - start
            node.count += 1
            self._stack.pop()

    def note(self, **meta):
        """Attach metadata (row counts, pool size, sim count) to the current stage."""
        if self.active:
            self._stack[-1].meta.update({k: v for k, v in meta.items() if v is not None})

    def report(self):
        if self._root is None:
            return {}
        return {"run": dict(self._info), "profile": self._root.as_dict()}

    @staticmethod
    def _write(report, out_dir, label, date, slate):
        os.makedirs(out_dir, exist_ok=True)
        parts = [p for p in ("profile", label, date, slate) if p]
        path = os.path.join(out_dir, "_".join(str(p) for p in parts) + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        report["path"] = path
        return path


profiler = Profiler()


def format_report(report, indent=0, min_share=0.0):
    """Human-readable tree from a report dict, deepest times first at each level."""
    if not report:
        return "(no profile recorded)"
    root = report.get("profile", report)
    total = root.get("cumulative_s") or 1.0
    lines = []

    def walk(node, depth):
        share = node["cumulative_s"] / total * 100
        if depth and share < min_share:
            return
        calls = f" x{node['calls']}" if node["calls"] > 1 else ""
        meta = ""
        if node.get("meta"):
            meta = "  " + " ".join(f"{k}={v}" for k, v in node["meta"].items())
        lines.append(
            f"{'  ' * depth}{node['stage']:<24}{node['cumulative_s']:8.3f}s "
            f"{share:5.1f}%  self {node['self_s']:7.3f}s{calls}{meta}"
        )
        for child in sorted(node.get("children", []),
                            key=lambda c: -c["cumulative_s"]):
            walk(child, depth + 1)

    walk(root, indent)
    return "\n".join(lines)
