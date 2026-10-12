"""Summarise the tau2 cells of a ``run_benchmark.py`` run as paired tables.

Reads a ``run_benchmark.py`` manifest, loads each tau2 cell's ``results.json``
and compares two arms per (domain, agent):

* the ``enforce`` pair (the default run): ``enforce-off`` against
  ``enforce-on``.  Both arms have the natural-language policy in the agent's
  prompt and the same instrumented code path; ``enforce-off`` pins
  ``experiments/translation/allow_all.dl``, ``enforce-on`` pins the domain policy.
* the ``legacy`` pair: ``baseline`` (plain tau2, NI = non-instrumented) against
  ``sasy`` (policy text hidden, Datalog enforcement, I = instrumented).

``--pair auto`` (the default) takes the enforce pair when the manifest has any
of its cells, else the legacy pair.  Judge cells are not part of either pair.

Outputs (Markdown on stdout, ``--md`` to a file):

Paired table
    One row per (domain, agent): trials per arm, mean reward per arm and the
    difference (on minus off), and pass^k per arm for k = 1 .. the trial count.
    pass^k is tau2's consistency measure: the chance that k trials of a task
    drawn at random all succeed, averaged over tasks; pass^1 is the success
    rate.
Overhead
    Mean wall-clock time and mean API cost per trial (agent plus simulated user).
Termination reasons
    How every trial ended.  A trial tau2 cut short (``infrastructure_error``,
    ``timeout``, ...) counts as a failed trial everywhere above.
Per-task pass rates (``--per-task``)
    One row per task, ``passes/trials`` per arm; written for the curated
    three-task subset, long for a full domain.

Usage::

    uv run python scripts/evaluation/summarize_tau2.py \\
        results/<ts>/manifest.json \\
        --md results/<ts>/tau2.md

Optional ``--task-labels`` reads a JSON file ``{"7": "Booking", ...}``
to substitute paper-friendly row labels for raw tau2 task IDs in the
per-task table.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tau2_results import (
    abnormal_terminations,
    cell_results_file,
    is_success,
    pass_hat_k,
    resolve_results_file,
    termination_counts,
)

# ── Static config ────────────────────────────────────────────────────

# Display order matches the paper layout: Claude, GPT, Gemini.
_MODEL_ORDER = ("opus", "gpt", "gemini")
_MODEL_LABEL = {
    "opus":   "Claude",
    "gpt":    "GPT",
    "gemini": "Gemini",
}

# Domain order in Table 4: airline first (left block), retail right.
_DOMAIN_ORDER = ("airline", "retail")
_DOMAIN_LABEL = {"airline": "Airline", "retail": "Retail"}

# Each pair maps two manifest modes onto the internal arm keys "NI"
# (the comparison arm) and "I" (the enforcing arm); judge mode is in
# neither. ``ARM_LABELS`` is how each pair's arms are printed.
PAIRS = {
    "enforce": {"enforce-off": "NI", "enforce-on": "I"},
    "legacy": {"baseline": "NI", "sasy": "I"},
}
ARM_LABELS = {"enforce": ("Off", "On"), "legacy": ("NI", "I")}
_MODE_TO_NI_I = PAIRS["legacy"]


def detect_pair(manifest: dict) -> str:
    """``enforce`` when the manifest has an enforce-off/-on tau2 cell, else ``legacy``."""
    modes = {c.get("mode") for c in manifest.get("cells", []) if c.get("target") == "tau2"}
    return "enforce" if modes & set(PAIRS["enforce"]) else "legacy"

# Default labels for selected airline and retail tasks. Use --task-labels
# to supply labels for other task sets.
_DEFAULT_TASK_LABELS: dict[str, str] = {
    # Airline (matches description.purpose first sentence)
    "8":  "Booking",          # Booking with extra passenger
    "43": "User Error",       # Reframed cancellation (accidental booking)
    "47": "Social Event",     # Insurance covers only health/weather, not events
    # Retail (curated; description.purpose is None upstream)
    "15": "Order Update",     # Modify pending order items / address
    "16": "Multi-Order",      # Cancel + return spanning multiple orders
    "91": "Return/Exchange",  # Return-then-exchange with prerequisite checks
}


# ── Per-cell aggregation ─────────────────────────────────────────────


@dataclass
class TaskStat:
    passes: int = 0
    total: int = 0
    reward_sum: float = 0.0


@dataclass
class CellData:
    """All trials for one (domain, model, ni_i) bucket."""
    by_task: dict[str, TaskStat] = field(default_factory=lambda: defaultdict(TaskStat))
    runtime_samples: list = field(default_factory=list)
    cost_total_usd: float = 0.0
    reward_total: float = 0.0
    n_trials: int = 0  # for runtime/cost/reward denominator
    # How every trial ended (tau2's termination_reason). A trial tau2
    # cut short -- infrastructure_error, timeout, context_window_exceeded,
    # ... -- still counts as a (failed) trial above; this says how many.
    terminations: Counter = field(default_factory=Counter)

    def add_trial(self, task_id: str, reward: float, duration_s: float, cost_usd: float) -> None:
        ts = self.by_task[task_id]
        ts.total += 1
        ts.reward_sum += reward
        if is_success(reward):
            ts.passes += 1
        self.runtime_samples.append(duration_s)
        self.cost_total_usd += cost_usd
        self.reward_total += reward
        self.n_trials += 1

    @property
    def mean_reward(self) -> float:
        return self.reward_total / self.n_trials if self.n_trials else 0.0

    @property
    def max_k(self) -> int:
        """The largest k every task has trials for (0 with no trials)."""
        return min((ts.total for ts in self.by_task.values()), default=0)

    def pass_hat(self, k: int) -> float | None:
        return pass_hat_k(((ts.passes, ts.total) for ts in self.by_task.values()), k)

    @property
    def median_runtime_s(self) -> float:
        if not self.runtime_samples:
            return 0.0
        s = sorted(self.runtime_samples)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

    @property
    def mean_runtime_s(self) -> float:
        return sum(self.runtime_samples) / len(self.runtime_samples) if self.runtime_samples else 0.0

    @property
    def runtime_min(self) -> float:
        return min(self.runtime_samples) if self.runtime_samples else 0.0

    @property
    def runtime_max(self) -> float:
        return max(self.runtime_samples) if self.runtime_samples else 0.0

    @property
    def mean_cost_usd(self) -> float:
        return self.cost_total_usd / self.n_trials if self.n_trials else 0.0


def _load_sim(sim_path: Path) -> list[dict]:
    """Return the list of simulation records from a tau2 sim JSON.

    ``sim_path`` may name either layout (``<save_to>/results.json`` from
    tau2 1.0, or the older flat ``<save_to>.json``); whichever exists is read.
    """
    sim_path = resolve_results_file(sim_path)
    if not sim_path.exists():
        return []
    data = json.loads(sim_path.read_text())
    return data.get("simulations", []) or []


def _aggregate(
    manifest: dict,
    manifest_dir: Path | None = None,
    pair: str | None = None,
) -> dict[tuple[str, str, str], CellData]:
    """Bucket trials by (domain, model, NI/I) for one pair of modes.

    ``pair`` is a key of ``PAIRS``; ``None`` picks it with
    :func:`detect_pair`. Cells of other modes (judge, the other pair) and
    cells that did not finish are skipped.
    """
    mode_to_arm = PAIRS[pair or detect_pair(manifest)]
    run_dir = manifest.get("run_dir")
    out: dict[tuple[str, str, str], CellData] = defaultdict(CellData)
    for cell in manifest.get("cells", []):
        if cell.get("target") != "tau2" or cell.get("returncode") != 0:
            continue
        domain = cell["domain"]
        model = cell["model"]
        ni_i = mode_to_arm.get(cell["mode"])
        if ni_i is None:
            continue  # judge, the other pair, etc.
        # Recorded relative to the manifest and absolute; a run directory
        # that was moved with its manifest still resolves. Pass run_dir
        # to prefer local copies over original absolute paths.
        p = cell_results_file(cell, manifest_dir, run_dir=run_dir)
        if p is None:
            continue
        sims = _load_sim(p)
        bucket = out[(domain, model, ni_i)]
        bucket.terminations.update(termination_counts(sims))
        for s in sims:
            task_id = str(s.get("task_id"))
            # An infrastructure_error simulation carries reward_info null;
            # tau2 1.0 costs are None when the provider reported none.
            reward = float((s.get("reward_info") or {}).get("reward") or 0.0)
            duration = float(s.get("duration") or 0.0)
            cost = float(s.get("agent_cost") or 0.0) + float(s.get("user_cost") or 0.0)
            bucket.add_trial(task_id, reward, duration, cost)
    return out


# ── Output formatting ────────────────────────────────────────────────


def _frac(num: int, denom: int) -> str:
    return f"{num}/{denom}"



def _markdown(
    aggs: dict[tuple[str, str, str], CellData],
    task_labels: dict[str, str],
    labels: tuple[str, str] = ARM_LABELS["legacy"],
    per_task: bool = True,
) -> str:
    """Plain-Markdown dump: the paired table, overhead, terminations, and
    (``per_task``) the per-task pass rates."""
    off, on = labels
    lines: list[str] = _paired_markdown(aggs, labels)
    for domain in (_DOMAIN_ORDER if per_task else ()):
        task_ids: set[str] = set()
        for (d, _m, _ni_i), bucket in aggs.items():
            if d == domain:
                task_ids.update(bucket.by_task.keys())
        sorted_ids = sorted(task_ids, key=lambda x: int(x) if x.isdigit() else x)
        lines.append(f"### {_DOMAIN_LABEL[domain]} pass rates by task")
        lines.append("")
        header = ["Task"]
        for m in _MODEL_ORDER:
            header.extend([f"{_MODEL_LABEL[m]} {off}", f"{_MODEL_LABEL[m]} {on}"])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] + ["---:"] * (2 * len(_MODEL_ORDER))) + "|")
        totals: dict[tuple[str, str], TaskStat] = {
            (m, ni_i): TaskStat() for m in _MODEL_ORDER for ni_i in ("NI", "I")
        }
        for tid in sorted_ids:
            row = [task_labels.get(tid, _DEFAULT_TASK_LABELS.get(tid, f"Task {tid}"))]
            for m in _MODEL_ORDER:
                a_ni = aggs.get((domain, m, "NI"), CellData())
                a_i  = aggs.get((domain, m, "I"),  CellData())
                ts_ni = a_ni.by_task.get(tid, TaskStat())
                ts_i  = a_i.by_task.get(tid,  TaskStat())
                row.append(_frac(ts_ni.passes, ts_ni.total) if ts_ni.total else "—")
                row.append(_frac(ts_i.passes, ts_i.total) if ts_i.total else "—")
                totals[(m, "NI")].passes += ts_ni.passes
                totals[(m, "NI")].total  += ts_ni.total
                totals[(m, "I")].passes  += ts_i.passes
                totals[(m, "I")].total   += ts_i.total
            lines.append("| " + " | ".join(row) + " |")
        row = ["**Total**"]
        for m in _MODEL_ORDER:
            ts_ni = totals[(m, "NI")]
            ts_i  = totals[(m, "I")]
            row.append(_frac(ts_ni.passes, ts_ni.total) if ts_ni.total else "—")
            row.append(_frac(ts_i.passes, ts_i.total) if ts_i.total else "—")
        lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("### Mean runtime per trial (s)")
    lines.append("")
    lines.append(f"| Domain | Model | {off} | {on} |")
    lines.append("|---|---|---:|---:|")
    for domain in _DOMAIN_ORDER:
        for mi, m in enumerate(_MODEL_ORDER):
            a_ni = aggs.get((domain, m, "NI"), CellData())
            a_i  = aggs.get((domain, m, "I"),  CellData())
            v_ni = f"{a_ni.mean_runtime_s:.1f}" if a_ni.n_trials else "—"
            v_i  = f"{a_i.mean_runtime_s:.1f}"  if a_i.n_trials  else "—"
            lines.append(f"| {_DOMAIN_LABEL[domain] if mi == 0 else ''} | "
                         f"{_MODEL_LABEL[m]} | {v_ni} | {v_i} |")
    lines.append("")
    lines.append("### Mean API cost per trial ($)")
    lines.append("")
    lines.append(f"| Domain | Model | {off} | {on} |")
    lines.append("|---|---|---:|---:|")
    for domain in _DOMAIN_ORDER:
        for mi, m in enumerate(_MODEL_ORDER):
            a_ni = aggs.get((domain, m, "NI"), CellData())
            a_i  = aggs.get((domain, m, "I"),  CellData())
            v_ni = f"${a_ni.mean_cost_usd:.3f}" if a_ni.n_trials else "—"
            v_i  = f"${a_i.mean_cost_usd:.3f}"  if a_i.n_trials  else "—"
            lines.append(f"| {_DOMAIN_LABEL[domain] if mi == 0 else ''} | "
                         f"{_MODEL_LABEL[m]} | {v_ni} | {v_i} |")
    lines.append("")
    lines.extend(_terminations_markdown(aggs, labels))
    lines.append("")
    lines.extend(_supplement_markdown(aggs, labels))
    return "\n".join(lines)


def _pair_rows(aggs: dict[tuple[str, str, str], CellData]):
    """(domain, model, off CellData, on CellData, k) for every pair with data."""
    for domain in _DOMAIN_ORDER:
        for m in _MODEL_ORDER:
            off = aggs.get((domain, m, "NI"))
            on = aggs.get((domain, m, "I"))
            if not (off and off.n_trials) and not (on and on.n_trials):
                continue
            off, on = off or CellData(), on or CellData()
            k = max(off.max_k, on.max_k)
            yield domain, m, off, on, k


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _paired_markdown(
    aggs: dict[tuple[str, str, str], CellData],
    labels: tuple[str, str] = ARM_LABELS["legacy"],
) -> list[str]:
    """Mean reward and pass^k per (domain, agent), the two arms side by side."""
    off_l, on_l = labels
    rows = list(_pair_rows(aggs))
    k_max = max((k for *_, k in rows), default=0)
    lines = [f"### Paired comparison ({off_l} vs {on_l})", ""]
    header = ["Domain", "Agent", f"Trials {off_l}/{on_l}",
              f"Reward {off_l}", f"Reward {on_l}", "Δ reward"]
    for k in range(1, k_max + 1):
        header += [f"pass^{k} {off_l}", f"pass^{k} {on_l}"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---", "---"] + ["---:"] * (len(header) - 2)) + "|")
    for domain, m, off, on, _k in rows:
        delta = (on.mean_reward - off.mean_reward) if (off.n_trials and on.n_trials) else None
        row = [
            _DOMAIN_LABEL[domain], _MODEL_LABEL[m], f"{off.n_trials}/{on.n_trials}",
            _fmt(off.mean_reward if off.n_trials else None),
            _fmt(on.mean_reward if on.n_trials else None),
            "—" if delta is None else f"{delta:+.3f}",
        ]
        for k in range(1, k_max + 1):
            row += [_fmt(off.pass_hat(k)), _fmt(on.pass_hat(k))]
        lines.append("| " + " | ".join(row) + " |")
    if not rows:
        lines.append("| (no finished cells of this pair) |")
    lines.append("")
    return lines


def _terminations_markdown(
    aggs: dict[tuple[str, str, str], CellData],
    labels: tuple[str, str] = ARM_LABELS["legacy"],
) -> list[str]:
    """How every trial ended, per cell, so a trial tau2 cut short is visible.

    Such trials (infrastructure_error, timeout, context_window_exceeded,
    unexpected_error, ...) are counted as failed trials in Table 4 and in
    the Table 5 means; this table says how many there were.
    """
    lines = ["### Termination reasons", ""]
    lines.append("| Domain | Model | Mode | Trials | Termination reasons | Cut short |")
    lines.append("|---|---|---|---:|---|---:|")
    for (domain, model, ni_i), bucket in sorted(aggs.items()):
        counts = dict(sorted(bucket.terminations.items()))
        reasons = ", ".join(f"{reason}={n}" for reason, n in counts.items()) or "—"
        cut_short = sum(abnormal_terminations(counts).values())
        arm = labels[0] if ni_i == "NI" else labels[1]
        lines.append(f"| {domain} | {model} | {arm} | {bucket.n_trials} | {reasons} | {cut_short} |")
    return lines


def _supplement_markdown(
    aggs: dict[tuple[str, str, str], CellData],
    labels: tuple[str, str] = ARM_LABELS["legacy"],
) -> list[str]:
    """Runtime statistics supplement: N, mean, min-max per cell."""
    lines = ["### Supplementary: Runtime statistics per group", ""]
    lines.append("| Domain | Model | Mode | N | Mean (s) | Min (s) | Max (s) |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for domain in _DOMAIN_ORDER:
        for m in _MODEL_ORDER:
            for ni_i, arm_label in [("NI", labels[0]), ("I", labels[1])]:
                bucket = aggs.get((domain, m, ni_i))
                if not bucket or not bucket.n_trials:
                    continue
                lines.append(
                    f"| {_DOMAIN_LABEL[domain]} | {_MODEL_LABEL[m]} | {arm_label} | "
                    f"{bucket.n_trials} | {bucket.mean_runtime_s:.1f} | "
                    f"{bucket.runtime_min:.1f} | {bucket.runtime_max:.1f} |"
                )
    return lines


# ── Entry point ──────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path,
                        help="Path to a run_benchmark.py manifest.json")
    parser.add_argument("--pair", choices=("auto", *PAIRS), default="auto",
                        help="Which two modes to compare: enforce (enforce-off"
                        " vs enforce-on), legacy (baseline vs sasy), or auto"
                        " (enforce when the manifest has it; the default).")
    parser.add_argument("--per-task", action="store_true",
                        help="Also print the per-task pass-rate table.")
    parser.add_argument("--md", type=Path, default=None,
                        help="Write Markdown to this path.")
    parser.add_argument("--task-labels", type=Path, default=None,
                        help="Optional JSON file mapping task IDs to "
                        "paper-friendly labels, e.g. "
                        "{\"7\": \"Booking\", \"8\": \"User Error Cancel\"}.")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    task_labels: dict[str, str] = {}
    if args.task_labels:
        task_labels = {
            str(k): str(v)
            for k, v in json.loads(args.task_labels.read_text()).items()
        }


    pair = detect_pair(manifest) if args.pair == "auto" else args.pair
    labels = ARM_LABELS[pair]
    aggs = _aggregate(manifest, manifest_dir=args.manifest.resolve().parent, pair=pair)
    modes = " vs ".join(PAIRS[pair])
    print(f"pair: {pair} ({modes})", file=sys.stderr)
    for (domain, model, ni_i), bucket in sorted(aggs.items()):
        cut = abnormal_terminations(dict(bucket.terminations))
        if cut:
            print(f"WARN {domain}/{model}/{ni_i}: trials cut short, counted as failures: {cut}",
                  file=sys.stderr)

    md = _markdown(aggs, task_labels, labels, per_task=args.per_task)

    print(md)

    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md + "\n")
        print(f"\nMarkdown → {args.md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
