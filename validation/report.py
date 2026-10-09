"""Build docs/VALIDATION_REPORT.md from the campaign's result files.

Every number in the report comes from a result file under
results/validation/ (named next to each table), and MANIFEST.sha256 pins
those files. The narrative findings live in validation/FINDINGS.md and are
included verbatim, so analysis and data are reviewed separately.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "validation"
HIDDEN = 8


def _load(pattern: str) -> list[dict]:
    return [
        json.loads(p.read_text()) | {"_file": str(p.relative_to(ROOT))}
        for p in sorted(RESULTS.glob(pattern))
    ]


def _pct(x: float) -> str:
    return f"{100 * x:.2f} %" if x < 0.1 else f"{100 * x:.0f} %"


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if v == 0 or 1e-3 <= abs(v) < 1e4:
        return f"{v:.4g}"
    return f"{v:.2e}"


def probe_tables(probes: list[dict]) -> list[str]:
    out: list[str] = []
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in probes:
        by_task[r["task"]].append(r)
    for task, rs in sorted(by_task.items()):
        out += [
            f"### {task} (grader v{rs[0]['task_version']})",
            "",
            "| probe | kind | what it is | lang | cases | passed | failing codes | expected | verdict |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        order = {"reference": 0, "boundary": 1, "negative": 2}
        for r in sorted(rs, key=lambda r: (order.get(r["kind"], 3), r["probe"])):
            a = r["assessment"]
            if r["kind"] == "reference":
                verdict = (
                    ("OK, false-reject rate <= " + _pct(a["false_reject_upper_95"]) + " (95 %)")
                    if a["ok"]
                    else f"FALSE REJECTS {r['n_cases'] - r['n_passed']}"
                )
            elif r["kind"] == "boundary":
                verdict = "OK, exact codes" if a["ok"] else "check"
            else:
                bound = a[f"reward_probability_{HIDDEN}_cases_upper"]
                verdict = (
                    f"OK, P(reward) <= {bound:.1e}"
                    if a["ok"]
                    else (
                        "expected code missing"
                        if not a["expected_codes_fired"]
                        else f"weak: P(reward) <= {bound:.2f}"
                    )
                )
            codes = ", ".join(f"{k} x{v}" for k, v in r["fail_codes"].items()) or "-"
            name = r["probe"].split("/", 1)[1]
            out.append(
                f"| `{name}` | {r['kind']} | {r['note']} | {r['lang']} | {r['n_cases']} | "
                f"{r['n_passed']} | {codes} | {', '.join(r['expected_codes']) or '-'} | {verdict} |"
            )
        out += ["", f"Files: `results/validation/{task}/*.json`", ""]
    return out


def margin_tables(probes: list[dict]) -> list[str]:
    """For each measured check: how close the references came to the
    threshold, next to the nearest negative probe on the other side."""
    out = [
        "| task | check | threshold | references (min..max) | negatives (min..max) |",
        "|---|---|---|---|---|",
    ]
    rows: dict[tuple[str, str], dict] = {}
    for r in probes:
        for code, m in r["margins"].items():
            row = rows.setdefault(
                (r["task"], code), {"threshold": m["threshold"], "ref": [], "neg": []}
            )
            (row["ref"] if r["kind"] == "reference" else row["neg"]).extend([m["min"], m["max"]])
    for (task, code), row in sorted(rows.items()):
        ref = f"{_fmt(min(row['ref']))} .. {_fmt(max(row['ref']))}" if row["ref"] else "-"
        neg = f"{_fmt(min(row['neg']))} .. {_fmt(max(row['neg']))}" if row["neg"] else "-"
        out.append(f"| {task.split('-')[0]} | {code} | {_fmt(row['threshold'])} | {ref} | {neg} |")
    return out


def mutation_tables(muts: list[dict]) -> list[str]:
    out = ["| task | check switched off: killed | acceptance thresholds x0.5 / x2: killed | "
           "configuration constants: killed | surviving acceptance-threshold mutants |",
           "|---|---|---|---|---|"]
    for m in muts:
        s = m["summary"]

        def cell(key: str, s: dict = s) -> str:
            v = s.get(key)
            return f"{v['killed']}/{v['total']}" if v and v["total"] else "not run"

        surv = [x["description"] for x in m["mutants"]
                if not x["killed"] and x.get("role") == "acceptance"]
        out.append(f"| {m['task']} | {cell('check_off')} | {cell('threshold_acceptance')} | "
                   f"{cell('threshold_configuration')} | {'; '.join(surv) if surv else '-'} |")
    return out


def redteam_table(rt: dict) -> list[str]:
    out = ["| attack | sandboxed | unsandboxed |", "|---|---|---|"]
    rows: dict[str, dict[bool, dict]] = defaultdict(dict)
    for r in rt["results"]:
        rows[r["attack"]][r["sandboxed"]] = r
    for attack, by in rows.items():
        s, u = by.get(True), by.get(False)
        out.append(
            f"| `{attack}` | {'contained' if s and s['contained'] else 'NOT CONTAINED'}: {s['verdict'] if s else '-'} "
            f"| {'contained' if u and u['contained'] else 'not contained'}: {u['verdict'] if u else '-'} |"
        )
    return out


def main(out_path: str) -> int:
    probes = _load("t0*/*.json")
    muts = _load("mutation/*.json")
    rt_path = RESULTS / "redteam.json"
    rt = json.loads(rt_path.read_text()) if rt_path.exists() else None
    refs = [p for p in probes if p["kind"] == "reference"]
    negs = [p for p in probes if p["kind"] in {"negative", "boundary"}]
    total_ref_cases = sum(p["n_cases"] for p in refs)
    total_ref_fail = sum(p["n_cases"] - p["n_passed"] for p in refs)
    prov = probes[0]["provenance"] if probes else {}
    lines = [
        "# Grader validation report",
        "",
        "Generated by `validation/report.py` from the result files under `results/validation/`;",
        "do not edit by hand. Reproduce with `make validate`.",
        "",
        "## Summary",
        "",
        f"* **{len(refs)} reference solutions** graded on {total_ref_cases} hidden cases in total: "
        f"**{total_ref_fail} false rejects**.",
        f"* **{len(negs)} negative and boundary probes**: "
        f"{sum(p['assessment']['ok'] for p in negs)} behave as designed (expected codes fire and, for "
        f"negatives, the chance of earning reward over {HIDDEN} hidden cases is below 5 % at 95 % confidence).",
    ]
    if muts:
        co = sum(m["summary"].get("check_off", {}).get("killed", 0) for m in muts)
        cot = sum(m["summary"].get("check_off", {}).get("total", 0) for m in muts)
        lines.append(
            f"* **Grader mutation testing**: {co}/{cot} 'check switched off' mutants killed by the tests."
        )
    if rt:
        lines.append(
            f"* **Red team**: {'every attack contained' if rt['all_sandboxed_contained'] else 'ATTACKS NOT CONTAINED'} "
            "in the supported (sandboxed) configuration."
        )
    lines += [
        "",
        "## Findings",
        "",
        (ROOT / "validation/FINDINGS.md").read_text().strip(),
        "",
        "## Probes",
        "",
        "Reference: must pass every case. Negative: a realistic mistake, must fail with the expected codes; "
        f"'P(reward)' is the 95 % upper bound on passing all {HIDDEN} hidden cases. Boundary: a defensible "
        "method the contract excludes, must fail on exactly the expected codes.",
        "",
    ]
    lines += probe_tables(probes)
    lines += [
        "## Margins",
        "",
        "Measured values per check across all cases: references must sit on the passing side, "
        "negatives that fail it on the other. Wide gaps on both sides mean the threshold is not tuned "
        "to the probes. Where the threshold depends on the case (HRV tolerances, COVERAGE), the "
        "column shows one case's value; the per-case verdicts in the result files are authoritative.",
        "",
    ]
    lines += margin_tables(probes) + [""]
    if muts:
        lines += [
            "## Grader mutation testing",
            "",
            "Each mutant changes the grader, then the task's tests run; a killed mutant means the tests "
            "would catch that grader defect. 'Check switched off': the check runs but can never fail. "
            "Acceptance thresholds are constants a check reads; each is halved and doubled. "
            "Configuration constants set the scenario (durations, deadlines, pump size, grid levels); "
            "survivors there are expected and not a validity gap. A surviving gate (SCHEMA, SIM_FORMAT) "
            "means later checks also fail on malformed files, so the gate only improves diagnostics.",
            "",
        ]
        lines += mutation_tables(muts) + ["", "Files: `results/validation/mutation/*.json`", ""]
    if rt:
        lines += [
            "## Red team",
            "",
            "Attacks in `validation/redteam/`, graded as in the verifier (root grader, `sandbox` user, "
            "agent-image Python), and again without the sandbox user to show what the sandbox provides.",
            "",
        ]
        lines += redteam_table(rt) + ["", "File: `results/validation/redteam.json`", ""]
    lines += [
        "## Provenance",
        "",
        "* campaign master seed `0xB10E46A17E2C0D5B`; per-probe seeds derived from it and the probe name "
        "(listed in each result file)",
        f"* git commit `{prov.get('git', {}).get('commit')}` (dirty: {prov.get('git', {}).get('dirty')})",
        f"* `uv.lock` sha256 `{prov.get('uv_lock_sha256')}`",
        f"* Python {prov.get('python')}, packages {prov.get('packages')}",
        f"* Octave: {prov.get('octave')}",
        f"* platform: {prov.get('platform')}",
        "* every result file: `results/validation/MANIFEST.sha256`",
        "",
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n")
    files = sorted(p for p in RESULTS.rglob("*.json"))
    manifest = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n" for p in files
    )
    (RESULTS / "MANIFEST.sha256").write_text(manifest)
    print(f"wrote {out_path} ({len(files)} result files)")
    return 0
