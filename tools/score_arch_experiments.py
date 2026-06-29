#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize architecture experiments and rank by pseudo-private + public-val weighted score"
    )
    parser.add_argument("--project", type=Path, default=Path("runs/GAIIC2024_archscan"))
    parser.add_argument(
        "--summary-jsonl",
        type=Path,
        default=None,
        help="Optional summary jsonl from batch script. Empty means <project>/archscan_summary.jsonl",
    )
    parser.add_argument("--score-pseudo-weight", type=float, default=0.7)
    parser.add_argument("--score-public-weight", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def _norm_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {k.strip(): v for k, v in row.items() if k is not None}


def _first_present(row: dict[str, str], keys: list[str]) -> str:
    for k in keys:
        if k in row:
            return row[k]
    return ""


def parse_results_csv(results_csv: Path) -> tuple[float, float, int]:
    if not results_csv.exists():
        return float("nan"), float("nan"), 0

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = [_normalize_row(r) for r in csv.DictReader(f)]

    if not rows:
        return float("nan"), float("nan"), 0

    key_options = ["metrics/mAP50-95(B)", "metrics/mAP50-95"]
    best_idx = 0
    best_val = -1.0
    for i, r in enumerate(rows):
        v = _norm_float(_first_present(r, key_options))
        if not math.isnan(v) and v > best_val:
            best_val = v
            best_idx = i

    last_val = _norm_float(_first_present(rows[-1], key_options))
    return _norm_float(best_val), last_val, len(rows)


def score(pseudo_map: float, public_map: float, w_pseudo: float, w_public: float) -> float:
    pseudo_map = _norm_float(pseudo_map)
    public_map = _norm_float(public_map)
    if math.isnan(pseudo_map) or math.isnan(public_map):
        return float("nan")

    total = max(1e-12, w_pseudo + w_public)
    wp = w_pseudo / total
    wv = w_public / total
    return wp * pseudo_map + wv * public_map


def load_from_summary_jsonl(summary_jsonl: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not summary_jsonl.exists():
        return records

    for line in summary_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue

        run_name = str(rec.get("run_name", "")).strip()
        if not run_name:
            continue

        old = records.get(run_name)
        if old is None or str(rec.get("timestamp", "")) >= str(old.get("timestamp", "")):
            records[run_name] = rec

    return records


def fallback_scan_project(project_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not project_dir.exists():
        return records

    for d in sorted([x for x in project_dir.iterdir() if x.is_dir()]):
        if d.name.startswith("_archscan_meta"):
            continue

        results_csv = d / "results.csv"
        if not results_csv.exists():
            continue

        pseudo_best, pseudo_last, epochs_recorded = parse_results_csv(results_csv)

        public_json = d / "archscan_public_val.json"
        public_map = float("nan")
        if public_json.exists():
            try:
                obj = json.loads(public_json.read_text(encoding="utf-8"))
                public_map = _norm_float(obj.get("public_val_map50_95", float("nan")))
            except Exception:
                pass

        arch_source = ""
        args_yaml = d / "args.yaml"
        if args_yaml.exists():
            try:
                args_obj = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
                arch_source = str(args_obj.get("model", ""))
            except Exception:
                pass

        records[d.name] = {
            "run_name": d.name,
            "run_dir": str(d.resolve()),
            "arch_source": arch_source,
            "arch_n_alias": arch_source,
            "pseudo_best_map50_95": pseudo_best,
            "pseudo_last_map50_95": pseudo_last,
            "epochs_recorded": epochs_recorded,
            "public_val_map50_95": public_map,
            "status": "ok",
        }

    return records


def main() -> None:
    args = parse_args()
    project_dir = args.project.resolve()
    summary_jsonl = args.summary_jsonl.resolve() if args.summary_jsonl else (project_dir / "archscan_summary.jsonl")

    records = load_from_summary_jsonl(summary_jsonl)
    if not records:
        records = fallback_scan_project(project_dir)

    out = []
    for run_name, rec in records.items():
        if rec.get("status") not in {"ok", "completed", None}:
            continue

        pseudo_map = _norm_float(rec.get("pseudo_best_map50_95", float("nan")))
        public_map = _norm_float(rec.get("public_val_map50_95", float("nan")))
        s = score(pseudo_map, public_map, args.score_pseudo_weight, args.score_public_weight)
        if math.isnan(s):
            continue

        out.append(
            {
                "run_name": run_name,
                "arch_source": rec.get("arch_source", ""),
                "arch_n_alias": rec.get("arch_n_alias", ""),
                "pseudo_best_map50_95": pseudo_map,
                "public_val_map50_95": public_map,
                "score": s,
                "run_dir": rec.get("run_dir", ""),
            }
        )

    out.sort(key=lambda x: x["score"], reverse=True)

    output_csv = args.output_csv.resolve() if args.output_csv else (project_dir / "archscan_leaderboard.csv")
    output_md = args.output_md.resolve() if args.output_md else (project_dir / "archscan_recommendation.md")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "rank",
        "run_name",
        "arch_source",
        "arch_n_alias",
        "pseudo_best_map50_95",
        "public_val_map50_95",
        "score",
        "run_dir",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, r in enumerate(out, start=1):
            writer.writerow(
                [
                    i,
                    r["run_name"],
                    r["arch_source"],
                    r["arch_n_alias"],
                    f"{r['pseudo_best_map50_95']:.6f}",
                    f"{r['public_val_map50_95']:.6f}",
                    f"{r['score']:.6f}",
                    r["run_dir"],
                ]
            )

    md_lines = [
        "# Architecture Recommendation",
        "",
        f"Weights: pseudo_private={args.score_pseudo_weight}, public_val={args.score_public_weight}",
        "",
        "| Rank | Run | Source Arch | n Alias | Pseudo mAP50-95 | Public mAP50-95 | Score |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for i, r in enumerate(out[: max(1, args.topk)], start=1):
        md_lines.append(
            "| {rank} | {run} | {src} | {alias} | {pm:.6f} | {vm:.6f} | {sc:.6f} |".format(
                rank=i,
                run=r["run_name"],
                src=Path(str(r["arch_source"])).name,
                alias=Path(str(r["arch_n_alias"])).name,
                pm=r["pseudo_best_map50_95"],
                vm=r["public_val_map50_95"],
                sc=r["score"],
            )
        )
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("[SUMMARY] source_summary:", summary_jsonl)
    print("[SUMMARY] leaderboard_csv:", output_csv)
    print("[SUMMARY] recommendation_md:", output_md)

    if out:
        print(f"[SUMMARY] Top-{min(args.topk, len(out))} recommendations:")
        for i, r in enumerate(out[: args.topk], start=1):
            print(
                f"  {i}. {r['run_name']} | score={r['score']:.6f} | "
                f"pseudo={r['pseudo_best_map50_95']:.6f} | public={r['public_val_map50_95']:.6f}"
            )
    else:
        print("[SUMMARY] No valid scored runs found.")


if __name__ == "__main__":
    main()
