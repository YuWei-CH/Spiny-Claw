#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUCCESS_STATUSES = {"SUCCESS", "PASSED", "PASS"}
WORKLOAD_RE = re.compile(
    r"^\s*Workload\s+(?P<workload>[0-9a-fA-F]+)\.\.\.:\s+(?P<status>[A-Z_]+)"
    r"(?:\s+\|\s+(?P<latency>[0-9.]+)\s+ms)?"
    r"(?:\s+\|\s+(?P<speedup>[0-9.]+)x speedup)?"
    r"(?:\s+\|\s+abs_err=(?P<abs_err>[0-9.eE+-]+),\s+rel_err=(?P<rel_err>[0-9.eE+-]+))?\s*$"
)
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a task's Modal benchmark script and evaluate the result."
    )
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-workloads", type=int, default=0)
    parser.add_argument(
        "--workload-id",
        action="append",
        default=[],
        help="Run a specific workload by full UUID or unique prefix. Repeatable.",
    )
    parser.add_argument("--modal-bin", default="modal")
    parser.add_argument(
        "--run-log",
        type=Path,
        default=None,
        help="Reuse an existing Modal run log instead of launching a new remote run.",
    )
    return parser.parse_args()


def run_modal_command(
    task_dir: Path,
    modal_bin: str,
    max_workloads: int,
    workload_ids: list[str],
) -> tuple[int, str]:
    modal_path = shutil.which(modal_bin)
    if not modal_path:
        raise FileNotFoundError(
            f"Modal CLI '{modal_bin}' was not found on PATH. Activate the environment where Modal is installed."
        )

    script_path = task_dir / "scripts" / "run_modal.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing Modal runner: {script_path}")

    command = [modal_path, "run", str(script_path)]
    if workload_ids:
        command.extend(["--workload-ids", ",".join(workload_ids)])
    if max_workloads > 0:
        command.extend(["--max-workloads", str(max_workloads)])

    output_lines: list[str] = []
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=REPO_ROOT,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            clean_line = strip_ansi(line)
            output_lines.append(clean_line)
            print(clean_line, end="", flush=True)
        returncode = proc.wait()
    return returncode, "".join(output_lines)


def parse_modal_output(raw_output: str) -> dict:
    lines = raw_output.splitlines()
    definition_name: str | None = None
    workloads: list[dict] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.endswith(":") and "Workload " not in stripped:
            candidate_definition = stripped[:-1]
            if re.fullmatch(r"[A-Za-z0-9_.-]+", candidate_definition):
                definition_name = candidate_definition
                continue

        match = WORKLOAD_RE.match(line)
        if not match:
            continue

        workload = {
            "workload_id_display": match.group("workload"),
            "status": match.group("status"),
        }
        if definition_name:
            workload["definition"] = definition_name
        if match.group("latency"):
            workload["latency_ms"] = float(match.group("latency"))
        if match.group("speedup"):
            workload["speedup_factor"] = float(match.group("speedup"))
        if match.group("abs_err"):
            workload["max_abs_error"] = float(match.group("abs_err"))
        if match.group("rel_err"):
            workload["max_rel_error"] = float(match.group("rel_err"))
        workloads.append(workload)

    return {"definition": definition_name, "workloads": workloads}


def safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def classify_performance(pass_rate: float, speedups: list[float]) -> str:
    if not speedups:
        return "unknown"
    median_speedup = safe_median(speedups) or 0.0
    faster_count = sum(speedup > 1.0 for speedup in speedups)
    faster_ratio = faster_count / len(speedups)

    if pass_rate < 1.0:
        return "unstable"
    if faster_count == 0 and median_speedup < 1.0:
        return "regressed"
    if median_speedup >= 2.0 and faster_ratio >= 0.75:
        return "strong"
    if median_speedup >= 1.0 and faster_ratio >= 0.5:
        return "promising"
    return "weak"


def classify_accuracy(abs_errors: list[float], rel_errors: list[float]) -> str:
    if not abs_errors and not rel_errors:
        return "unknown"

    max_abs = max(abs_errors) if abs_errors else 0.0
    max_rel = max(rel_errors) if rel_errors else 0.0

    if max_rel > 1e3 or max_abs > 1e5:
        return "severe_risk"
    if max_rel > 0.5 or max_abs > 1e4:
        return "high_risk"
    if max_rel > 0.05 or max_abs > 1e3:
        return "medium_risk"
    return "low_risk"


def classify_overall(
    performance_assessment: str,
    accuracy_assessment: str,
    failed_count: int,
) -> str:
    if failed_count > 0:
        return "failing_candidate"
    if performance_assessment == "regressed":
        return "poor_candidate"
    if performance_assessment in {"strong", "promising"} and accuracy_assessment == "low_risk":
        return "strong_candidate"
    if performance_assessment in {"strong", "promising"} and accuracy_assessment == "medium_risk":
        return "promising_but_check_accuracy"
    if performance_assessment == "weak" and accuracy_assessment in {"high_risk", "severe_risk"}:
        return "poor_candidate"
    return "mixed_candidate"


def top_items(
    workloads: list[dict],
    key: str,
    reverse: bool,
    limit: int = 3,
) -> list[dict]:
    items = [w for w in workloads if key in w]
    items.sort(key=lambda item: item[key], reverse=reverse)
    return items[:limit]


def build_assessment(parsed: dict, command_exit_code: int) -> tuple[dict, dict, dict, dict]:
    workloads = parsed["workloads"]
    total_count = len(workloads)
    passed = [w for w in workloads if w["status"] in SUCCESS_STATUSES]
    failed = [w for w in workloads if w["status"] not in SUCCESS_STATUSES]
    pass_rate = (len(passed) / total_count) if total_count else 0.0

    latencies = [w["latency_ms"] for w in workloads if "latency_ms" in w]
    speedups = [w["speedup_factor"] for w in workloads if "speedup_factor" in w]
    abs_errors = [w["max_abs_error"] for w in workloads if "max_abs_error" in w]
    rel_errors = [w["max_rel_error"] for w in workloads if "max_rel_error" in w]

    performance_assessment = classify_performance(pass_rate, speedups)
    accuracy_assessment = classify_accuracy(abs_errors, rel_errors)
    overall_assessment = classify_overall(
        performance_assessment=performance_assessment,
        accuracy_assessment=accuracy_assessment,
        failed_count=len(failed),
    )

    benchmark_payload = {
        "definition": parsed.get("definition"),
        "workload_count": total_count,
        "passed_count": len(passed),
        "failed_count": len(failed),
        "metrics": {
            "average_latency_ms": safe_mean(latencies),
            "median_latency_ms": safe_median(latencies),
            "average_speedup_factor": safe_mean(speedups),
            "median_speedup_factor": safe_median(speedups),
            "min_speedup_factor": min(speedups) if speedups else None,
            "max_speedup_factor": max(speedups) if speedups else None,
            "faster_than_reference_count": sum(speedup > 1.0 for speedup in speedups),
            "slower_or_equal_reference_count": sum(speedup <= 1.0 for speedup in speedups),
        },
        "performance_assessment": performance_assessment,
        "best_speedup_workloads": top_items(workloads, "speedup_factor", reverse=True),
        "worst_speedup_workloads": top_items(workloads, "speedup_factor", reverse=False),
        "workloads": workloads,
    }

    correctness_payload = {
        "definition": parsed.get("definition"),
        "workload_count": total_count,
        "metrics": {
            "max_abs_error": max(abs_errors) if abs_errors else None,
            "max_rel_error": max(rel_errors) if rel_errors else None,
            "median_rel_error": safe_median(rel_errors),
        },
        "accuracy_assessment": accuracy_assessment,
        "highest_rel_error_workloads": top_items(workloads, "max_rel_error", reverse=True),
        "highest_abs_error_workloads": top_items(workloads, "max_abs_error", reverse=True),
        "workloads": [
            {
                "workload_id_display": w["workload_id_display"],
                "status": w["status"],
                "max_abs_error": w.get("max_abs_error"),
                "max_rel_error": w.get("max_rel_error"),
            }
            for w in workloads
        ],
    }

    profile_payload = {
        "status": "not_collected",
        "reason": "run_modal_eval currently evaluates benchmark and correctness output only.",
    }

    summary_payload = {
        "task_name": parsed.get("definition"),
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "success" if command_exit_code == 0 and total_count > 0 else "fail",
        "benchmark_passed": len(failed) == 0 and total_count > 0,
        "correctness_passed": len(failed) == 0 and total_count > 0,
        "profile_collected": False,
        "workload_count": total_count,
        "passed_count": len(passed),
        "failed_count": len(failed),
        "pass_rate": pass_rate,
        "performance_assessment": performance_assessment,
        "accuracy_assessment": accuracy_assessment,
        "overall_assessment": overall_assessment,
    }

    return summary_payload, benchmark_payload, correctness_payload, profile_payload


def main() -> int:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    task_name = task_dir.name
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else REPO_ROOT / "results" / task_name / "benchmark_result"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_workloads > 0 and args.workload_id:
        raise ValueError("Use either --max-workloads or --workload-id, not both")

    if args.run_log:
        run_log_path = args.run_log.resolve()
        raw_output = run_log_path.read_text(encoding="utf-8")
        exit_code = 0
    else:
        print("Launching Modal benchmark run...")
        print(
            "Note: full benchmark runs can take a while, and containerized output may appear in a batch near the end."
        )
        if args.max_workloads > 0:
            print(
                f"Debug mode enabled: limiting this run to {args.max_workloads} workload(s)."
            )
        elif args.workload_id:
            print(
                "Running selected workload(s): "
                + ", ".join(workload_id.strip() for workload_id in args.workload_id)
            )
        else:
            print(
                "Tip: use --max-workloads for shorter debug runs, or --workload-id to focus on a specific bottleneck workload."
            )
        try:
            exit_code, raw_output = run_modal_command(
                task_dir=task_dir,
                modal_bin=args.modal_bin,
                max_workloads=args.max_workloads,
                workload_ids=args.workload_id,
            )
        except Exception as exc:
            failure_log = output_dir / "run.log"
            failure_log.write_text(f"{exc}\n", encoding="utf-8")
            summary = {
                "task_name": task_name,
                "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "status": "fail",
                "benchmark_passed": False,
                "correctness_passed": False,
                "profile_collected": False,
                "overall_assessment": "setup_failure",
                "error": str(exc),
            }
            write_json(output_dir / "summary.json", summary)
            write_json(
                output_dir / "profile.json",
                {"status": "not_collected", "reason": "benchmark did not start"},
            )
            print(f"Failed to start Modal benchmark: {exc}")
            return 1

        run_log_path = output_dir / "run.log"
        run_log_path.write_text(raw_output, encoding="utf-8")

    parsed = parse_modal_output(raw_output)
    if not parsed["workloads"]:
        summary = {
            "task_name": task_name,
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "status": "fail",
            "benchmark_passed": False,
            "correctness_passed": False,
            "profile_collected": False,
            "overall_assessment": "parse_failure",
            "command_exit_code": exit_code,
            "error": "Modal command completed but no workload lines could be parsed from run.log",
        }
        write_json(output_dir / "summary.json", summary)
        write_json(
            output_dir / "profile.json",
            {"status": "not_collected", "reason": "benchmark output parsing failed"},
        )
        print("Modal command completed but no workload lines were parsed.")
        print(f"Raw log: {run_log_path}")
        return 1

    summary_payload, benchmark_payload, correctness_payload, profile_payload = build_assessment(
        parsed=parsed,
        command_exit_code=exit_code,
    )
    accuracy_passed = correctness_payload["accuracy_assessment"] in {"low_risk", "medium_risk"}
    summary_payload["task_name"] = task_name
    summary_payload["definition"] = parsed.get("definition")
    summary_payload["command_exit_code"] = exit_code
    summary_payload["correctness_passed"] = accuracy_passed
    summary_payload["faster_than_reference_count"] = benchmark_payload["metrics"][
        "faster_than_reference_count"
    ]
    summary_payload["slower_or_equal_reference_count"] = benchmark_payload["metrics"][
        "slower_or_equal_reference_count"
    ]
    summary_payload["artifacts"] = {
        "benchmark": str(output_dir / "benchmark.json"),
        "correctness": str(output_dir / "correctness.json"),
        "profile": str(output_dir / "profile.json"),
        "run_log": str(run_log_path),
    }

    write_json(output_dir / "summary.json", summary_payload)
    write_json(output_dir / "benchmark.json", benchmark_payload)
    write_json(output_dir / "correctness.json", correctness_payload)
    write_json(output_dir / "profile.json", profile_payload)

    print(f"Task: {task_name}")
    print(f"Definition: {parsed.get('definition')}")
    print(f"Output: {output_dir}")
    print(f"Workloads parsed: {len(parsed['workloads'])}")
    print(f"Performance assessment: {summary_payload['performance_assessment']}")
    print(f"Accuracy assessment: {summary_payload['accuracy_assessment']}")
    print(f"Overall assessment: {summary_payload['overall_assessment']}")
    print(f"Raw log: {run_log_path}")

    return 0 if exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
