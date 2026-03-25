---
name: run_modal_eval
description: Use this skill when you need to execute a CUDA task benchmark on Modal and evaluate the kernel from the Modal output, including per-workload latency, speedup versus reference, and correctness error signals.
metadata:
  openclaw:
    os: ["linux"]
    requires:
      bins: ["python3", "modal"]
---

# Run Modal Eval

Use this skill when the user wants to benchmark a task on Modal and needs a structured judgment of the kernel's effectiveness.

This skill wraps a task's existing `scripts/run_modal.py` runner, captures the Modal output, and writes structured artifacts under:

```text
results/<task-name>/benchmark_result/
```

## How To Run

```bash
python3 skills/run-modal-eval/scripts/run_modal_evaluate.py --task-dir task/Fused-MoE-YWS
```

Optional flags:

- `--max-workloads <n>` to limit the benchmark run for debugging
- `--workload-id <id>` to run a specific workload by full UUID or unique prefix; repeatable
- `--output-dir <dir>` to override the default result location
- `--modal-bin <path>` if the Modal CLI is not on the default `PATH`
- `--run-log <path>` to reuse an existing Modal log without launching a new remote run

## Runtime Notes

- Full Modal benchmark runs can take a long time, especially when the remote image must build or GPU capacity is busy.
- The current task runner executes inside a containerized Modal flow, so output may arrive in one large batch near the end instead of streaming smoothly line by line.
- `--max-workloads` is useful for cheap debug runs.
- This skill now supports selecting specific workload IDs, because a small number of workloads often dominate runtime or hit timeouts.

## What This Skill Does

1. Runs the task's `scripts/run_modal.py` through the Modal CLI
2. Captures the raw output into `run.log`
3. Parses per-workload benchmark lines such as:
   `Workload abc12345...: PASSED | 1.447 ms | 8.07x speedup | abs_err=1.02e+03, rel_err=3.48e-01`
4. Produces structured `benchmark.json` and `correctness.json`
5. Produces `summary.json` with:
   - workload counts
   - pass rate
   - average and median speedup
   - best and worst workloads
   - performance assessment
   - accuracy assessment
   - overall assessment

When `--run-log` is provided, the script skips the remote execution step and only re-parses an existing log. Use this when you are iterating on the evaluation logic and do not want to spend another Modal run.

When `--workload-id` is provided, the skill forwards those selectors to the task runner. Selectors may be full workload UUIDs or unique prefixes such as `b8f4f012`.

## Assessment Model

The script uses lightweight heuristics to classify the run:

- performance is judged from pass rate, median speedup, and how many workloads beat the reference
- clearly slower-than-reference runs are marked as `regressed`
- accuracy risk is judged mostly from relative error, with absolute error as a secondary warning signal
- overall assessment combines both signals into labels such as `strong_candidate`, `mixed_candidate`, or `poor_candidate`

These heuristics are meant to guide OpenClaw's next action, not replace domain-specific review.

## Failure Handling

- If `modal` is missing, tell the user to activate the environment where Modal is installed
- If the Modal command fails, include the exit code and `run.log` path
- If the Modal command succeeds but no workload lines can be parsed, treat that as a result-format failure and surface the raw log
