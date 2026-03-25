---
name: debug_instrumentation
description: Use this skill when benchmark-level Modal results are not enough to explain a slow, unstable, or suspicious CUDA workload and OpenClaw needs to add temporary timing or debug instrumentation to kernel.cu or binding.py safely.
metadata:
  openclaw:
    os: ["linux"]
---

# Debug Instrumentation

Use this skill when workload-level benchmark output is not enough to decide the next kernel patch.

This is a guide skill. It does not provide a bundled runner yet. Instead, it tells OpenClaw when it is appropriate to instrument `kernel.cu` or `binding.py`, how to do that safely, and where to store the resulting debug artifacts.

## When To Use

Use this skill only after one of these signals appears:

- `run_modal_eval` shows that one or a few workloads dominate total runtime
- a workload is much slower than reference and simple benchmark output does not explain why
- a workload is unstable, times out, or shows suspicious correctness behavior

Do not start with instrumentation if the issue is still a basic compile or binding failure. Those should be handled by `local_compile_checks` first.

## First Preference: External Timing

Before patching the code for instrumentation, first ask:

1. Can the issue be isolated by rerunning one workload with `run_modal_eval --workload-id`?
2. Can host-side timing around the launch or wrapper isolate the problem?
3. Would the future `run_timer` skill be enough to answer the question?

If the answer is yes, prefer those paths. Only instrument inside the task code when external timing is still too coarse.

## Safe Instrumentation Rules

- Target one workload or a very small set of workloads
- Keep instrumentation temporary and easy to remove
- Prefer compile-time or runtime guards such as `SPINY_CLAW_DEBUG_TIMER`
- Do not intentionally change algorithmic behavior while measuring
- Do not leave verbose debug output in the best candidate
- Always record the exact workload IDs, input sizes, and instrumentation scope

## Practical Techniques

Prefer these techniques in order:

1. Host-side timing in `binding.py`
   - measure packing, launch, synchronization, and output materialization
   - this is the safest first step
2. Coarse device-side timing in `kernel.cu`
   - instrument major phases inside the main `__global__` kernel or obvious launch path
   - prefer timing a single designated thread or warp rather than every thread
3. Narrow instrumentation for suspicious regions
   - once a coarse phase is identified, instrument only the loop or helper region that looks dominant

For host-side timing, prefer CUDA events or a synchronized wall-clock measurement around a single launch.

For device-side timing, prefer a small debug buffer and a low-overhead timer such as `clock64()` from one elected thread. Avoid attempting to time every `__device__` helper function directly, because many helpers will be inlined or optimized away.

## What OpenClaw Should Edit

OpenClaw may temporarily edit:

- `task/<task-name>/solution/cuda/kernel.cu`
- `task/<task-name>/solution/cuda/binding.py`

Preferred pattern:

- add a small guarded debug path
- write minimal timing data into a fixed-size buffer
- return or log only the fields needed for diagnosis
- remove the instrumentation or gate it off once the bottleneck is understood

## Output Contract

Write timing and debug artifacts under:

```text
results/<task-name>/benchmark_result/timing_debug/
```

Recommended files:

- `summary.json`
- `instrumentation_plan.json`
- `timing.json`
- `run.log`

`summary.json` should include at least:

- `task_name`
- `run_id`
- `target_workloads`
- `instrumentation_scope`
- `main_bottleneck`
- `next_action`

`timing.json` should include repeated records with fields such as:

- `workload_id`
- `input_description`
- `location`
- `latency_ms`
- `notes`

## Decision Policy

- If host-side launch or synchronization dominates, inspect the wrapper or binding path before rewriting the kernel
- If one kernel phase dominates, patch that phase first
- If instrumentation perturbs correctness or creates instability, back it out and use a coarser method
- After instrumentation-driven changes, return to `local_compile_checks` and `run_modal_eval`

## Relationship To Other Skills

- Use after `local_compile_checks`
- Usually trigger it from `run_modal_eval` findings
- Treat it as a bridge between benchmark-level results and deeper profiling
- It complements future `run_timer`, `run_ncu`, and `run_nsys`

The normal loop is:

```text
run_modal_eval
-> debug_instrumentation
-> patch kernel or binding
-> local_compile_checks
-> run_modal_eval
```
