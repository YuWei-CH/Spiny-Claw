# OpenClaw Workflow

This document defines how OpenClaw should execute a CUDA kernel optimization task in Spiny-Claw.

OpenClaw should read this file before starting a task iteration.

For the current task, the concrete objective is to optimize the performance of [`task/Fused-MoE-YWS/solution/cuda/kernel.cu`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/solution/cuda/kernel.cu) for Fused Mixture-of-Experts kernels with FP8 support on B200 hardware, while preserving correctness.

The goal is not to make OpenClaw write the fastest kernel in one shot. The goal is to run a stable optimization loop around a task:

1. Read the task
2. Run a local precheck
3. Run a remote benchmark
4. Read the results
5. Identify bottlenecks and risks
6. Patch the code
7. Validate again

## Phase 0: Read Task

OpenClaw should first read:

- [`TASK.md`](/home/yuwei/Documents/Spiny-Claw/TASK.md)
- [`task/<task-name>/README.md`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/README.md)
- [`task/<task-name>/FAQ.md`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/FAQ.md)
- [`task/<task-name>/config.toml`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/config.toml)
- [`task/<task-name>/solution/cuda/kernel.cu`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/solution/cuda/kernel.cu)
- [`task/<task-name>/solution/cuda/binding.py`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/solution/cuda/binding.py)

The purpose of this phase is to determine:

- which task is being optimized
- what the task background and evaluation constraints are
- what the target hardware is
- what the current build and binding contract is
- whether this iteration should run a full benchmark or only a few workloads

For the current task, OpenClaw should remember:

- the target workload family is fused MoE with FP8 support
- the target hardware is B200
- Modal runs are for development feedback, while the official evaluation target is still B200

## Phase 1: One-Time Setup

These actions are usually run once per machine or environment, not once per iteration.

Current setup scripts:

- [`scripts/init_env.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_env.sh)
- [`scripts/init_cuda_toolchain.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_cuda_toolchain.sh)
- [`scripts/init_dataset.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_dataset.sh)
- [`scripts/init_modal_volume.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_modal_volume.sh)

These should later be abstracted into skills:

- `init_modal`
- `sync_benchmark_data`

At this stage, OpenClaw only needs to confirm that the environment is ready. It should not repeat full setup every iteration.

## Phase 2: Local Precheck

This phase is handled by [`skills/local-compile-checks/SKILL.md`](/home/yuwei/Documents/Spiny-Claw/skills/local-compile-checks/SKILL.md).

OpenClaw should:

1. Call `local_compile_checks`
2. Read [`results/<task-name>/compile_result/summary.json`](/home/yuwei/Documents/Spiny-Claw/results/Fused-MoE-YWS/compile_result/summary.json)
3. If there is a compile or binding contract issue, fix that before any remote benchmark run

Main outputs from this phase:

- `binding_contract.json`
- `kernel_compile.json`
- `kernel_compile.log`

Decision rules:

- If `compile_result` is `fail`, fix the build or binding issue first
- If `compile_result` is only `warn`, continue to remote benchmark but keep that warning in later analysis

## Phase 3: Remote Benchmark

This phase is handled by [`skills/run-modal-eval/SKILL.md`](/home/yuwei/Documents/Spiny-Claw/skills/run-modal-eval/SKILL.md).

OpenClaw should:

1. Decide the benchmark scope for this iteration
2. Call `run_modal_eval`
3. Read `benchmark_result`

Current supported modes:

- full run
- `--max-workloads <n>` for a small debug run
- `--workload-id <id>` for one or a few specific workloads

Recommended strategy:

- Start with a small number of workloads during initial debugging
- If some workloads are especially slow or likely to timeout, focus on those first
- Return to the full benchmark once local bottleneck work is stable

Main outputs from this phase:

- [`results/<task-name>/benchmark_result/summary.json`](/home/yuwei/Documents/Spiny-Claw/results/Fused-MoE-YWS/benchmark_result/summary.json)
- [`results/<task-name>/benchmark_result/benchmark.json`](/home/yuwei/Documents/Spiny-Claw/results/Fused-MoE-YWS/benchmark_result/benchmark.json)
- [`results/<task-name>/benchmark_result/correctness.json`](/home/yuwei/Documents/Spiny-Claw/results/Fused-MoE-YWS/benchmark_result/correctness.json)
- [`results/<task-name>/benchmark_result/run.log`](/home/yuwei/Documents/Spiny-Claw/results/Fused-MoE-YWS/benchmark_result/run.log)

## Phase 4: Interpret Results

After reading `benchmark_result`, OpenClaw should answer:

1. Is this version faster than the reference?
2. Which workloads are the slowest?
3. Which workloads have the largest errors?
4. Does this iteration look more like a performance regression or a correctness risk?

`run_modal_eval` already provides:

- `performance_assessment`
- `accuracy_assessment`
- `overall_assessment`

OpenClaw should use them like this:

- `regressed`
  The current kernel is slower than the reference, so slow workloads should be inspected first
- `medium_risk` / `high_risk` / `severe_risk`
  Correctness should be investigated before more aggressive optimization
- `poor_candidate`
  The current version should not be kept as the best candidate

## Phase 4.5: Timing And Debug

If `run_modal_eval` can only say "it is slower" but cannot explain where the time is going, OpenClaw should enter a timing and debug phase.

This phase is guided by [`skills/debug-instrumentation/SKILL.md`](/home/yuwei/Documents/Spiny-Claw/skills/debug-instrumentation/SKILL.md).

OpenClaw should:

1. Narrow the scope to one or a few workloads
2. Try external timing before modifying the kernel
3. If external timing is still too coarse, temporarily instrument `kernel.cu` or `binding.py`
4. Write timing and debug artifacts to `results/<task-name>/benchmark_result/timing_debug/`
5. Use those results to decide which code region should be patched next

Key principles for this phase:

- prefer coarse-grained measurements before fine-grained instrumentation
- prefer host-side timing before device-side instrumentation
- instrumentation should be temporary and easy to disable
- do not assume every `__device__` helper function can or should be timed directly
- return to the normal validation loop after the instrumentation run

## Phase 5: Patch Kernel

After results have been interpreted, OpenClaw enters the patching phase.

OpenClaw should prioritize:

- [`kernel.cu`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/solution/cuda/kernel.cu)
- [`binding.py`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/solution/cuda/binding.py)

Each patch should have a clear purpose:

- fix a compile failure
- improve a particularly slow workload
- reduce correctness error

## Phase 6: Re-Validate

After each patch, OpenClaw should return to:

1. `local_compile_checks`
2. `run_modal_eval`

The correct loop is:

```text
read task
-> local_compile_checks
-> run_modal_eval
-> interpret results
-> optional debug_instrumentation
-> patch kernel
-> local_compile_checks
-> run_modal_eval
-> compare against previous result
```

## Current Skill Set

Skills currently available in this workspace:

- [`skills/local-compile-checks/SKILL.md`](/home/yuwei/Documents/Spiny-Claw/skills/local-compile-checks/SKILL.md)
- [`skills/run-modal-eval/SKILL.md`](/home/yuwei/Documents/Spiny-Claw/skills/run-modal-eval/SKILL.md)
- [`skills/debug-instrumentation/SKILL.md`](/home/yuwei/Documents/Spiny-Claw/skills/debug-instrumentation/SKILL.md)

Recommended next skills:

- `run_timer`
- `run_ncu`
- `run_nsys`

This means the workspace already supports a minimum optimization loop, but not yet a full profiling stack.

## Minimal Decision Policy

If OpenClaw starts running today, the minimum decision policy can be:

1. Run `local_compile_checks` first
2. If it fails, fix compile or binding issues first
3. If it passes, run one workload or a small workload subset first
4. If the result is `regressed`, focus on the slow workloads first
5. If benchmark output is too coarse, use `debug_instrumentation` for temporary timing or debug instrumentation
6. If the result shows correctness risk, fix correctness first
7. Once local workload improvements are stable, return to the full benchmark
8. Update the best candidate only when the new version is better in performance or correctness
