# Skills Contract

This document defines the current and future OpenClaw skills for Spiny-Claw.

## Goal

Each skill should expose one narrow action that the agent can call while optimizing a CUDA kernel task.

## Current Implementation Targets

### `init_modal`

Purpose: initialize the Modal environment required by Spiny-Claw.

Expected inputs:

- task path
- Modal configuration
- dataset location or volume settings

Expected outputs:

- authentication status
- environment validation status
- created or reused Modal resources

Notes:

- this is a one-time setup skill
- it should prepare permissions and basic runtime configuration
- local `modal setup` authentication is required before `modal run` and `modal volume` commands can work
- local GPU access is not required for Modal execution

### `sync_benchmark_data`

Purpose: move benchmark data to Modal or confirm that remote benchmark data is already available.

Expected inputs:

- task path
- local dataset path
- remote dataset target

Expected outputs:

- sync status
- remote dataset location
- transfer logs or validation logs

Notes:

- this is a one-time setup skill
- it should not be part of every optimization iteration

### `local_compile`

Purpose: run local static compile checks with `nvcc` before remote execution.

Expected inputs:

- task path
- build configuration
- target source files

Expected outputs:

- compile status
- compiler logs
- discovered build errors

Notes:

- this is a local precheck skill
- it is not the main benchmark path
- it is intended to catch obvious CUDA build failures before sending work to Modal
- the CUDA toolkit should be prepared ahead of time by a one-time local setup script
- installing `nvcc` and dev libraries is not the responsibility of this skill
- current workspace implementation lives at [`skills/local-compile-checks/SKILL.md`](/home/yuwei/Documents/Spiny-Claw/skills/local-compile-checks/SKILL.md)

### `run_modal`

Purpose: execute the task in the Modal cloud GPU environment.

Expected inputs:

- task path
- run configuration
- Modal execution settings

Expected outputs:

- remote run status
- artifact locations
- execution logs

Notes:

- this is the main remote execution entry point
- benchmark should run on Modal rather than locally

### `run_timer`

Purpose: collect structured timing data for the current candidate in the remote environment.

Expected inputs:

- task path
- timing configuration
- candidate identifier

Expected outputs:

- latency metrics
- timing summary
- raw timing logs

Notes:

- this is currently in scope
- it should teach OpenClaw how to time code in a repeatable way

## Future Skills

### `run_ncu`

Purpose: collect kernel-level profiling data with Nsight Compute.

Expected inputs:

- task path
- profiling configuration
- candidate identifier

Expected outputs:

- parsed NCU metrics
- raw profiler artifact paths

Notes:

- future implementation target

### `run_nsys`

Purpose: collect system-level execution traces with Nsight Systems.

Expected inputs:

- task path
- profiling configuration
- candidate identifier

Expected outputs:

- parsed NSYS summary
- raw profiler artifact paths

Notes:

- future implementation target
