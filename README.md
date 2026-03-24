# Spiny-Claw

Autonomous CUDA Kernel Optimization Agent based on OpenClaw

Spiny-Claw is an OpenClaw-style runtime for CUDA kernel optimization. Instead of asking a model to write a fast kernel in one shot, the system runs a profiling-guided optimization loop around an existing kernel task.

Given a CUDA kernel plus its correctness and benchmark harness, Spiny-Claw is intended to:

1. Read the kernel and optimization target
2. Build or load the kernel
3. Run correctness checks
4. Run benchmarks
5. Run CUDA profiling tools such as Nsight Compute
6. Extract key performance signals
7. Ask the model to propose the next patch
8. Apply the patch and validate again
9. Keep the best candidate after multiple iterations

In other words, Spiny-Claw is a closed-loop optimizer, not just a code generator.

## OpenClaw x CUDA Kernel

You can think of Spiny-Claw as OpenClaw specialized for CUDA kernel work. OpenClaw acts as the orchestration layer, while the CUDA kernel is the primary optimization target.

A minimal workflow looks like this:

1. A user provides a task
2. The task contains a CUDA kernel, Python or TVM bindings, a correctness harness, and a benchmark harness
3. OpenClaw loads the task into a working directory
4. The agent calls build, test, benchmark, and profile skills
5. The agent reads structured logs and metrics
6. The agent updates `kernel.cu` or `binding.py`
7. The runtime records each iteration and keeps the current best result

## Repository Layout

This repository is organized around three top-level working areas:

```text
Spiny-Claw/
├── task/                 # Task definitions and kernel source code
├── benchmarks/          # Benchmark outputs, reports, and run artifacts
├── skills/              # OpenClaw skills used by the optimization runtime
├── README.md
└── TASK.md
```

### `task/`

`task/` stores optimization tasks. Each task should contain the kernel code and the files needed to build, validate, and package it.

Recommended structure:

```text
task/<task-name>/
├── config.toml
├── README.md
├── solution/
│   └── cuda/
│       ├── kernel.cu
│       └── binding.py
├── scripts/
│   ├── run_local.py
│   ├── run_modal.py
│   └── pack_solution.py
└── images/
```

The current example task is [`task/Fused-MoE-YWS`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS).

### `benchmarks/`

`benchmarks/` stores benchmark results and optimization artifacts produced by Spiny-Claw runs.

This folder is meant for outputs such as:

- benchmark summaries
- correctness reports
- profiling exports
- per-run logs
- leaderboards and best-result snapshots

Recommended structure:

```text
benchmarks/
└── <task-name>/
    ├── latest/
    ├── history/
    ├── profiles/
    └── leaderboard.json
```

### `skills/`

`skills/` stores the OpenClaw skills that the runtime can call while optimizing kernels.

Typical skills include:

- `read_kernel`
- `apply_patch`
- `build_kernel`
- `run_correctness`
- `run_benchmark`
- `run_profile`
- `parse_metrics`
- `save_report`

These skills are the bridge between the agent and the concrete CUDA workflow.

## Current Mapping In This Repo

This repository already contains the first pieces of that contract:

- [`task/Fused-MoE-YWS/README.md`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/README.md) describes the existing kernel task
- [`task/Fused-MoE-YWS/config.toml`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/config.toml) defines build metadata
- [`task/Fused-MoE-YWS/solution/cuda/kernel.cu`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/solution/cuda/kernel.cu) is the CUDA kernel under optimization
- [`task/Fused-MoE-YWS/solution/cuda/binding.py`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/solution/cuda/binding.py) is the Python or TVM entry point
- [`task/Fused-MoE-YWS/scripts/run_local.py`](/home/yuwei/Documents/Spiny-Claw/task/Fused-MoE-YWS/scripts/run_local.py) shows the current local benchmark path

## Agent Loop

The intended optimization loop is:

```text
read task
-> inspect kernel and previous memory
-> build kernel
-> run correctness
-> run benchmark
-> run profile
-> summarize bottlenecks
-> propose patch
-> apply patch
-> rerun validation
-> update leaderboard
```

This contract keeps the system reusable across different CUDA tasks while preserving a consistent optimization workflow.
