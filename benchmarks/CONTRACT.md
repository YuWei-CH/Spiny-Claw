# Benchmark Output Contract

This document defines the minimum output layout for Spiny-Claw benchmark runs.

## Goal

Every optimization run should produce artifacts in a predictable location so the agent can compare iterations and keep the best result.

## Layout

```text
benchmarks/
└── <task-name>/
    ├── latest/
    │   ├── summary.json
    │   ├── correctness.json
    │   ├── benchmark.json
    │   ├── profile.json
    │   └── run.log
    ├── history/
    │   └── <run-id>/
    │       ├── summary.json
    │       ├── correctness.json
    │       ├── benchmark.json
    │       ├── profile.json
    │       └── run.log
    ├── profiles/
    │   └── <run-id>.ncu-rep
    └── leaderboard.json
```

## Required Files

- `summary.json`: high-level status for one run
- `correctness.json`: correctness metrics and pass or fail result
- `benchmark.json`: latency, throughput, and other benchmark metrics
- `profile.json`: parsed profiling metrics
- `run.log`: human-readable execution log
- `leaderboard.json`: best known candidates for the task

## Summary Schema

`summary.json` should contain at least:

```json
{
  "task_name": "Fused-MoE-YWS",
  "run_id": "20260323T220000Z",
  "status": "success",
  "candidate": "iteration-003",
  "correctness_passed": true,
  "benchmark_passed": true,
  "profile_collected": true
}
```
