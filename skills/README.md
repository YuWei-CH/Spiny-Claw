# Skills

This directory stores OpenClaw skills used by Spiny-Claw.

Examples:

- `read_kernel`
- `apply_patch`
- `build_kernel`
- `run_correctness`
- `run_benchmark`
- `run_profile`
- `parse_metrics`
- `save_report`

These skills should provide the action layer between the optimization agent and the CUDA task workspace.

See [`skills/CONTRACT.md`](/home/yuwei/Documents/Spiny-Claw/skills/CONTRACT.md) for the first minimum interface definition.

Current workspace skills:

- [`skills/local-compile-checks/SKILL.md`](/home/yuwei/Documents/Spiny-Claw/skills/local-compile-checks/SKILL.md)
  OpenClaw skill for local CUDA preflight checks against `kernel.cu` and `binding.py`.
