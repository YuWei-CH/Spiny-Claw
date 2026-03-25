---
name: local_compile_checks
description: Use this skill when you need a local preflight check for a CUDA task before Modal execution, especially to validate task/<name>/solution/cuda/kernel.cu compilation readiness and task/<name>/solution/cuda/binding.py contract consistency.
metadata:
  openclaw:
    os: ["linux"]
    requires:
      bins: ["python3"]
---

# Local Compile Checks

Use this skill when the user wants to locally validate a CUDA task before running it on Modal.

This skill performs two checks:

1. a static compile-oriented check for `solution/cuda/kernel.cu`
2. a binding contract check for `solution/cuda/binding.py`

## How To Run

Run the bundled script with the task directory:

```bash
python3 skills/local-compile-checks/scripts/run_local_compile_checks.py --task-dir task/Fused-MoE-YWS
```

Optional flags:

- `--output-dir <dir>` to override the default artifact location
- `--cuda-home <path>` if CUDA is installed outside the common system locations
- `--tvm-include-dir <path>` if TVM headers are installed in a nonstandard location
- `--extra-nvcc-flag <flag>` to forward additional compile flags

## What The Script Checks

### Binding Contract

The script validates:

- `config.toml` exists and declares a CUDA build
- `solution/cuda/binding.py` exists and has valid Python syntax
- `solution/cuda/kernel.cu` exists
- `entry_point` matches the CUDA source file and exported symbol
- `binding.py` registers a TVM entry function
- `binding.py` references the expected callable
- `kernel.cu` exports the expected symbol

It also emits warnings when the binding style and kernel export style appear inconsistent.

### Kernel Compile Readiness

The script validates:

- `nvcc` is available on `PATH`
- CUDA headers can be found
- `kernel.cu` can be compiled as a translation unit with the available toolchain and include paths

## Output

Artifacts are written to:

```text
results/<task-name>/compile_result/
```

The script writes:

- `summary.json`
- `binding_contract.json`
- `kernel_compile.json`
- `kernel_compile.log`

## Failure Handling

- If `nvcc` is missing, report that the local CUDA toolchain is not ready and point the user to [`scripts/init_cuda_toolchain.sh`](/home/yuwei/Documents/Spiny-Claw/scripts/init_cuda_toolchain.sh), which now handles installation and verification.
- If CUDA is installed but TVM FFI headers are missing, rerun with `--tvm-include-dir <path>` or set `TVM_INCLUDE_DIR`.
- If the binding contract fails, report those findings before remote execution.
- If the kernel compile fails, include the exact `nvcc` command and log path in the response.
