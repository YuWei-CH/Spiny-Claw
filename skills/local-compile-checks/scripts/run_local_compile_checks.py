#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import py_compile
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class CheckResult:
    name: str
    status: str
    details: list[str]

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "details": self.details}


def load_config(task_dir: Path) -> dict:
    config_path = task_dir / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with config_path.open("rb") as f:
        return tomllib.load(f)


def parse_entry_point(entry_point: str) -> tuple[str | None, str | None]:
    if "::" not in entry_point:
        return None, None
    source_file, symbol = entry_point.split("::", 1)
    return source_file.strip(), symbol.strip()


def compile_python_file(path: Path) -> tuple[bool, str | None]:
    try:
        py_compile.compile(str(path), doraise=True)
        return True, None
    except py_compile.PyCompileError as exc:
        return False, str(exc)


def safe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_binding_contract_check(task_dir: Path) -> tuple[dict, str | None]:
    results: list[CheckResult] = []
    warning: str | None = None

    config = load_config(task_dir)
    build_config = config.get("build", {})

    binding_path = task_dir / "solution" / "cuda" / "binding.py"
    kernel_path = task_dir / "solution" / "cuda" / "kernel.cu"

    language = build_config.get("language")
    binding_mode = build_config.get("binding")
    entry_point = build_config.get("entry_point", "")
    entry_source, entry_symbol = parse_entry_point(entry_point)

    results.append(
        CheckResult(
            name="build_language",
            status="pass" if language == "cuda" else "fail",
            details=[f"config.toml build.language = {language!r}"],
        )
    )
    results.append(
        CheckResult(
            name="build_binding",
            status="pass" if binding_mode == "tvm-ffi" else "warn",
            details=[f"config.toml build.binding = {binding_mode!r}"],
        )
    )
    results.append(
        CheckResult(
            name="entry_point_format",
            status="pass" if entry_source and entry_symbol else "fail",
            details=[f"config.toml build.entry_point = {entry_point!r}"],
        )
    )

    results.append(
        CheckResult(
            name="binding_file_exists",
            status="pass" if binding_path.exists() else "fail",
            details=[f"binding path: {binding_path}"],
        )
    )
    results.append(
        CheckResult(
            name="kernel_file_exists",
            status="pass" if kernel_path.exists() else "fail",
            details=[f"kernel path: {kernel_path}"],
        )
    )

    if not binding_path.exists() or not kernel_path.exists():
        report = {
            "task_dir": str(task_dir),
            "status": summarize_status([r.status for r in results]),
            "checks": [r.to_dict() for r in results],
        }
        return report, warning

    py_ok, py_error = compile_python_file(binding_path)
    results.append(
        CheckResult(
            name="binding_python_syntax",
            status="pass" if py_ok else "fail",
            details=["Python syntax check passed."] if py_ok else [py_error or "Syntax error."],
        )
    )

    binding_source = safe_read(binding_path)
    kernel_source = safe_read(kernel_path)

    results.append(
        CheckResult(
            name="entry_point_source_matches_kernel",
            status="pass" if entry_source == "kernel.cu" else "fail",
            details=[f"entry source: {entry_source!r}"],
        )
    )
    results.append(
        CheckResult(
            name="entry_point_symbol_present",
            status="pass"
            if entry_symbol and re.search(rf"\b{re.escape(entry_symbol)}\s*\(", kernel_source)
            else "fail",
            details=[f"expected symbol: {entry_symbol!r}"],
        )
    )

    register_matches = re.findall(r'register_func\("([^"]+)"\)', binding_source)
    results.append(
        CheckResult(
            name="binding_registers_tvm_function",
            status="pass" if "flashinfer.kernel" in register_matches else "warn",
            details=register_matches or ["No TVM register_func call found."],
        )
    )

    load_style = "torch_cpp_extension_load" if "torch.utils.cpp_extension" in binding_source else "unknown"
    results.append(
        CheckResult(
            name="binding_load_style",
            status="pass" if load_style == "torch_cpp_extension_load" else "warn",
            details=[f"detected load style: {load_style}"],
        )
    )

    binding_calls_entry = (
        f"_load_extension().{entry_symbol}(" in binding_source if entry_symbol else False
    )
    results.append(
        CheckResult(
            name="binding_calls_expected_symbol",
            status="pass" if binding_calls_entry else "warn",
            details=[f"expected call target: _load_extension().{entry_symbol}(...)"],
        )
    )

    exports_tvm_symbol = bool(
        entry_symbol
        and re.search(
            rf"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(\s*{re.escape(entry_symbol)}\s*,\s*{re.escape(entry_symbol)}\s*\)",
            kernel_source,
        )
    )
    results.append(
        CheckResult(
            name="kernel_exports_tvm_symbol",
            status="pass" if exports_tvm_symbol else "warn",
            details=[f"expected TVM export for symbol: {entry_symbol!r}"],
        )
    )

    if load_style == "torch_cpp_extension_load" and exports_tvm_symbol:
        warning = (
            "binding.py appears to JIT load kernel.cu as a PyTorch extension, while kernel.cu "
            "exports a TVM FFI typed function. Verify that the runtime expects this combination."
        )

    dependency_checks = []
    for module_name in ("torch", "tvm"):
        dependency_checks.append(
            {
                "module": module_name,
                "available": module_available(module_name),
            }
        )

    report = {
        "task_dir": str(task_dir),
        "status": summarize_status([r.status for r in results], warning=warning),
        "checks": [r.to_dict() for r in results],
        "dependency_probe": dependency_checks,
    }
    if warning:
        report["warning"] = warning
    return report, warning


def module_available(module_name: str) -> bool:
    try:
        __import__("importlib.util")
        import importlib.util

        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def summarize_status(statuses: list[str], warning: str | None = None) -> str:
    if any(status == "fail" for status in statuses):
        return "fail"
    if warning or any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def detect_cuda_root(user_cuda_home: str | None) -> str:
    candidates: list[Path] = []
    if user_cuda_home:
        candidates.append(Path(user_cuda_home))
    candidates.extend(
        [
            Path("/usr/local/cuda"),
            Path("/usr/lib/cuda"),
            Path("/usr/lib/nvidia-cuda-toolkit"),
            Path("/usr"),
        ]
    )
    for candidate in candidates:
        if (candidate / "include" / "cuda_runtime.h").exists():
            return str(candidate)
    return user_cuda_home or "/usr/local/cuda"


def run_kernel_compile_check(
    task_dir: Path,
    output_dir: Path,
    cuda_home: str,
    tvm_include_dir: str | None,
    extra_nvcc_flags: list[str],
) -> dict:
    kernel_path = task_dir / "solution" / "cuda" / "kernel.cu"
    log_path = output_dir / "kernel_compile.log"
    obj_path = output_dir / "kernel_compile.o"

    if not kernel_path.exists():
        return {
            "status": "fail",
            "reason": f"Missing kernel source: {kernel_path}",
            "log_path": str(log_path),
        }

    nvcc = shutil.which("nvcc")
    if not nvcc:
        log_path.write_text("nvcc not found on PATH\n", encoding="utf-8")
        return {
            "status": "fail",
            "reason": "nvcc not found on PATH",
            "log_path": str(log_path),
            "hint": "Run scripts/init_cuda_toolchain.sh to install and verify the local CUDA toolchain before using this check.",
        }

    resolved_cuda_home = detect_cuda_root(cuda_home)
    cmd = [nvcc, "-std=c++17", "-c", str(kernel_path), "-o", str(obj_path)]

    cuda_include = Path(resolved_cuda_home) / "include"
    if cuda_include.exists():
        cmd.extend(["-I", str(cuda_include)])
    if tvm_include_dir:
        cmd.extend(["-I", tvm_include_dir])

    extra_from_env = shlex.split(os.environ.get("EXTRA_NVCC_FLAGS", ""))
    cmd.extend(extra_nvcc_flags)
    cmd.extend(extra_from_env)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=task_dir,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")

    report = {
        "status": "pass" if proc.returncode == 0 else "fail",
        "command": cmd,
        "exit_code": proc.returncode,
        "log_path": str(log_path),
        "object_path": str(obj_path),
        "cuda_home": resolved_cuda_home,
        "tvm_include_dir": tvm_include_dir,
    }
    if proc.returncode != 0:
        compile_log = proc.stdout
        if "tvm/ffi/" in compile_log:
            report["hint"] = (
                "TVM headers were not found during local compilation. "
                "Set TVM_INCLUDE_DIR or pass --tvm-include-dir to this script."
            )
    return report


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local compile-oriented checks for a CUDA task."
    )
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cuda-home", default=os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    parser.add_argument(
        "--tvm-include-dir",
        default=os.environ.get("TVM_INCLUDE_DIR"),
        help="Optional include path for TVM headers.",
    )
    parser.add_argument(
        "--extra-nvcc-flag",
        action="append",
        default=[],
        help="Additional flag forwarded to nvcc. Repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    task_name = task_dir.name

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else REPO_ROOT / "results" / task_name / "compile_result"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    binding_report, warning = run_binding_contract_check(task_dir)
    kernel_report = run_kernel_compile_check(
        task_dir=task_dir,
        output_dir=output_dir,
        cuda_home=args.cuda_home,
        tvm_include_dir=args.tvm_include_dir,
        extra_nvcc_flags=args.extra_nvcc_flag,
    )

    write_json(output_dir / "binding_contract.json", binding_report)
    write_json(output_dir / "kernel_compile.json", kernel_report)

    statuses = [binding_report["status"], kernel_report["status"]]
    summary = {
        "task_name": task_name,
        "task_dir": str(task_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": summarize_status(statuses, warning=warning),
        "checks": {
            "binding_contract": binding_report["status"],
            "kernel_compile": kernel_report["status"],
        },
        "artifacts": {
            "binding_contract": str(output_dir / "binding_contract.json"),
            "kernel_compile": str(output_dir / "kernel_compile.json"),
            "kernel_compile_log": str(output_dir / "kernel_compile.log"),
        },
    }
    if warning:
        summary["warning"] = warning

    write_json(output_dir / "summary.json", summary)

    print(f"Task: {task_name}")
    print(f"Output: {output_dir}")
    print(f"Binding contract: {binding_report['status']}")
    print(f"Kernel compile: {kernel_report['status']}")
    if warning:
        print(f"Warning: {warning}")

    return 0 if summary["status"] != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
