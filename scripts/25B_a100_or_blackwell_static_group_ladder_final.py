"""25B final collection: separate static extent, padding, and grouping for OLMoE experts.

This is deliberately a final-collection-only, expert-core microbenchmark on the saved
Layer-7 B=64 trace. It requires the separately reviewed 25B dry-run report and
performs no dry collection itself.

The 50 non-sentinel active experts have 465 real routed rows. The comparison is
the following causal ladder, always followed by the same M=47 sentinel call and
host materialization:

  dynamic                 50 calls, dynamic M, 465 real core rows
  static_exact             50 calls, E=1/static actual M, 465 core rows
  static_padded_e{1..50}  static M=47, 2,350 padded core rows, with exact
                           divisor group sizes E={1,2,5,10,25,50}.

The dynamic and static-exact sequences are pre-resolved (function handle plus
device arguments) before timing. The static-exact warmup traverses the complete
sequence, so every M-specialized module is warm. Inputs and pre-stacked weights
are GPU resident before timing; routing, packing, runtime weight selection,
combination, attention, and full-model work are excluded.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("MOE_IREE_ROOT", "/content/drive/MyDrive/moe_iree_olmoe"))
RUN_ID = "25B_a100_or_blackwell_static_group_ladder_final"
OUT = ROOT / RUN_ID
REPORT = OUT / f"{RUN_ID}_report.json"
FINAL_ROWS = OUT / f"{RUN_ID}_final_rows.csv"
SCRIPT_PATH = Path(globals().get("__file__", ROOT / f"{RUN_ID}.py"))

DRY_RUN_ID = "25B_a100_or_blackwell_static_group_ladder_dry"
DRY_REPORT = ROOT / DRY_RUN_ID / f"{DRY_RUN_ID}_report.json"

TRACE_PATH = ROOT / "olmoe_layer7_prefill64_decode1.pt"
WEIGHTS_PATH = ROOT / "olmoe_layer7_weights.safetensors"
DYNAMIC_SOURCE_PATH = ROOT / "olmoe_single_expert_dynamic.mlir"

HIDDEN = 2048
INTERMEDIATE = 1024
MAX_ROWS = 47
GROUP_SIZES = (1, 2, 5, 10, 25, 50)
RTOL = ATOL = 3e-4
WARMUPS = 2
SEED = 20261022
FINAL_CAP_SECONDS = 600.0

IREE_ARMS = ("iree_dynamic", "iree_static_exact", *(f"iree_static_padded_e{e}" for e in GROUP_SIZES))
TORCH_ARMS = ("pytorch_dynamic", "pytorch_static_exact", *(f"pytorch_static_padded_e{e}" for e in GROUP_SIZES))
ARMS = IREE_ARMS + TORCH_ARMS
FINAL_BLOCKS = 2 * len(ARMS)  # Each arm appears first exactly twice.
ROW_FIELDS = (
    "phase", "block_index", "sequence", "arm", "framework", "position_in_block",
    "duration_ms", "boundary", "warmups_per_arm", "core_expert_count",
    "core_assignment_count", "core_rows_executed", "core_call_count", "group_size",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clipped(text: str, limit: int = 12000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def command_result(command: list[str]) -> dict:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": result.returncode,
            "stdout": clipped(result.stdout), "stderr": clipped(result.stderr)}


def write_rows(rows: list[dict]) -> None:
    with FINAL_ROWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report["paths"] = {"output_directory": str(OUT), "report": str(REPORT), "final_rows": str(FINAL_ROWS)}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def print_summary(report: dict) -> None:
    print(json.dumps({"status": report["status"], "report": str(REPORT),
                      "final_rows": len(report["final_run"]["rows"]), "gates": report["gates"]}, indent=2))


def stop(report: dict, reason: str) -> None:
    report["status"] = "STOP_25B_FINAL_RUN_FAILED"
    report["failure_reason"] = reason
    report["next_step"] = "Preserve this report and partial rows. Review the failed gate; do not rerun into this output directory."
    write_rows(report["final_run"]["rows"])
    write_report(report)
    print_summary(report)


def static_source(module_name: str, expert_count: int, rows: int) -> str:
    """Exact no-bias OLMoE expert math with static [E,M,H] tensors."""
    shape_x = f"tensor<{expert_count}x{rows}x{HIDDEN}xf32>"
    shape_gu = f"tensor<{expert_count}x{HIDDEN}x{2 * INTERMEDIATE}xf32>"
    shape_proj = f"tensor<{expert_count}x{rows}x{2 * INTERMEDIATE}xf32>"
    shape_half = f"tensor<{expert_count}x{rows}x{INTERMEDIATE}xf32>"
    shape_down = f"tensor<{expert_count}x{INTERMEDIATE}x{HIDDEN}xf32>"
    return f'''module @{module_name} {{
  func.func @forward(%x: {shape_x}, %gate_up: {shape_gu}, %down: {shape_down}) -> {shape_x} {{
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %projected_empty = tensor.empty() : {shape_proj}
    %projected_init = linalg.fill ins(%zero : f32) outs(%projected_empty : {shape_proj}) -> {shape_proj}
    %projected = linalg.batch_matmul ins(%x, %gate_up : {shape_x}, {shape_gu}) outs(%projected_init : {shape_proj}) -> {shape_proj}
    %gate = tensor.extract_slice %projected[0, 0, 0] [{expert_count}, {rows}, {INTERMEDIATE}] [1, 1, 1] : {shape_proj} to {shape_half}
    %up = tensor.extract_slice %projected[0, 0, {INTERMEDIATE}] [{expert_count}, {rows}, {INTERMEDIATE}] [1, 1, 1] : {shape_proj} to {shape_half}
    %activated_empty = tensor.empty() : {shape_half}
    %activated = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%gate, %up : {shape_half}, {shape_half}) outs(%activated_empty : {shape_half}) {{
      ^bb0(%g: f32, %u: f32, %out: f32):
        %neg = arith.negf %g : f32
        %exp = math.exp %neg : f32
        %denom = arith.addf %one, %exp : f32
        %silu = arith.divf %g, %denom : f32
        %value = arith.mulf %silu, %u : f32
        linalg.yield %value : f32
    }} -> {shape_half}
    %output_empty = tensor.empty() : {shape_x}
    %output_init = linalg.fill ins(%zero : f32) outs(%output_empty : {shape_x}) -> {shape_x}
    %output = linalg.batch_matmul ins(%activated, %down : {shape_half}, {shape_down}) outs(%output_init : {shape_x}) -> {shape_x}
    return %output : {shape_x}
  }}
}}\n'''


@dataclass(frozen=True)
class ModuleSpec:
    label: str
    module_name: str
    expert_count: int
    rows: int


def static_orders(block_count: int) -> list[tuple[str, ...]]:
    if block_count != 2 * len(ARMS):
        raise ValueError("final blocks must equal twice the arm count for first-position balance")
    result = []
    for cycle in range(2):
        base = list(ARMS)
        random.Random(SEED + cycle).shuffle(base)
        result.extend(tuple(base[rotation:] + base[:rotation]) for rotation in range(len(base)))
    return result


def final_audit(rows: list[dict]) -> dict:
    expected = FINAL_BLOCKS * len(ARMS)
    sequences = [int(row["sequence"]) for row in rows]
    complete_blocks = all(
        len([row for row in rows if int(row["block_index"]) == block]) == len(ARMS)
        and {row["arm"] for row in rows if int(row["block_index"]) == block} == set(ARMS)
        for block in range(FINAL_BLOCKS)
    )
    first_counts = Counter(row["arm"] for row in rows if int(row["position_in_block"]) == 0)
    return {
        "expected_rows": expected,
        "actual_rows": len(rows),
        "contiguous_unique_sequences": sorted(sequences) == list(range(expected)),
        "complete_blocks": complete_blocks,
        "first_arm_counts": dict(sorted(first_counts.items())),
        "first_arm_balance": set(first_counts) == set(ARMS) and set(first_counts.values()) == {2},
        "positive_finite_durations": all(np.isfinite(float(row["duration_ms"])) and float(row["duration_ms"]) > 0 for row in rows),
    }


def medians(rows: list[dict]) -> dict[str, float]:
    return {arm: float(np.median([float(row["duration_ms"]) for row in rows if row["arm"] == arm])) for arm in ARMS}


def main() -> None:
    if REPORT.exists() or FINAL_ROWS.exists():
        print(json.dumps({"status": "STOP_EXISTING_OUTPUT", "report": str(REPORT),
                          "next_step": "Use a new run ID; preserve existing final-collection data."}, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "25B final collection: static-exact, padding, and static-group-size ladder",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": ("One captured OLMoE Layer-7 B=64 trace. Fifty non-sentinel active experts form the timed core; "
                  "the same M=47 sentinel static call is materialized after every arm. Expert core only: no router, "
                  "pack, runtime weight lookup, combine, attention, or full-model timing."),
        "metric": {"name": "common_sentinel_host_ready_after_gpu_resident_expert_core",
                   "includes": ["Python/runtime invocation", "GPU completion", "common static M=47 sentinel", "final host materialization"],
                   "excludes": ["compilation", "module loading", "initial input/weight setup", "warmups", "router", "pack", "weight gather", "combine", "full model"],
                   "not_a_measurement_of": ["kernel-only time", "whole-layer time", "whole-model latency"]},
        "protocol": {"dtype": "fp32", "tf32": "disabled in PyTorch", "warmups_per_arm": WARMUPS,
                     "final_blocks": FINAL_BLOCKS, "final_cap_seconds": FINAL_CAP_SECONDS, "order_seed": SEED,
                     "padded_group_sizes": list(GROUP_SIZES), "final_collection_included": True},
        "hashes": {"entrypoint": sha256(SCRIPT_PATH) if SCRIPT_PATH.is_file() else None,
                   "trace": sha256(TRACE_PATH) if TRACE_PATH.is_file() else None,
                   "weights": sha256(WEIGHTS_PATH) if WEIGHTS_PATH.is_file() else None,
                   "dynamic_source": sha256(DYNAMIC_SOURCE_PATH) if DYNAMIC_SOURCE_PATH.is_file() else None},
        "environment": {}, "artifacts": {}, "compile": {}, "correctness": {},
        "final_run": {"rows": [], "audit": {}, "median_ms_descriptive_only": {}},
        "gates": {"experiment_root_exists": ROOT.is_dir(), "real_trace_present": TRACE_PATH.is_file(),
                  "real_weight_bank_present": WEIGHTS_PATH.is_file(), "dynamic_source_present": DYNAMIC_SOURCE_PATH.is_file(),
                  "supported_gpu_visible": False, "iree_compiler_runtime_available": False,
                  "dynamic_vmfb_compiled": False, "all_static_vmfb_compiled": False, "all_modules_loaded": False,
                  "trace_contract_valid": False, "fp32_tf32_policy_set": False,
                  "accepted_dry_run": False,
                  "all_arms_gpu_resident_before_timing": False, "all_dynamic_and_static_outputs_match": False,
                  "pre_resolved_sequences_confirmed": False, "full_sequence_warmup_complete": False,
                  "final_run_complete_and_balanced": False, "final_run_time_cap_not_reached": False},
        "status": "STOP_PRECONDITION_FAILED", "next_step": "Resolve failed prerequisite before timing."}
    required_files = ("experiment_root_exists", "real_trace_present", "real_weight_bank_present", "dynamic_source_present")
    if not all(report["gates"][name] for name in required_files):
        stop(report, "Missing required real-model artifact")
        return
    try:
        dry_report = json.loads(DRY_REPORT.read_text(encoding="utf-8"))
        required_dry_gates = (
            "all_static_vmfb_compiled", "all_modules_loaded", "all_dynamic_and_static_outputs_match",
            "pre_resolved_sequences_confirmed", "full_sequence_warmup_complete",
            "dry_run_complete_and_balanced", "dry_run_time_cap_not_reached",
        )
        if dry_report.get("status") != "REVIEW_REQUIRED_25B_DRY_RUN":
            raise RuntimeError(f"unexpected dry status: {dry_report.get('status')!r}")
        if len(dry_report.get("dry_run", {}).get("rows", [])) != 256:
            raise RuntimeError("dry report does not contain the required 256 balanced rows")
        if not all(dry_report.get("gates", {}).get(name) is True for name in required_dry_gates):
            raise RuntimeError("a required 25B dry-run gate is not true")
        if dry_report.get("protocol", {}).get("padded_group_sizes") != list(GROUP_SIZES):
            raise RuntimeError("dry report group-size ladder differs from this final protocol")
        for artifact_name in ("trace", "weights", "dynamic_source"):
            if dry_report.get("hashes", {}).get(artifact_name) != report["hashes"][artifact_name]:
                raise RuntimeError(f"dry report {artifact_name} hash does not match the current artifact")
        report["dry_run_provenance"] = {"report": str(DRY_REPORT), "sha256": sha256(DRY_REPORT),
                                        "status": dry_report["status"], "row_count": 256,
                                        "padded_group_sizes": dry_report["protocol"]["padded_group_sizes"]}
        report["gates"]["accepted_dry_run"] = True
    except Exception as exc:
        report["dry_run_provenance_error"] = traceback.format_exc(limit=10)
        stop(report, f"Required reviewed 25B dry run is unavailable or inconsistent: {exc!r}")
        return

    try:
        import torch
        import torch.nn.functional as F
        from iree import compiler as ireec
        from iree import runtime as ireert
        from safetensors import safe_open

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot see CUDA")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        gpu = torch.cuda.get_device_properties(0)
        target_arch = f"sm_{gpu.major}{gpu.minor}"
        gpu_name = gpu.name.upper()
        report["environment"] = {
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "iree_base_compiler": importlib.metadata.version("iree-base-compiler"),
            "iree_base_runtime": importlib.metadata.version("iree-base-runtime"),
            "gpu": {"name": gpu.name, "total_memory_bytes": int(gpu.total_memory),
                    "compute_capability": f"{gpu.major}.{gpu.minor}", "target_arch": target_arch},
            "torch_tf32": {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                           "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)},
        }
        report["gates"]["supported_gpu_visible"] = int(gpu.major) >= 10 or ("A100" in gpu_name and int(gpu.major) == 8)
        report["gates"]["iree_compiler_runtime_available"] = shutil.which("iree-compile") is not None
        report["gates"]["fp32_tf32_policy_set"] = not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    except Exception as exc:
        report["environment_error"] = traceback.format_exc(limit=10)
        stop(report, f"CUDA/IREE import or FP32 setup failed: {exc!r}")
        return
    if not all(report["gates"][name] for name in ("supported_gpu_visible", "iree_compiler_runtime_available", "fp32_tf32_policy_set")):
        stop(report, "Expected A100/Blackwell GPU or required compiler/FP32 policy unavailable")
        return
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        command = [nvidia_smi, "--query-gpu=name,uuid,driver_version,pstate,memory.used,memory.total,clocks.current.sm,temperature.gpu", "--format=csv,noheader"]
        report["environment"]["nvidia_smi_before"] = command_result(command)

    try:
        bundle = torch.load(TRACE_PATH, map_location="cpu", weights_only=False)["prefill64"]
        x = np.ascontiguousarray(bundle["moe_input"][0].numpy().astype(np.float32, copy=False))
        expert_ids = np.ascontiguousarray(bundle["expert_ids"].numpy().astype(np.int32, copy=False))
        if x.shape != (64, HIDDEN) or expert_ids.shape != (64, 8):
            raise RuntimeError(f"unexpected trace shapes: {x.shape}, {expert_ids.shape}")
        counts = np.bincount(expert_ids.reshape(-1), minlength=64)
        active = [int(eid) for eid in np.flatnonzero(counts)]
        sentinel = int(np.argmax(counts))
        if len(active) != 51 or int(counts[sentinel]) != MAX_ROWS:
            raise RuntimeError(f"expected 51 active experts and M=47 sentinel, found {len(active)} and {counts[sentinel]}")
        core = [eid for eid in active if eid != sentinel]
        if len(core) != 50 or any(len(core) % size for size in GROUP_SIZES):
            raise RuntimeError("core must contain 50 experts divisible by every declared group size")
        token_rows = {eid: np.ascontiguousarray(x[np.where(expert_ids == eid)[0]], dtype=np.float32) for eid in active}
        core_rows = int(sum(token_rows[eid].shape[0] for eid in core))
        if core_rows != 465 or sum(rows.shape[0] for rows in token_rows.values()) != 512:
            raise RuntimeError(f"unexpected core/total assignment count: {core_rows}/{sum(rows.shape[0] for rows in token_rows.values())}")
        exact_rows = sorted({int(token_rows[eid].shape[0]) for eid in core})
        report["artifacts"]["trace"] = {
            "active_experts": active, "counts": [int(value) for value in counts],
            "active_expert_count": len(active), "assignment_count": int(expert_ids.size),
            "sentinel_expert": sentinel, "sentinel_rows": int(counts[sentinel]),
            "core_expert_ids_in_execution_order": core, "core_expert_count": len(core),
            "core_assignment_count": core_rows, "static_exact_rows": exact_rows,
            "static_exact_module_count": len(exact_rows), "padded_rows_per_group_ladder_arm": len(core) * MAX_ROWS,
            "padded_to_real_core_row_ratio": (len(core) * MAX_ROWS) / core_rows,
            "padded_groups": {str(size): [core[index:index + size] for index in range(0, len(core), size)] for size in GROUP_SIZES},
        }
        report["gates"]["trace_contract_valid"] = True
    except Exception as exc:
        report["trace_error"] = traceback.format_exc(limit=12)
        stop(report, f"Captured-trace preparation failed: {exc!r}")
        return

    module_specs: list[ModuleSpec] = [
        *(ModuleSpec(f"exact_m{rows}", f"olmoe_{RUN_ID}_exact_m{rows}", 1, rows) for rows in exact_rows),
        *(ModuleSpec(f"padded_e{size}", f"olmoe_{RUN_ID}_padded_e{size}", size, MAX_ROWS) for size in GROUP_SIZES),
        ModuleSpec("sentinel", f"olmoe_{RUN_ID}_sentinel_m47", 1, MAX_ROWS),
    ]
    target_arch = report["environment"]["gpu"]["target_arch"]
    compile_args = [f"--iree-cuda-target={target_arch}", "--iree-hal-target-backends=cuda"]
    if target_arch == "sm_120":
        compile_args.append("--iree-cuda-target-features=+ptx87")
    try:
        sources = {"dynamic": DYNAMIC_SOURCE_PATH.read_text(encoding="utf-8")}
        for spec in module_specs:
            source = static_source(spec.module_name, spec.expert_count, spec.rows)
            (OUT / f"{spec.label}.mlir").write_text(source, encoding="utf-8")
            sources[spec.label] = source
        vmfbs, compile_meta = {}, {}
        for label, source in sources.items():
            started = time.perf_counter()
            vmfb = ireec.compile_str(source, target_backends=["cuda"], extra_args=compile_args)
            path = OUT / ("dynamic.vmfb" if label == "dynamic" else f"{label}.vmfb")
            path.write_bytes(vmfb)
            vmfbs[label] = vmfb
            compile_meta[label] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
                                   "seconds": time.perf_counter() - started}
        report["compile"] = {"extra_args": compile_args, "modules": compile_meta}
        report["gates"]["dynamic_vmfb_compiled"] = "dynamic" in vmfbs
        report["gates"]["all_static_vmfb_compiled"] = len(vmfbs) == 1 + len(module_specs)
    except Exception as exc:
        report["compile_error"] = traceback.format_exc(limit=14)
        stop(report, f"IREE CUDA compilation failed: {exc!r}")
        return

    try:
        cfg = ireert.Config("cuda")
        ctx = ireert.SystemContext(config=cfg)
        module_functions = {}
        dynamic_module = ireert.VmModule.from_flatbuffer(cfg.vm_instance, vmfbs["dynamic"], warn_if_copy=False)
        ctx.add_vm_module(dynamic_module)
        module_functions["dynamic"] = ctx.modules.olmoe_expert.forward
        for spec in module_specs:
            module = ireert.VmModule.from_flatbuffer(cfg.vm_instance, vmfbs[spec.label], warn_if_copy=False)
            ctx.add_vm_module(module)
            module_functions[spec.label] = getattr(ctx.modules, spec.module_name).forward
        report["gates"]["all_modules_loaded"] = len(module_functions) == 1 + len(module_specs)
    except Exception as exc:
        report["module_load_error"] = traceback.format_exc(limit=12)
        stop(report, f"IREE CUDA module load failed: {exc!r}")
        return

    try:
        gate_np, down_np = {}, {}
        with safe_open(WEIGHTS_PATH, framework="pt", device="cpu") as sf:
            gate_bank, down_bank = sf.get_tensor("gate_up_proj"), sf.get_tensor("down_proj")
            if tuple(gate_bank.shape) != (64, HIDDEN, 2 * INTERMEDIATE) or tuple(down_bank.shape) != (64, HIDDEN, INTERMEDIATE):
                raise RuntimeError(f"unexpected weight shapes {tuple(gate_bank.shape)}, {tuple(down_bank.shape)}")
            for eid in range(64):
                gate_np[eid] = np.array(gate_bank[eid].T.contiguous().numpy(), dtype=np.float32, order="C", copy=True)
                down_np[eid] = np.array(down_bank[eid].T.contiguous().numpy(), dtype=np.float32, order="C", copy=True)

        iree_dynamic_weights = {eid: (ireert.asdevicearray(cfg.device, gate_np[eid]), ireert.asdevicearray(cfg.device, down_np[eid])) for eid in range(64)}
        torch_dynamic_weights = {eid: (torch.from_numpy(gate_np[eid]).to("cuda"), torch.from_numpy(down_np[eid]).to("cuda")) for eid in range(64)}
        iree_dynamic_inputs = {eid: ireert.asdevicearray(cfg.device, token_rows[eid]) for eid in active}
        torch_dynamic_inputs = {eid: torch.from_numpy(token_rows[eid]).to("cuda") for eid in active}

        def packed_values(expert_group: list[int], rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            packed = np.zeros((len(expert_group), rows, HIDDEN), dtype=np.float32)
            for index, eid in enumerate(expert_group):
                packed[index, :token_rows[eid].shape[0]] = token_rows[eid]
            return packed, np.stack([gate_np[eid] for eid in expert_group]), np.stack([down_np[eid] for eid in expert_group])

        iree_exact_args, torch_exact_args = {}, {}
        for eid in core:
            values = packed_values([eid], int(token_rows[eid].shape[0]))
            iree_exact_args[eid] = tuple(ireert.asdevicearray(cfg.device, value) for value in values)
            torch_exact_args[eid] = tuple(torch.from_numpy(value).to("cuda") for value in values)
        iree_padded_args, torch_padded_args = {}, {}
        for size in GROUP_SIZES:
            for group_index, group in enumerate(report["artifacts"]["trace"]["padded_groups"][str(size)]):
                values = packed_values(group, MAX_ROWS)
                key = (size, group_index)
                iree_padded_args[key] = tuple(ireert.asdevicearray(cfg.device, value) for value in values)
                torch_padded_args[key] = tuple(torch.from_numpy(value).to("cuda") for value in values)
        sentinel_values = packed_values([sentinel], MAX_ROWS)
        iree_sentinel_args = tuple(ireert.asdevicearray(cfg.device, value) for value in sentinel_values)
        torch_sentinel_args = tuple(torch.from_numpy(value).to("cuda") for value in sentinel_values)
        torch.cuda.synchronize()
        report["artifacts"]["gpu_residency"] = {
            "dynamic_bank_experts": len(iree_dynamic_weights), "exact_static_input_sets": len(iree_exact_args),
            "padded_static_input_sets": len(iree_padded_args), "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        }
        report["gates"]["all_arms_gpu_resident_before_timing"] = (
            len(iree_dynamic_weights) == 64 and len(iree_exact_args) == len(core)
            and len(iree_padded_args) == sum(len(core) // size for size in GROUP_SIZES)
        )
    except Exception as exc:
        report["residency_error"] = traceback.format_exc(limit=14)
        stop(report, f"GPU-residency setup failed: {exc!r}")
        return

    def torch_dynamic_call(args):
        x_value, gate, down = args
        projected = x_value @ gate
        gate_part, up = projected.chunk(2, dim=-1)
        return (F.silu(gate_part) * up) @ down

    def torch_static_call(args):
        x_value, gate, down = args
        projected = torch.bmm(x_value, gate)
        gate_part, up = projected.chunk(2, dim=-1)
        return torch.bmm(F.silu(gate_part) * up, down)

    dynamic_iree_sequence = [(module_functions["dynamic"], (iree_dynamic_inputs[eid], *iree_dynamic_weights[eid])) for eid in core]
    dynamic_torch_sequence = [(torch_dynamic_call, (torch_dynamic_inputs[eid], *torch_dynamic_weights[eid])) for eid in core]
    exact_iree_sequence = [(module_functions[f"exact_m{token_rows[eid].shape[0]}"], iree_exact_args[eid]) for eid in core]
    exact_torch_sequence = [(torch_static_call, torch_exact_args[eid]) for eid in core]
    padded_iree_sequences = {size: [(module_functions[f"padded_e{size}"], iree_padded_args[(size, index)]) for index in range(len(core) // size)] for size in GROUP_SIZES}
    padded_torch_sequences = {size: [(torch_static_call, torch_padded_args[(size, index)]) for index in range(len(core) // size)] for size in GROUP_SIZES}
    report["gates"]["pre_resolved_sequences_confirmed"] = (
        len(dynamic_iree_sequence) == len(exact_iree_sequence) == len(core)
        and all(len(padded_iree_sequences[size]) == len(core) // size for size in GROUP_SIZES)
    )

    try:
        reference = {}
        with torch.inference_mode():
            for eid in active:
                reference[eid] = torch_dynamic_call((torch_dynamic_inputs[eid], *torch_dynamic_weights[eid])).detach().cpu().numpy()
        max_error = 0.0
        for eid in active:
            out = module_functions["dynamic"](iree_dynamic_inputs[eid], *iree_dynamic_weights[eid]).to_host()
            np.testing.assert_allclose(out, reference[eid], rtol=RTOL, atol=ATOL)
            max_error = max(max_error, float(np.max(np.abs(out - reference[eid]))))
        for eid in core:
            out = module_functions[f"exact_m{token_rows[eid].shape[0]}"](*iree_exact_args[eid]).to_host()
            np.testing.assert_allclose(out[0], reference[eid], rtol=RTOL, atol=ATOL)
            max_error = max(max_error, float(np.max(np.abs(out[0] - reference[eid]))))
        for size in GROUP_SIZES:
            for index, group in enumerate(report["artifacts"]["trace"]["padded_groups"][str(size)]):
                iree_out = module_functions[f"padded_e{size}"](*iree_padded_args[(size, index)]).to_host()
                torch_out = torch_static_call(torch_padded_args[(size, index)]).detach().cpu().numpy()
                np.testing.assert_allclose(iree_out, torch_out, rtol=RTOL, atol=ATOL)
                max_error = max(max_error, float(np.max(np.abs(iree_out - torch_out))))
                for group_position, eid in enumerate(group):
                    valid = token_rows[eid].shape[0]
                    np.testing.assert_allclose(iree_out[group_position, :valid], reference[eid], rtol=RTOL, atol=ATOL)
        sentinel_out = module_functions["sentinel"](*iree_sentinel_args).to_host()
        np.testing.assert_allclose(sentinel_out[0], reference[sentinel], rtol=RTOL, atol=ATOL)
        report["correctness"] = {"rtol": RTOL, "atol": ATOL, "max_abs_error": max_error,
                                 "dynamic_experts_checked": len(active), "static_exact_experts_checked": len(core),
                                 "static_padded_group_sizes_checked": list(GROUP_SIZES)}
        report["gates"]["all_dynamic_and_static_outputs_match"] = True
    except Exception as exc:
        report["correctness_error"] = traceback.format_exc(limit=14)
        stop(report, f"Numerical correctness gate failed: {exc!r}")
        return

    def invoke_iree(sequence):
        for fn, args in sequence:
            fn(*args)
        return module_functions["sentinel"](*iree_sentinel_args).to_host()

    def invoke_torch_dynamic(sequence):
        with torch.inference_mode():
            for fn, args in sequence:
                fn(args)
            return torch_static_call(torch_sentinel_args).detach().cpu().numpy()

    def invoke_torch_static(sequence):
        with torch.inference_mode():
            for fn, args in sequence:
                fn(args)
            return torch_static_call(torch_sentinel_args).detach().cpu().numpy()

    calls = {
        "iree_dynamic": lambda: invoke_iree(dynamic_iree_sequence),
        "iree_static_exact": lambda: invoke_iree(exact_iree_sequence),
        "pytorch_dynamic": lambda: invoke_torch_dynamic(dynamic_torch_sequence),
        "pytorch_static_exact": lambda: invoke_torch_static(exact_torch_sequence),
    }
    for size in GROUP_SIZES:
        calls[f"iree_static_padded_e{size}"] = lambda size=size: invoke_iree(padded_iree_sequences[size])
        calls[f"pytorch_static_padded_e{size}"] = lambda size=size: invoke_torch_static(padded_torch_sequences[size])
    arm_metadata = {
        "iree_dynamic": ("iree", core_rows, len(core), None),
        "iree_static_exact": ("iree", core_rows, len(core), 1),
        "pytorch_dynamic": ("pytorch", core_rows, len(core), None),
        "pytorch_static_exact": ("pytorch", core_rows, len(core), 1),
    }
    for size in GROUP_SIZES:
        for framework in ("iree", "pytorch"):
            arm_metadata[f"{framework}_static_padded_e{size}"] = (framework, len(core) * MAX_ROWS, len(core) // size, size)

    try:
        for arm in ARMS:
            for _ in range(WARMUPS):
                calls[arm]()
        report["gates"]["full_sequence_warmup_complete"] = True
        rows, sequence = [], 0
        started, capped = time.perf_counter(), False
        for block_index, order in enumerate(static_orders(FINAL_BLOCKS)):
            if time.perf_counter() - started >= FINAL_CAP_SECONDS:
                capped = True
                break
            for position, arm in enumerate(order):
                begin = time.perf_counter_ns()
                calls[arm]()
                elapsed_ms = (time.perf_counter_ns() - begin) / 1_000_000.0
                framework, static_rows, call_count, group_size = arm_metadata[arm]
                rows.append({"phase": "25B final collection", "block_index": block_index, "sequence": sequence,
                             "arm": arm, "framework": framework, "position_in_block": position,
                             "duration_ms": elapsed_ms,
                             "boundary": "common static-M47 sentinel output materialized on host after GPU-resident expert core",
                             "warmups_per_arm": WARMUPS, "core_expert_count": len(core),
                             "core_assignment_count": core_rows, "core_rows_executed": static_rows,
                             "core_call_count": call_count, "group_size": group_size})
                sequence += 1
            report["final_run"]["rows"] = rows
            write_rows(rows)
            write_report(report)
        audit = final_audit(rows)
        report["final_run"].update({"rows": rows, "elapsed_seconds": time.perf_counter() - started,
                                  "audit": audit, "median_ms_descriptive_only": medians(rows)})
        report["gates"]["final_run_time_cap_not_reached"] = not capped
        report["gates"]["final_run_complete_and_balanced"] = all((
            audit["actual_rows"] == audit["expected_rows"], audit["contiguous_unique_sequences"],
            audit["complete_blocks"], audit["first_arm_balance"], audit["positive_finite_durations"],
        ))
    except Exception as exc:
        report["final_run_error"] = traceback.format_exc(limit=14)
        stop(report, f"Final-collection collection failed: {exc!r}")
        return
    if nvidia_smi:
        report["environment"]["nvidia_smi_after"] = command_result(report["environment"]["nvidia_smi_before"]["command"])
    report["status"] = "COMPLETE_PENDING_AUDIT_25B" if all(report["gates"].values()) else "STOP_25B_FINAL_RUN_FAILED"
    report["next_step"] = (
        "Review the final CSV/report before interpreting final results. This script is the final collection; audit its report and CSV before interpreting speedups."
        if report["status"] == "COMPLETE_PENDING_AUDIT_25B" else "Review the failed final-collection gate; do not interpret this collection as a performance conclusion."
    )
    write_rows(report["final_run"]["rows"])
    write_report(report)
    print_summary(report)


if __name__ == "__main__":
    main()

