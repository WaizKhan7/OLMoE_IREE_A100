"""25C1 adversarial functional-correctness gate; deliberately not a benchmark.

The accepted trace-specialized VMFBs are compiled and loaded once. Active-51
is used only for the 51-active primary trace and Active-55 only for the
55-active second trace; Stationary-64 is reused for both. The graph arms return
expert-grouped values to the existing combine module. This runner checks the
route/pack stage, those expert values, and the final Layer-7 output against
independent host references. It has no timer, warmup, duration collection, or
performance conclusion.

Set MOE_IREE_SECOND_TRACE to a different recorded Layer-7 prefill trace before
running. A missing or duplicate second trace is an intentional stop condition.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("MOE_IREE_ROOT", "/content/drive/MyDrive/moe_iree_olmoe"))
SOURCES = ROOT / "25C1_graph_sources"
LOWERING = ROOT / "25C1_lowering_review" / "25C1_graph_lowering_report.json"
STRUCTURE = ROOT / "25C0_selected_structure" / "25C0_selected_structure_report.json"
OUT = ROOT / "25C1_functional_correctness"
REPORT = OUT / "25C1_functional_correctness_report.json"
IREE_COMPILE = shutil.which("iree-compile")
TARGET_ARCH = "sm_80"
RTOL = 3e-4
ATOL = 3e-4
CAPACITY = 47
EXPECTED_GRAPH_HASHES = {
    "active51": "629e035d49657364976efef846cb48ecaa334e4407f8a822c6a1fa6543b7ddfa",
    "active55": "d0055d64a3fee0c1300300da093421f8eb307cf82a54643e0599c84440eafaac",
    "stationary64": "b4805fcefb884772e57593d0ab764c02b48ab4dc806785e9d90f20e25b85cc93",
}
TRACE_CONTRACTS = {
    "primary": {
        "active_arm": "active51",
        "expected": {"assignments": 512, "active_expert_count": 51,
                     "inactive_expert_count": 13, "max_expert_count": 47},
    },
    "second": {
        "active_arm": "active55",
        "expected": {"assignments": 512, "active_expert_count": 55,
                     "inactive_expert_count": 9, "max_expert_count": 45},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def clipped(text: str, limit: int = 12000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def write_and_print(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "report": str(REPORT),
        "gates": report["gates"],
        "checks": report.get("checks"),
        "next_step": report["next_step"],
    }, indent=2))


def require_mapping_item(mapping: dict, label: str, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"{label} not found; expected one of {names}, found {sorted(mapping)}")


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def one_candidate(label: str, patterns: tuple[str, ...]) -> Path:
    candidates = {path for pattern in patterns for path in ROOT.glob(pattern) if path.is_file()}
    if len(candidates) != 1:
        rendered = [str(path) for path in sorted(candidates)]
        raise FileNotFoundError(f"{label} requires exactly one candidate; found {rendered}")
    return next(iter(candidates))


def resolve_traces() -> tuple[Path, Path]:
    primary_text = os.environ.get("MOE_IREE_PRIMARY_TRACE")
    primary = Path(primary_text) if primary_text else one_candidate(
        "primary Layer-7 trace", ("*layer7*prefill64*decode1*.pt",)
    )
    if not primary.is_file():
        raise FileNotFoundError(f"Primary trace does not exist: {primary}")
    second_text = os.environ.get("MOE_IREE_SECOND_TRACE")
    if not second_text:
        raise FileNotFoundError(
            "MOE_IREE_SECOND_TRACE is required: set it to a distinct recorded Layer-7 trace. "
            "The gate intentionally does not verify only the primary 51-active/max-47 trace."
        )
    second = Path(second_text)
    if not second.is_file():
        raise FileNotFoundError(f"Second trace does not exist: {second}")
    if primary.resolve() == second.resolve():
        raise ValueError("MOE_IREE_SECOND_TRACE must be different from the primary trace")
    return primary, second


def compile_vmfb(source: Path, input_type: str, output: Path) -> dict:
    command = [
        IREE_COMPILE, str(source), f"--iree-input-type={input_type}",
        "--iree-hal-target-backends=cuda", f"--iree-cuda-target={TARGET_ARCH}",
        "-o", str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "source": str(source), "command": command, "returncode": result.returncode,
        "stdout": clipped(result.stdout), "stderr": clipped(result.stderr),
        "vmfb": str(output), "vmfb_exists": output.is_file(),
        "vmfb_sha256": sha256(output) if output.is_file() else None,
    }


def check_close(actual: np.ndarray, reference: np.ndarray) -> dict:
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    difference = np.abs(actual - reference)
    return {
        "shape_matches": list(actual.shape) == list(reference.shape),
        "all_finite": bool(np.all(np.isfinite(actual))),
        "max_abs_error": float(np.max(difference)),
        "mean_abs_error": float(np.mean(difference)),
        "allclose_rtol_atol": bool(np.allclose(actual, reference, rtol=RTOL, atol=ATOL)),
    }


def allclose_check(check: dict) -> bool:
    return check["shape_matches"] and check["all_finite"] and check["allclose_rtol_atol"]


def load_fixture(torch, path: Path) -> dict:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise TypeError(f"Trace must be a tensor dictionary: {path}")
    # The durable 17B trace artifacts are bundles. Their 64-token prefill
    # case lives under ``prefill64``; accept a flat dictionary too so a
    # future capture can use either representation. Decode-only bundles are
    # deliberately unsupported because this graph contract is B=64, K=8.
    trace = artifact.get("prefill64", artifact)
    if not isinstance(trace, dict):
        raise TypeError(f"prefill64 trace must be a tensor dictionary: {path}")
    x = as_numpy(require_mapping_item(trace, "moe_input", ("moe_input",)))
    selected = as_numpy(require_mapping_item(trace, "expert_ids", ("expert_ids", "selected")))
    gates = as_numpy(require_mapping_item(trace, "router_scores", ("router_scores", "gates")))
    reference = as_numpy(require_mapping_item(trace, "moe_output", ("moe_output",)))
    if x.shape == (1, 64, 2048):
        x = x[0]
    if selected.shape == (1, 64, 8):
        selected = selected[0]
    if gates.shape == (1, 64, 8):
        gates = gates[0]
    if reference.shape == (1, 64, 2048):
        reference = reference[0]
    x = np.ascontiguousarray(x, dtype=np.float32)
    selected = np.ascontiguousarray(selected, dtype=np.int32)
    gates = np.ascontiguousarray(gates, dtype=np.float32)
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    if x.shape != (64, 2048) or selected.shape != (64, 8) or gates.shape != (64, 8):
        raise ValueError(f"Unexpected trace shapes in {path}: x={x.shape}, ids={selected.shape}, gates={gates.shape}")
    if reference.shape != (64, 2048):
        raise ValueError(f"Unexpected reference shape in {path}: {reference.shape}")
    return {"path": path, "x": x, "selected": selected, "gates": gates, "reference": reference}


def host_route_reference(x: np.ndarray, selected: np.ndarray, gates: np.ndarray) -> dict:
    """Independent NumPy implementation of the stable, token-major route/pack contract."""
    flat_ids = selected.reshape(-1)
    flat_scores = gates.reshape(-1)
    token_ids = np.repeat(np.arange(64, dtype=np.int32), 8)
    rows = np.repeat(x[:, None, :], 8, axis=1).reshape(512, 2048)
    if np.any(flat_ids < 0) or np.any(flat_ids >= 64):
        raise ValueError("Expert IDs must be in [0, 64)")
    counts = np.bincount(flat_ids, minlength=64).astype(np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int32))).astype(np.int32)
    ranks = np.empty(512, dtype=np.int32)
    seen = np.zeros(64, dtype=np.int32)
    for i, expert_id in enumerate(flat_ids):
        ranks[i] = seen[expert_id]
        seen[expert_id] += 1
    destination = offsets[flat_ids] + ranks
    if not np.array_equal(np.sort(destination), np.arange(512, dtype=np.int32)):
        raise AssertionError("Host route destination is not a permutation of [0,512)")
    packed = np.empty((512, 2048), dtype=np.float32)
    packed_tokens = np.empty(512, dtype=np.int32)
    packed_scores = np.empty(512, dtype=np.float32)
    packed[destination] = rows
    packed_tokens[destination] = token_ids
    packed_scores[destination] = flat_scores
    return {
        "packed": packed, "counts": counts, "offsets": offsets, "destination": destination,
        "packed_tokens": packed_tokens, "packed_scores": packed_scores,
    }


def expert_reference(torch, packed: np.ndarray, counts: np.ndarray, offsets: np.ndarray,
                     gate_up_raw: np.ndarray, down_raw: np.ndarray) -> np.ndarray:
    """Independent per-expert FP32 Layer-7 MLP reference in packed expert order."""
    import torch.nn.functional as F

    values = np.zeros((512, 2048), dtype=np.float32)
    with torch.no_grad():
        for expert_id, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            start = int(offsets[expert_id])
            end = start + int(count)
            inp = torch.from_numpy(np.ascontiguousarray(packed[start:end]))
            gate_up = F.linear(inp, torch.from_numpy(gate_up_raw[expert_id]))
            gate, up = gate_up.chunk(2, dim=-1)
            ffn = F.silu(gate) * up
            out = F.linear(ffn, torch.from_numpy(down_raw[expert_id]))
            values[start:end] = out.cpu().numpy()
    return values


def combine_reference(values: np.ndarray, packed_tokens: np.ndarray, packed_scores: np.ndarray) -> np.ndarray:
    output = np.zeros((64, 2048), dtype=np.float32)
    np.add.at(output, packed_tokens, values * packed_scores[:, None])
    return output


def capacity_check(counts: np.ndarray) -> dict:
    max_count = int(counts.max())
    return {"capacity": CAPACITY, "max_expert_count": max_count, "accepted": max_count <= CAPACITY}


def invoke_arm(arm_fn, combine_fn, packed_d, counts_d, offsets_d, gate_up_d, down_d,
               sentinel_d, packed_tokens_d, packed_scores_d) -> tuple[np.ndarray, np.ndarray]:
    values_d = arm_fn(packed_d, counts_d, offsets_d, gate_up_d, down_d, sentinel_d)
    output_d = combine_fn(values_d, packed_tokens_d, packed_scores_d)
    return (np.asarray(values_d.to_host(), dtype=np.float32),
            np.asarray(output_d.to_host(), dtype=np.float32))


def run_trace(torch, ireert, cfg, functions: dict, weights_d: dict, fixture: dict,
              name: str, active_arm: str) -> dict:
    host_route = host_route_reference(fixture["x"], fixture["selected"], fixture["gates"])
    cap = capacity_check(host_route["counts"])
    if not cap["accepted"]:
        raise ValueError(f"{name} exceeds trace-tuned M={CAPACITY}: max count {cap['max_expert_count']}")
    host_values = expert_reference(
        torch, host_route["packed"], host_route["counts"], host_route["offsets"],
        weights_d["gate_up_host"], weights_d["down_host"],
    )
    host_output = combine_reference(host_values, host_route["packed_tokens"], host_route["packed_scores"])

    x_d = ireert.asdevicearray(cfg.device, fixture["x"])
    selected_d = ireert.asdevicearray(cfg.device, fixture["selected"])
    gates_d = ireert.asdevicearray(cfg.device, fixture["gates"])
    route_outputs = functions["route"](x_d, selected_d, gates_d)
    if not isinstance(route_outputs, tuple) or len(route_outputs) != 6:
        raise ValueError(f"route_pack returned {type(route_outputs)} with unexpected arity")
    packed_d, counts_d, offsets_d, destination_d, packed_tokens_d, packed_scores_d = route_outputs
    actual_route = {
        "packed": np.asarray(packed_d.to_host(), dtype=np.float32),
        "counts": np.asarray(counts_d.to_host(), dtype=np.int32),
        "offsets": np.asarray(offsets_d.to_host(), dtype=np.int32),
        "destination": np.asarray(destination_d.to_host(), dtype=np.int32),
        "packed_tokens": np.asarray(packed_tokens_d.to_host(), dtype=np.int32),
        "packed_scores": np.asarray(packed_scores_d.to_host(), dtype=np.float32),
    }
    route_checks = {
        "packed_rows": check_close(actual_route["packed"], host_route["packed"]),
        "counts_exact": bool(np.array_equal(actual_route["counts"], host_route["counts"])),
        "offsets_exact": bool(np.array_equal(actual_route["offsets"], host_route["offsets"])),
        "destination_exact": bool(np.array_equal(actual_route["destination"], host_route["destination"])),
        "packed_token_ids_exact": bool(np.array_equal(actual_route["packed_tokens"], host_route["packed_tokens"])),
        "packed_scores": check_close(actual_route["packed_scores"], host_route["packed_scores"]),
    }

    sentinel_values = {"zero": 0.0, "nan": np.nan, "posinf": np.inf}
    arm_checks: dict[str, dict] = {}
    for arm in (active_arm, "stationary64"):
        arm_checks[arm] = {}
        zero_values = zero_output = None
        for label, sentinel in sentinel_values.items():
            sentinel_d = ireert.asdevicearray(cfg.device, np.asarray(sentinel, dtype=np.float32))
            values, output = invoke_arm(
                functions[arm], functions["combine"], packed_d, counts_d, offsets_d,
                weights_d[arm]["gate_up"], weights_d[arm]["down"], sentinel_d,
                packed_tokens_d, packed_scores_d,
            )
            checks = {
                "expert_values_vs_host": check_close(values, host_values),
                "final_vs_trace_reference": check_close(output, fixture["reference"]),
                "final_vs_independent_host_reference": check_close(output, host_output),
            }
            if label == "zero":
                zero_values, zero_output = values, output
            else:
                checks["expert_values_vs_zero_sentinel"] = check_close(values, zero_values)
                checks["final_vs_zero_sentinel"] = check_close(output, zero_output)
            arm_checks[arm][label] = checks

        # A correctness repeat, not a timed repeat: GPU outputs need only match within tolerance.
        sentinel_d = ireert.asdevicearray(cfg.device, np.asarray(0.0, dtype=np.float32))
        repeated_values, repeated_output = invoke_arm(
            functions[arm], functions["combine"], packed_d, counts_d, offsets_d,
            weights_d[arm]["gate_up"], weights_d[arm]["down"], sentinel_d,
            packed_tokens_d, packed_scores_d,
        )
        arm_checks[arm]["repeat_zero"] = {
            "expert_values_vs_first_zero": check_close(repeated_values, zero_values),
            "final_vs_first_zero": check_close(repeated_output, zero_output),
        }

    inactive = int(np.count_nonzero(actual_route["counts"] == 0))
    active = int(np.count_nonzero(actual_route["counts"]))
    return {
        "trace_path": str(fixture["path"]), "trace_input_sha256": array_sha256(fixture["x"]),
        "capacity_precondition": cap, "route_pack": route_checks,
        "routing": {"assignments": int(actual_route["counts"].sum()), "active_expert_count": active,
                    "inactive_expert_count": inactive, "max_expert_count": int(actual_route["counts"].max())},
        "active_set_module": active_arm,
        "arms": arm_checks,
        "reference": "saved Layer-7 moe_output with recorded, unnormalized top-8 router gates",
        "stationary64_inactive_slots": {
            "zero_count_slots": inactive,
            "poison_has_no_final_effect": all(
                allclose_check(arm_checks["stationary64"][label]["final_vs_zero_sentinel"])
                for label in ("nan", "posinf")
            ),
        },
    }


def trace_passed(trace: dict, contract: dict) -> bool:
    route = trace["route_pack"]
    route_ok = (allclose_check(route["packed_rows"]) and route["counts_exact"] and route["offsets_exact"]
                and route["destination_exact"] and route["packed_token_ids_exact"] and allclose_check(route["packed_scores"]))
    routing = trace["routing"]
    routing_ok = (trace["capacity_precondition"]["accepted"]
                  and trace["active_set_module"] == contract["active_arm"]
                  and all(routing[key] == expected for key, expected in contract["expected"].items()))
    arms_ok = True
    for arm in (contract["active_arm"], "stationary64"):
        for label in ("zero", "nan", "posinf"):
            checks = trace["arms"][arm][label]
            arms_ok = arms_ok and allclose_check(checks["expert_values_vs_host"])
            arms_ok = arms_ok and allclose_check(checks["final_vs_trace_reference"])
            arms_ok = arms_ok and allclose_check(checks["final_vs_independent_host_reference"])
            if label != "zero":
                arms_ok = arms_ok and allclose_check(checks["expert_values_vs_zero_sentinel"])
                arms_ok = arms_ok and allclose_check(checks["final_vs_zero_sentinel"])
        repeat = trace["arms"][arm]["repeat_zero"]
        arms_ok = arms_ok and allclose_check(repeat["expert_values_vs_first_zero"])
        arms_ok = arms_ok and allclose_check(repeat["final_vs_first_zero"])
    return route_ok and routing_ok and arms_ok and trace["stationary64_inactive_slots"]["poison_has_no_final_effect"]


def main() -> None:
    report = {
        "phase": "25C1 poison-safe functional graph correctness gate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": ("correctness only: compile/load once, then evaluate two recorded traces using zero/NaN/+Inf "
                  "padding sentinels and one tolerance-level repeat. No timer, warmup, duration collection, "
                  "or performance conclusion. Router projection, softmax, and top-k are outside this boundary."),
        "target_backend": "cuda", "target_arch": TARGET_ARCH, "tolerances": {"rtol": RTOL, "atol": ATOL},
        "capacity": {"rows_per_expert": CAPACITY, "meaning": "trace-tuned harness precondition, not deployable overflow handling"},
        "gates": {"lowering_report_present": LOWERING.is_file(), "structure_report_present": STRUCTURE.is_file(),
                  "iree_compile_available": IREE_COMPILE is not None, "no_timing_runner_in_script": True,
                  "second_trace_path_declared": bool(os.environ.get("MOE_IREE_SECOND_TRACE"))},
        "status": "STOP_25C1_CORRECTNESS_PRECONDITION_FAILED",
        "next_step": "Resolve the stated correctness precondition; do not start a timing run.",
    }
    if not all(report["gates"].values()):
        write_and_print(report)
        return
    try:
        lowering = json.loads(LOWERING.read_text(encoding="utf-8"))
        structure = json.loads(STRUCTURE.read_text(encoding="utf-8"))
        report["gates"]["lowering_exports_reviewed"] = lowering.get("gates", {}).get("all_flow_and_stream_exports_succeeded") is True
        report["gates"]["poison_safe_lowering_contract_reviewed"] = lowering.get("gates", {}).get("all_arms_have_poison_safe_padding_contract") is True
        report["gates"]["accepted_25c0_source_hashes_present"] = bool(structure.get("source_hashes"))
        source_specs = {
            "route": (ROOT / "10-olmoe_route_pack_b64_e64_k8.mlir", "stablehlo", "moe_route"),
            "active51": (SOURCES / "25C_active51_device_gather.mlir", "auto", "olmoe_25c_active51_grouped"),
            "active55": (SOURCES / "25C_active55_device_gather.mlir", "auto", "olmoe_25c_active55_grouped"),
            "stationary64": (SOURCES / "25C_stationary64_full_bank.mlir", "auto", "olmoe_25c_stationary64_grouped"),
            "combine": (ROOT / "11-olmoe_combine_b64_k8.mlir", "stablehlo", "moe_combine"),
        }
        report["source_hashes"] = {}
        for name, (path, _, _) in source_specs.items():
            report["gates"][f"{name}_source_present"] = path.is_file()
            if path.is_file():
                report["source_hashes"][name] = sha256(path)
        for arm, expected in EXPECTED_GRAPH_HASHES.items():
            report["gates"][f"{arm}_source_hash_matches_poison_safe_revision"] = report["source_hashes"].get(arm) == expected
        if not all(report["gates"].values()):
            write_and_print(report)
            return

        import torch
        from iree import runtime as ireert

        primary_path, second_path = resolve_traces()
        primary, second = load_fixture(torch, primary_path), load_fixture(torch, second_path)
        primary_key = (array_sha256(primary["x"]), array_sha256(primary["selected"]), array_sha256(primary["gates"]))
        second_key = (array_sha256(second["x"]), array_sha256(second["selected"]), array_sha256(second["gates"]))
        if primary_key == second_key:
            raise ValueError("The second trace has identical input/routing content to the primary trace")
        weights_path = one_candidate("Layer-7 FP32 weights", ("*layer7*weight*.pt", "*layer7*weights*.pt"))
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(weights, dict):
            raise TypeError("Layer-7 weights must be a tensor dictionary")
        gate_up_raw = np.ascontiguousarray(as_numpy(require_mapping_item(weights, "gate_up", ("gate_up_proj", "gate_up"))), dtype=np.float32)
        down_raw = np.ascontiguousarray(as_numpy(require_mapping_item(weights, "down", ("down_proj", "down"))), dtype=np.float32)
        if gate_up_raw.shape != (64, 2048, 2048) or down_raw.shape != (64, 2048, 1024):
            raise ValueError(f"Unexpected raw weight shapes: gate_up={gate_up_raw.shape}, down={down_raw.shape}")
        gate_up_bank = np.ascontiguousarray(np.swapaxes(gate_up_raw, 1, 2), dtype=np.float32)
        down_bank = np.ascontiguousarray(np.swapaxes(down_raw, 1, 2), dtype=np.float32)
        report["input_contract"] = {
            "primary_trace": str(primary_path), "second_trace": str(second_path), "weights_path": str(weights_path),
            "weight_layout_transform": "transpose each per-expert Linear weight [out,in] -> graph bank [in,out]",
            "reference": "saved Layer-7 moe_output with recorded, unnormalized top-8 router gates",
            "sentinel_fixtures": ["zero", "nan", "posinf"],
            "trace_specialization": TRACE_CONTRACTS,
        }

        OUT.mkdir(parents=True, exist_ok=True)
        report["compilation"] = {}
        all_compiled = True
        for name, (source, input_type, _) in source_specs.items():
            result = compile_vmfb(source, input_type, OUT / f"25C1_{name}.vmfb")
            report["compilation"][name] = result
            all_compiled = all_compiled and result["returncode"] == 0 and result["vmfb_exists"]
        report["gates"]["all_cuda_vmfb_compilations_succeeded"] = all_compiled
        if not all_compiled:
            report["status"] = "STOP_25C1_CORRECTNESS_COMPILATION_FAILED"
            report["next_step"] = "Resolve only the recorded compiler issue; do not time either arm."
            write_and_print(report)
            return

        cfg = ireert.Config("cuda")
        ctx = ireert.SystemContext(config=cfg)
        for name, (_, _, module_name) in source_specs.items():
            vmfb = Path(report["compilation"][name]["vmfb"])
            ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb.read_bytes()))
        functions = {
            "route": getattr(ctx.modules, source_specs["route"][2])["route_pack"],
            "active51": getattr(ctx.modules, source_specs["active51"][2])["forward"],
            "active55": getattr(ctx.modules, source_specs["active55"][2])["forward"],
            "stationary64": getattr(ctx.modules, source_specs["stationary64"][2])["forward"],
            "combine": getattr(ctx.modules, source_specs["combine"][2])["combine"],
        }
        weights_d = {
            "gate_up_host": gate_up_raw, "down_host": down_raw,
            "active51": {"gate_up": ireert.asdevicearray(cfg.device, gate_up_bank), "down": ireert.asdevicearray(cfg.device, down_bank)},
            "active55": {"gate_up": ireert.asdevicearray(cfg.device, gate_up_bank), "down": ireert.asdevicearray(cfg.device, down_bank)},
            "stationary64": {"gate_up": ireert.asdevicearray(cfg.device, gate_up_bank), "down": ireert.asdevicearray(cfg.device, down_bank)},
        }
        report["residency"] = {
            "weight_banks": "each arm receives its full 64-expert bank once; active-set selection is inside its trace-specialized graph",
            "sentinel": "a scalar is uploaded per zero/NaN/+Inf correctness fixture to the same loaded VMFBs",
            "selected_weight_host_prestacking": False,
            "final_and_intermediate_host_reads": "only for correctness comparisons; no timing boundary exists in this script",
        }
        overflow_selected = np.zeros((64, 8), dtype=np.int32)
        overflow_counts = host_route_reference(primary["x"], overflow_selected, primary["gates"])["counts"]
        overflow = capacity_check(overflow_counts)
        report["overflow_rejection"] = {
            **overflow, "synthetic_assignments": 512, "rejected_before_graph_execution": not overflow["accepted"],
            "deployable_design_note": "A production graph needs an in-graph overflow flag and larger-bucket fallback; that is outside this experiment and its future timing boundary.",
        }
        report["traces"] = {
            "primary": run_trace(torch, ireert, cfg, functions, weights_d, primary, "primary",
                                 TRACE_CONTRACTS["primary"]["active_arm"]),
            "second": run_trace(torch, ireert, cfg, functions, weights_d, second, "second",
                                TRACE_CONTRACTS["second"]["active_arm"]),
        }
        report["checks"] = {
            "primary_passed": trace_passed(report["traces"]["primary"], TRACE_CONTRACTS["primary"]),
            "second_passed": trace_passed(report["traces"]["second"], TRACE_CONTRACTS["second"]),
            "synthetic_overflow_rejected": report["overflow_rejection"]["rejected_before_graph_execution"],
        }
        passed = all(report["checks"].values())
        report["gates"]["adversarial_functional_correctness_passed"] = passed
        if passed:
            report["status"] = "REVIEW_REQUIRED_25C1_ADVERSARIAL_FUNCTIONAL_CORRECTNESS"
            report["next_step"] = ("Review each trace separately, including intermediate route/expert checks and NaN/+Inf poison checks. "
                                   "Only after acceptance may the 25C2 timing protocol be frozen; no timings have been collected.")
        else:
            report["status"] = "STOP_25C1_ADVERSARIAL_FUNCTIONAL_CORRECTNESS_FAILED"
            report["next_step"] = "Resolve only the stated correctness mismatch; timing remains unauthorized."
    except Exception as error:
        report["exception"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=10)}
        report["status"] = "STOP_25C1_ADVERSARIAL_FUNCTIONAL_CORRECTNESS_EXCEPTION"
        report["next_step"] = "Resolve only the recorded correctness setup/source issue; do not time either arm."
    write_and_print(report)


if __name__ == "__main__":
    main()
