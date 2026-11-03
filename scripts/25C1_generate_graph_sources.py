"""Draft the two 25C graph-boundary expert sources; no compiler/runtime work.

The generated functions accept the existing device-resident route/pack outputs
and return a device-resident [512, 2048] expert-value buffer for the existing
combine module. Active-51 derives the 51 trace-active IDs on device from
counts, then explicitly gathers selected weights from the resident 64-expert
bank. Stationary-64 uses the same padded-pack/unpack pattern but reads the full
bank directly. This script writes source only: it does not compile, import
IREE/Torch, create a runtime, allocate CUDA, invoke, synchronize, or time.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("MOE_IREE_ROOT", "/content/drive/MyDrive/moe_iree_olmoe"))
STRUCTURE = ROOT / "25C0_selected_structure" / "25C0_selected_structure_report.json"
OUT = ROOT / "25C1_graph_sources"
MANIFEST = OUT / "25C1_graph_source_manifest.json"
EXPECTED = {
    "dynamic_rank2": "53ce8bd7e933748b9620f4fb67f274be993785d7291f1ca17f1e63f0a072d05a",
    "static_rank3_e50_final": "9ff7da1a60694032f74d36980e6f655a919f2ee560bad79f40d3f456502db190",
    "route_pack_b64_e64_k8": "dc78269f1a38d55b0603204d2f2b1808e7247dae4ff5ae1848c5b2f5a6ddc122",
    "combine_b64_k8": "0b5906252e92c3bcaea99d9494879757fe367d8de804aff3daa588f2714fecce",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_block(arm: str, e: int) -> str:
    if arm == "active51":
        return """    // Exactly 51 experts have nonzero trace counts. Sorting makes the
    // active-ID list device-derived; a later correctness gate verifies this contract.
    %all_ids = stablehlo.iota dim = 0 : tensor<64xi32>
    %sorted_counts, %sorted_ids = "stablehlo.sort"(%counts, %all_ids) ({
      ^bb0(%count_l: tensor<i32>, %id_l: tensor<i32>, %count_r: tensor<i32>, %id_r: tensor<i32>):
        %gt = stablehlo.compare GT, %count_l, %count_r : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %gt : tensor<i1>
    }) {dimension = 0 : i64, is_stable = true} : (tensor<64xi32>, tensor<64xi32>) -> (tensor<64xi32>, tensor<64xi32>)
    %expert_ids = stablehlo.slice %sorted_ids [0:51] : (tensor<64xi32>) -> tensor<51xi32>
"""
    return """    // Stationary bank: all 64 expert slots are always present; inactive
    // slots will be zeroed by the count-derived activation mask below.
    %expert_ids = stablehlo.iota dim = 0 : tensor<64xi32>
"""


def weight_block(arm: str, e: int) -> str:
    if arm == "active51":
        return """    // Explicit device-side selection from the resident 64-expert banks.
    // This is intentionally a gather in source; lowering review decides whether it
    // remains a distinct materialization/copy or is transformed by the compiler.
    %selected_gate_up = "stablehlo.gather"(%gate_up_bank, %id_indices) {
      dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0],
        start_index_map = [0], index_vector_dim = 1>,
      slice_sizes = array<i64: 1, 2048, 2048>, indices_are_sorted = false
    } : (tensor<64x2048x2048xf32>, tensor<51x1xi32>) -> tensor<51x2048x2048xf32>
    %selected_down = "stablehlo.gather"(%down_bank, %id_indices) {
      dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0],
        start_index_map = [0], index_vector_dim = 1>,
      slice_sizes = array<i64: 1, 1024, 2048>, indices_are_sorted = false
    } : (tensor<64x1024x2048xf32>, tensor<51x1xi32>) -> tensor<51x1024x2048xf32>
"""
    return """    // No expert-weight selection: the resident 64-expert banks are passed
    // directly to the same grouped contractions.
"""


def source(arm: str, e: int) -> str:
    rows = e * 47
    name = f"olmoe_25c_{arm}_grouped"
    gate_up_operand = "%selected_gate_up" if arm == "active51" else "%gate_up_bank"
    down_operand = "%selected_down" if arm == "active51" else "%down_bank"
    return f'''// 25C_ARM: {arm}
// 25C_CONTRACT: B64_K8_H2048_I1024_FP32
// 25C_SOURCE_HASHES: dynamic=53ce8bd7e933748b9620f4fb67f274be993785d7291f1ca17f1e63f0a072d05a static_e50=9ff7da1a60694032f74d36980e6f655a919f2ee560bad79f40d3f456502db190
// Output is expert-grouped [512,2048] and feeds the existing B64/K8 combine module.
module @{name} {{
  func.func @forward(
      %packed: tensor<512x2048xf32>,
      %counts: tensor<64xi32>,
      %offsets: tensor<64xi32>,
      %gate_up_bank: tensor<64x2048x2048xf32>,
      %down_bank: tensor<64x1024x2048xf32>) -> tensor<512x2048xf32> {{
{ids_block(arm, e)}    %id_indices = stablehlo.reshape %expert_ids : (tensor<{e}xi32>) -> tensor<{e}x1xi32>
    %selected_counts = "stablehlo.gather"(%counts, %id_indices) {{
      dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [0],
        start_index_map = [0], index_vector_dim = 1>,
      slice_sizes = array<i64: 1>, indices_are_sorted = false
    }} : (tensor<64xi32>, tensor<{e}x1xi32>) -> tensor<{e}xi32>
    %selected_offsets = "stablehlo.gather"(%offsets, %id_indices) {{
      dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [0],
        start_index_map = [0], index_vector_dim = 1>,
      slice_sizes = array<i64: 1>, indices_are_sorted = false
    }} : (tensor<64xi32>, tensor<{e}x1xi32>) -> tensor<{e}xi32>

    // Append 46 zero rows so every fixed 47-row gather is in bounds, even
    // for the final active expert. Counts mask padded rows before compute.
    %tail = stablehlo.constant dense<0.0> : tensor<46x2048xf32>
    %packed_padded = stablehlo.concatenate %packed, %tail, dim = 0
      : (tensor<512x2048xf32>, tensor<46x2048xf32>) -> tensor<558x2048xf32>
    %row = stablehlo.iota dim = 1 : tensor<{e}x47xi32>
    %offsets_2d = stablehlo.broadcast_in_dim %selected_offsets, dims = [0]
      : (tensor<{e}xi32>) -> tensor<{e}x47xi32>
    %packed_row_ids = stablehlo.add %offsets_2d, %row : tensor<{e}x47xi32>
    %gather_ids_flat = stablehlo.reshape %packed_row_ids : (tensor<{e}x47xi32>) -> tensor<{rows}xi32>
    %gather_ids = stablehlo.reshape %gather_ids_flat : (tensor<{rows}xi32>) -> tensor<{rows}x1xi32>
    %input_rows = "stablehlo.gather"(%packed_padded, %gather_ids) {{
      dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0],
        start_index_map = [0], index_vector_dim = 1>,
      slice_sizes = array<i64: 1, 2048>, indices_are_sorted = false
    }} : (tensor<558x2048xf32>, tensor<{rows}x1xi32>) -> tensor<{rows}x2048xf32>
    %input_grouped = stablehlo.reshape %input_rows
      : (tensor<{rows}x2048xf32>) -> tensor<{e}x47x2048xf32>
    %counts_2d = stablehlo.broadcast_in_dim %selected_counts, dims = [0]
      : (tensor<{e}xi32>) -> tensor<{e}x47xi32>
    %valid = stablehlo.compare LT, %row, %counts_2d
      : (tensor<{e}x47xi32>, tensor<{e}x47xi32>) -> tensor<{e}x47xi1>
    %valid_f = stablehlo.convert %valid : (tensor<{e}x47xi1>) -> tensor<{e}x47xf32>
    %valid_f3 = stablehlo.broadcast_in_dim %valid_f, dims = [0, 1]
      : (tensor<{e}x47xf32>) -> tensor<{e}x47x2048xf32>
    %masked_input = stablehlo.multiply %input_grouped, %valid_f3
      : tensor<{e}x47x2048xf32>

{weight_block(arm, e)}    // batch_matmul is an accumulation op: its output must be initialized.
    %zero_f32 = arith.constant 0.0 : f32
    %gate_up_empty = tensor.empty() : tensor<{e}x47x2048xf32>
    %gate_up_init = linalg.fill ins(%zero_f32 : f32)
      outs(%gate_up_empty : tensor<{e}x47x2048xf32>) -> tensor<{e}x47x2048xf32>
    %gate_up_out = linalg.batch_matmul
      ins(%masked_input, {gate_up_operand} : tensor<{e}x47x2048xf32>, tensor<{e}x2048x2048xf32>)
      outs(%gate_up_init : tensor<{e}x47x2048xf32>) -> tensor<{e}x47x2048xf32>
    %gate = tensor.extract_slice %gate_up_out[0, 0, 0] [ {e}, 47, 1024 ] [1, 1, 1]
      : tensor<{e}x47x2048xf32> to tensor<{e}x47x1024xf32>
    %up = tensor.extract_slice %gate_up_out[0, 0, 1024] [ {e}, 47, 1024 ] [1, 1, 1]
      : tensor<{e}x47x2048xf32> to tensor<{e}x47x1024xf32>
    %ffn_init = tensor.empty() : tensor<{e}x47x1024xf32>
    %ffn = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                                affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                                affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                          iterator_types = ["parallel", "parallel", "parallel"]}}
      ins(%gate, %up : tensor<{e}x47x1024xf32>, tensor<{e}x47x1024xf32>)
      outs(%ffn_init : tensor<{e}x47x1024xf32>) {{
      ^bb0(%g: f32, %u: f32, %unused: f32):
        %neg_g = arith.negf %g : f32
        %exp_neg_g = math.exp %neg_g : f32
        %one = arith.constant 1.0 : f32
        %denom = arith.addf %one, %exp_neg_g : f32
        %silu = arith.divf %g, %denom : f32
        %value = arith.mulf %silu, %u : f32
        linalg.yield %value : f32
      }} -> tensor<{e}x47x1024xf32>
    %out_empty = tensor.empty() : tensor<{e}x47x2048xf32>
    %out_init = linalg.fill ins(%zero_f32 : f32)
      outs(%out_empty : tensor<{e}x47x2048xf32>) -> tensor<{e}x47x2048xf32>
    %grouped_out = linalg.batch_matmul
      ins(%ffn, {down_operand} : tensor<{e}x47x1024xf32>, tensor<{e}x1024x2048xf32>)
      outs(%out_init : tensor<{e}x47x2048xf32>) -> tensor<{e}x47x2048xf32>
    %masked_out = stablehlo.multiply %grouped_out, %valid_f3
      : tensor<{e}x47x2048xf32>

    // Padded rows may point beyond the 512 live assignments. Clamp their
    // destinations and scatter-add zeros, preserving all valid disjoint rows.
    %max_index_scalar = stablehlo.constant dense<511> : tensor<i32>
    %max_index = stablehlo.broadcast_in_dim %max_index_scalar, dims = []
      : (tensor<i32>) -> tensor<{e}x47xi32>
    %safe_row_ids = stablehlo.minimum %packed_row_ids, %max_index
      : tensor<{e}x47xi32>
    %scatter_ids_flat = stablehlo.reshape %safe_row_ids : (tensor<{e}x47xi32>) -> tensor<{rows}xi32>
    %scatter_ids = stablehlo.reshape %scatter_ids_flat : (tensor<{rows}xi32>) -> tensor<{rows}x1xi32>
    %updates = stablehlo.reshape %masked_out
      : (tensor<{e}x47x2048xf32>) -> tensor<{rows}x2048xf32>
    %zero_output = stablehlo.constant dense<0.0> : tensor<512x2048xf32>
    %result = "stablehlo.scatter"(%zero_output, %scatter_ids, %updates) ({{
      ^bb0(%old: tensor<f32>, %update: tensor<f32>):
        %sum = stablehlo.add %old, %update : tensor<f32>
        stablehlo.return %sum : tensor<f32>
    }}) {{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1],
        inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>,
        indices_are_sorted = false, unique_indices = false}}
      : (tensor<512x2048xf32>, tensor<{rows}x1xi32>, tensor<{rows}x2048xf32>) -> tensor<512x2048xf32>
    return %result : tensor<512x2048xf32>
  }}
}}
'''


def write_and_print(report: dict) -> None:
    MANIFEST.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "manifest": str(MANIFEST),
        "sources": report.get("sources"),
        "gates": report["gates"],
        "next_step": report["next_step"],
    }, indent=2))


def main() -> None:
    report = {
        "phase": "25C1 graph-source drafting",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "source generation only; no IREE/Torch import, compilation, runtime creation, CUDA "
            "allocation, invocation, synchronization, correctness test, or timing"
        ),
        "gates": {
            "selected_structure_report_present": STRUCTURE.is_file(),
            "no_runtime_or_timing_in_script": True,
        },
        "status": "STOP_25C1_SOURCE_DRAFT_PRECONDITION_FAILED",
        "next_step": "Restore the accepted 25C0 selected-structure report.",
    }
    if not STRUCTURE.is_file():
        OUT.mkdir(parents=True, exist_ok=True)
        write_and_print(report)
        return
    structure = json.loads(STRUCTURE.read_text(encoding="utf-8"))
    report["gates"]["accepted_25c0_hashes_match"] = structure.get("source_hashes") == EXPECTED
    if not all(report["gates"].values()):
        OUT.mkdir(parents=True, exist_ok=True)
        write_and_print(report)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "active51": OUT / "25C_active51_device_gather.mlir",
        "stationary64": OUT / "25C_stationary64_full_bank.mlir",
    }
    paths["active51"].write_text(source("active51", 51), encoding="utf-8")
    paths["stationary64"].write_text(source("stationary64", 64), encoding="utf-8")
    report.update({
        "source_revision": (
            "revised after compiler review: both batch_matmul accumulator tensors are explicitly "
            "zero-initialized with linalg.fill, and Stationary-64 passes its resident banks directly "
            "to grouped contractions instead of using an invalid SSA alias"
        ),
        "sources": {
            name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in paths.items()
        },
        "architectural_contract": {
            "active51": (
                "sorts device counts to derive 51 active IDs, gathers activation rows and both "
                "weight banks on device, then runs two 51x47 grouped contractions"
            ),
            "stationary64": (
                "uses IDs 0..63 and the full resident weight banks directly, with count masks "
                "zeroing inactive/padded rows before two 64x47 grouped contractions"
            ),
            "shared": (
                "both accept route/pack outputs on device and return expert-grouped [512,2048] "
                "for the existing combine module; router top-k remains outside scope"
            ),
        },
        "status": "REVIEW_REQUIRED_25C1_GRAPH_SOURCES_DRAFTED",
        "next_step": (
            "Review the two source contracts and source hashes. The next gate may parse/compile "
            "them and inspect Flow/Stream IR only; no runtime correctness or timing is authorized."
        ),
    })
    write_and_print(report)


if __name__ == "__main__":
    main()
