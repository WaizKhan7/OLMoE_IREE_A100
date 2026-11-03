"""Capture a distinct, correctness-only 64-token Layer-7 prefill trace.

This script records the same nested ``{"prefill64": ...}`` tensor-bundle
contract as the durable original trace. It performs no IREE work and no
benchmarking. The saved trace is accepted only when it is genuinely different
from the primary trace and does not exceed the current experiment's trace-tuned
capacity of 47 rows for a single expert.

Run on an A100 GPU. Set MOE_OLMOE_MODEL to an existing local model directory
if desired; otherwise Hugging Face resolves the pinned model revision.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(os.environ.get("MOE_IREE_ROOT", "/content/drive/MyDrive/moe_iree_olmoe"))
OUT = ROOT / "olmoe_layer7_prefill64_second_trace.pt"
REPORT = ROOT / "olmoe_layer7_prefill64_second_trace_capture.json"
PRIMARY = ROOT / "olmoe_layer7_prefill64_decode1.pt"
MODEL = os.environ.get("MOE_OLMOE_MODEL", "allenai/OLMoE-1B-7B-0125")
REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
LAYER = 7
BATCH = 1
TOKENS = 64
EXPERTS = 64
TOP_K = 8
CAPACITY = 47

# Deliberately different prose from the original capture. The script takes the
# first 64 model tokens after tokenization, so this remains deterministic.
TEXT = os.environ.get(
    "MOE_IREE_SECOND_TRACE_TEXT",
    """A compiler has to respect the structure of a model, not just its arithmetic.
Sparse mixture-of-experts layers route every token to a few specialists, and a
useful experiment checks packing, routing, expert execution, and recombination
separately. This alternate prompt exists only to capture a second fixed,
64-token Layer-7 prefill trace for a correctness test. It is not a benchmark
prompt, a performance measurement, or a training sample. Reproducible evidence
comes from recording the input, router choices, gates, and output together.""",
)


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this capture")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    encoded = tokenizer(TEXT, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"][:, :TOKENS]
    attention_mask = encoded["attention_mask"][:, :TOKENS]
    if tuple(input_ids.shape) != (BATCH, TOKENS):
        raise ValueError(
            f"Prompt tokenized to only {input_ids.shape[1]} tokens; provide a longer "
            "MOE_IREE_SECOND_TRACE_TEXT."
        )

    # The original Layer-7 trace and exported weights are FP32. Keeping this
    # capture FP32 avoids changing the correctness reference merely to fit a
    # lower-precision model load.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    mlp = model.model.layers[LAYER].mlp
    router = mlp.gate
    captured: dict[str, torch.Tensor] = {}

    def mlp_pre_hook(_module, args):
        captured["moe_input"] = args[0].detach().float().cpu().contiguous()

    def mlp_post_hook(_module, _args, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        captured["moe_output"] = value.detach().float().cpu().contiguous()

    def router_post_hook(_module, _args, output):
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise TypeError(f"Unexpected OLMoE router output: {type(output)}")
        logits, scores, expert_ids = output
        captured["router_logits"] = logits.detach().float().cpu().contiguous()
        captured["router_scores"] = scores.detach().float().cpu().contiguous()
        captured["expert_ids"] = expert_ids.detach().cpu().contiguous()

    handles = [
        mlp.register_forward_pre_hook(mlp_pre_hook),
        mlp.register_forward_hook(mlp_post_hook),
        router.register_forward_hook(router_post_hook),
    ]
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
                output_router_logits=False,
            )
    finally:
        for handle in handles:
            handle.remove()
        del model
        torch.cuda.empty_cache()

    required = {"moe_input", "router_logits", "router_scores", "expert_ids", "moe_output"}
    missing = required - set(captured)
    if missing:
        raise RuntimeError(f"Layer-7 hooks did not capture: {sorted(missing)}")
    bundle = {
        "moe_input": captured["moe_input"],
        "router_logits": captured["router_logits"],
        "router_scores": captured["router_scores"],
        "expert_ids": captured["expert_ids"],
        "moe_output": captured["moe_output"],
        "input_ids": input_ids.cpu().contiguous(),
        "attention_mask": attention_mask.cpu().contiguous(),
    }
    shapes = {name: list(value.shape) for name, value in bundle.items()}
    if (shapes["moe_input"], shapes["router_logits"], shapes["router_scores"],
            shapes["expert_ids"], shapes["moe_output"]) != (
                [1, 64, 2048], [64, 64], [64, 8], [64, 8], [1, 64, 2048]):
        raise ValueError(f"Unexpected Layer-7 trace shapes: {shapes}")
    ids = bundle["expert_ids"].to(torch.int64)
    if torch.any(ids < 0) or torch.any(ids >= EXPERTS):
        raise ValueError("Router emitted expert ID outside [0, 64)")
    counts = torch.bincount(ids.reshape(-1), minlength=EXPERTS)
    max_count = int(counts.max())
    if max_count > CAPACITY:
        raise ValueError(
            f"Second trace violates trace-tuned M={CAPACITY}: max expert count is {max_count}. "
            "Change MOE_IREE_SECOND_TRACE_TEXT and capture another trace."
        )
    if PRIMARY.is_file():
        primary = torch.load(PRIMARY, map_location="cpu", weights_only=True)
        primary_case = primary.get("prefill64", primary) if isinstance(primary, dict) else None
        if isinstance(primary_case, dict):
            primary_key = tuple(tensor_hash(primary_case[name]) for name in ("moe_input", "expert_ids", "router_scores"))
            second_key = tuple(tensor_hash(bundle[name]) for name in ("moe_input", "expert_ids", "router_scores"))
            if primary_key == second_key:
                raise ValueError("Captured trace duplicates the primary trace; choose different text")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prefill64": bundle}, OUT)
    report = {
        "status": "REVIEW_REQUIRED_25C1_SECOND_TRACE_CAPTURED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "capture only; no IREE execution or timing",
        "output": str(OUT),
        "output_sha256": file_hash(OUT),
        "model": MODEL,
        "revision": REVISION,
        "layer": LAYER,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "shapes": shapes,
        "routing": {
            "assignments": int(ids.numel()),
            "active_experts": int((counts > 0).sum()),
            "inactive_experts": int((counts == 0).sum()),
            "max_expert_count": max_count,
            "capacity": CAPACITY,
        },
        "next_step": "Use this file as MOE_IREE_SECOND_TRACE in the 25C1 correctness gate; timing remains unauthorized.",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
