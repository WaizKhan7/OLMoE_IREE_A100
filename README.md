# Tuning OLMoE Prefill in IREE on an A100

**A faster expert core lost at the routed-layer boundary. A different, padded full-model graph became 3.18× faster after targeted GPU schedule changes.**

This repository documents my experiments with [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0125): source scripts, original result bundles, checks, lowering changes, and selected traces. The question was whether giving IREE regular, padded expert work could outperform an implementation that computes only the rows routed to each expert. The answer depended on *which part of the system I measured*.

At **256-token prefill** on one **A100-SXM4-40GB**, tuning the padded IREE model reduced completed-request latency from **1,657.67 ms to 520.62 ms**. Matched grouped PyTorch took **188.96 ms**. This is a measured improvement to the tested IREE implementation, not a claim that it beat PyTorch or that padding always wins.

## Main results

| Full-model path | S64/C64 | S256/C256 |
|---|---:|---:|
| Grouped PyTorch (`torch.compile` + CUDA graphs) | 97.41 ms | **188.96 ms** |
| Original padded IREE | 473.13 ms | 1,657.67 ms |
| **Tuned padded IREE** | **308.15 ms** | **520.62 ms** |

The frozen S256 result is **3.18× faster** and **68.6% lower latency** than original padded IREE. Tuned IREE remained **2.76× slower** than the matched PyTorch arm. The frozen comparison includes four prompts at two lengths and **960 timed requests**; entries are means of prompt medians.

A separate four-way **S256** run added an *untuned dynamic-row IREE* path. Its expert multiplications processed only actual routed rows, yet the completed request took **2,704.23 ms**, versus **520.97 ms** for tuned padded IREE and **1,658.24 ms** for original padded IREE. That arm used Python expert orchestration, different graph partitions and different lowering, so the comparison does **not** isolate the cost of padding or settle what a tuned dynamic design could do.

## The experiment story

### The core win reversed at Layer 7

Using real Layer-7 weights, the grouped prepared expert core took about **16.6 ms**, versus **37.7 ms** for the dynamic core. But that narrow test excluded routing, weight selection and output assembly. At the integrated routed-layer boundary, the active grouped graph took about **273 ms**, versus **87 ms** for the dynamic path. The trace showed about **217–234 ms** in the grouped graph's scatter-back dispatch, which put expert outputs back in token order. Its reported launch used one block and one thread.

Those core and layer timings cover different work and are not an additive breakdown. I did **not** tune that scatter dispatch and reuse the graph. The result showed why optimizing an expert calculation alone could make the wider layer slower.

### I tuned a different full-model graph

I built a 16-layer IREE VMFB with input-dependent routing inside the graph and fixed expert capacity for each prefill length. At S256, top-8 routing creates **2,048 real expert assignments per layer**, while C256 reserves **16,384 padded row slots** across 64 experts. The regular shapes exposed gate/up and down expert matrix multiplications whose generated CUDA dispatches I could inspect and tune.

First, I reduced the output tile assigned to a GPU workgroup in the S256 expert contractions: **1,658 → 996 ms** at full-model level. I then tested schedules where a thread handled two and four output rows, changing gate/up and down separately. The model arithmetic and weights stayed the same; I checked outputs and retimed the full model after each accepted change.

| Separate S256 A/B test | Current | Candidate | Speedup |
|---|---:|---:|---:|
| Initial output tile change | 1,657.74 ms | 995.74 ms | 1.665× |
| Gate/up: rows2 | 1,003.00 ms | 780.02 ms | 1.286× |
| Down: rows2 | 779.90 ms | 668.58 ms | 1.167× |
| Gate/up: rows4 | 663.97 ms | 570.92 ms | 1.163× |
| Down: rows4 | 570.87 ms | 520.69 ms | 1.096× |

Each row is its own current-versus-candidate test, **not** a continuous trace or additive breakdown. The tuned full-model graph had no comparable 217–279 ms scatter dispatch in its profile. The measured **3.18×** gain compares original and tuned *full-model schedules*, not a fix to the earlier grouped layer's scatter-back.

### Fewer calculated rows did not win in the tested dynamic path

The four-way S256 comparison used the same prompts, weights, strict FP32 settings and completed-output boundary, with **640 timed requests**. A separate trace of one dynamic request recorded **869 expert calls** and **32,768 actual routed rows** across 16 layers. Gate/up and down expert kernels took about **2,123 ms**. Their combined durations fit `12.29 + 64.47 × M` microseconds per call, where `M` is the assigned row count (**R² ≈ 0.999998**).

That shows a strong per-row cost in this **generated, untuned** expert schedule. The trace lacks memory-byte counters; it cannot distinguish repeated weight reads from block scheduling, reduction cost or a mixture. The dynamic arm still used padded routing and assembly—only its expert multiplications skipped the unassigned row slots.

## Explore the evidence

Offline inspection needs **Python 3.10+** and no third-party packages:

```bash
python run.py list
python run.py verify
python run.py summarize
python run.py gpu base
```

`summarize` prints the full-model and tuning tables from archived reports. `verify` checks ZIP integrity and selected headline relationships; it does **not** rerun inference or independently certify numerical correctness. `gpu PHASE` identifies the historical source and prerequisites without launching a job against unresolved Colab paths. The original `request.json`, `report.json`, raw samples, checks, lowering diffs and selected traces remain in `evidence/*.zip`.

| Path | What is inside |
|---|---|
| [`run.py`](run.py) | Offline evidence listing, checks and summary |
| [`scripts/`](scripts/) | Base build and numbered layer, tuning, dynamic and comparison scripts |
| [`evidence/`](evidence/) | Per-phase original review bundles and selected traces |
| [`result_and_summary.md`](result_and_summary.md) | Earlier CPU/G4 findings, A100 results and limitations |
| [`REPRODUCING.md`](REPRODUCING.md) | A100 rerun dependencies and guidance for other GPUs |

The major phases are **Layer-7 integration → full-model tuning → frozen S64/S256 comparison → four-way S256 comparison and trace**. Each A/B bundle also retains the original `experiment.py` and run metadata. The numbered top-level scripts are copies for navigation; editing them may invalidate the historical hash guards.

## Scope, correctness and reproduction limits

- **Model:** `allenai/OLMoE-1B-7B-0125`, revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`; 16 layers, 64 experts, top-8 routing.
- **Environment:** NVIDIA A100-SXM4-40GB; IREE compiler/runtime 3.12.0 for full-model artifacts; batch one, strict FP32 with TF32 disabled, no KV cache. `S` is prompt length; `C` is fixed row capacity per expert.
- **Full-model timer:** resident input to host-ready serving logits and all-layer expert counts. Layer-7 subpath and profile timings use narrower boundaries.
- **Numerical qualification:** serving logits, all-position logits, expert counts and all 1,280 checked top-1 predictions passed against the reference. Original and tuned static VMFBs shared one final hidden-state tolerance violation at token 0/channel 505 for each of two code prompts. The dynamic S256 path passed its hidden-state gate. The cause remains open; see the [detailed note](result_and_summary.md#numerical-qualification-and-conclusion).

This is an **auditable experiment record**, not a self-contained one-command GPU rerun. The multi-GB checkpoint, parameter archive, full-model VMFBs and complete original Colab run directory are not included. The base launcher can fetch the pinned checkpoint, but later scripts expect specific preceding artifacts, paths and hashes. A fresh run or another GPU needs a newly recorded baseline and correctness checks; follow [REPRODUCING.md](REPRODUCING.md).

This work follows my [first MoE/IREE article](https://www.linkedin.com/pulse/benchmarking-dynamic-moe-execution-iree-when-faster-result-waiz-khan-guusf/). One open next experiment is a **redesigned, tuned bucketed full-model path** that avoids the early grouped graph's scatter cost and is measured against the tuned stationary model at the same boundary.
