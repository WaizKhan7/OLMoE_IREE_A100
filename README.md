# OLMoE prefill in IREE: experiment reproduction archive

Companion evidence and source scripts for the follow-up to [Benchmarking Dynamic MoE with IREE](https://www.linkedin.com/pulse/benchmarking-dynamic-moe-execution-iree-when-faster-result-waiz-khan-guusf/). This archive preserves the major A100 comparisons and their original outputs. It does **not** contain the multi-GB checkpoint, parameter archive, full-model VMFBs, or the complete original Colab run directory.

## Start here

Requires Python 3.10+ for offline inspection, with no third-party packages:

```bash
python run.py list
python run.py verify
python run.py summarize
python run.py gpu base
```

`verify` checks ZIP integrity and selected headline relationships; it does not rerun inference or independently certify correctness. The original reports, checks, raw samples, lowering diffs and selected traces are in `evidence/*.zip`; inspect with `unzip -l` and `unzip -p`. `run.py gpu PHASE` prints the relevant source/evidence and prerequisite reminder. It deliberately does not execute a historical GPU job with an unresolved Colab path.

## Major phases

| Phase | Question | Source and evidence |
|---|---|---|
| Layer-7 | Does grouping help once routing and output assembly are included? | `25B*`, `25C1*` source; `prepared_core_bridge_bundle.zip` and `device_ready_traces.zip`. The frozen integrated runner and full historical VMFB/weights are not included. |
| Full-model tuning | Which expert contraction schedules improve completed model latency? | `27*` base build; `30*`, `34*`, `36*`, `39*`, `41*` A/B scripts and matching five bundles. |
| Frozen comparison | How does the final tuned VMFB compare to original IREE and grouped compiled PyTorch at S64/S256? | `42*` script; `final_full_model_gap_bundle.zip`. |
| Four-way S256 | How does the untuned exact-row split path compare at the same output boundary? | `44*` dynamic builder, `46*` comparison; their two bundles. |

The original `experiment.py` copied into each bundle is retained. The numbered scripts are convenient top-level copies. The runs used exact file and source hashes; moving or editing a historical script can invalidate its guards. See [REPRODUCING.md](REPRODUCING.md) for the A100 workflow and other-GPU guidance, and [result_and_summary.md](result_and_summary.md) for the outcomes and limitations.

## Scope

`allenai/OLMoE-1B-7B-0125` at revision `9b0c1aa87e34a20052389dce1f0cf01da783f654`; batch one, prefill, strict FP32 with TF32 disabled, no KV cache. The full-model results used IREE 3.12.0 and one NVIDIA A100-SXM4-40GB. S denotes prompt length; C denotes fixed capacity per expert. The frozen comparison uses S64/C64 and S256/C256. Timings cover resident input to host-ready logits and all-layer expert counts; the Layer-7 and trace measurements have narrower boundaries.

## Evidence limits

The per-phase archives are review bundles rather than a fully self-contained GPU benchmark. The pinned checkpoint is fetched by `27_full_model_aot.py`, but subsequent scripts refer to an exact original run directory, source hashes, VMFBs, parameter bank, and sometimes historical Layer-7 trace files. The optional CUDA trace source is included, but byte counters were never recorded. See the per-bundle `request.json` / `report.json` for protocol and exact provenance. No new GPU measurements were performed while assembling this ZIP.
