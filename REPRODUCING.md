# Reproducing and porting the GPU experiments

## A100 source run

Use an A100-SXM4-40GB, ample host memory (the original launcher requests at least 48 GiB) and about 100 GB free disk. An initial model download and compile may take hours. The original launcher installs a pinned isolated environment (`torch==2.11.0`, `transformers==4.51.3`, `iree-turbine==3.9.0`, IREE compiler/runtime `3.12.0`, plus the pins in its source). Ensure access to the checkpoint revision and accept the model's license/terms where relevant.

```bash
python scripts/27_full_model_aot.py --root /content/drive/MyDrive/moe_iree_olmoe --lengths 64,256 --logits all
```

This script runs correctness gates and produces a timestamped `27_full_model_aot/` run, including imported MLIR, compiled VMFB, IRPA parameters, prompt fixtures, reference outputs, hashes and timings. The exact historical path was `/content/drive/MyDrive/moe_iree_olmoe/27_full_model_aot/20261028T140743_199932Z`. The A/B and final scripts were frozen against that run and contain `RUN` constants and SHA256 checks. To reproduce the *archived* comparisons, restore the complete original source-run directory and any previous candidate VMFBs at their recorded paths; then use the numbered scripts in order. A newly generated timestamped run is a **new experiment**: inspect the scripts and deliberately update their run paths, artifact hashes and baseline references, then record the new protocol rather than treating its numbers as a byte-identical rerun.

Run order after the base build: `30` tiling, `34` gate/up rows2, `36` down rows2, `39` gate/up rows4, `41` down rows4, `42` frozen comparison, `44` dynamic split build, `46` four-way comparison. Each candidate should be independently built, checked against HF for logits, counts and diagnostic outputs, warmed, and timed in isolated balanced processes. The scripts' headers and `request.json` records are authoritative for command arguments, warmups, repeats and dependencies. The `44` source is in its own evidence ZIP with an accompanying README. The `46` run requires outputs from `44` and the same original/tuned artifacts used by `42`.

The historical Layer-7 work also used a fixed fixture (`olmoe_layer7_prefill64_decode1.pt`), real expert weights, separate route/combine VMFBs and a frozen `25C2_runner_validation.py`. Those large artifacts and the runner are absent here. Its included bridge and trace bundles support review of the reported numbers, not a one-command Layer-7 rerun.

## Other GPUs

Start with `27_full_model_aot.py` as a *port*, not a direct replay. It explicitly checks compute capability `(8,0)` and uses `--iree-cuda-target=sm_80`; update both for the actual NVIDIA architecture. Other scripts and VMFBs may contain A100-specific target assumptions, device-name checks and pinned native-library hashes. Recompile every VMFB and rerun all correctness checks. Verify strict FP32/TF32 controls and supported IREE lowering flags on the chosen IREE release. GPU memory and host-memory needs may change; reduce the tested sequence set if required and report that change. Reusing A100 compiled VMFBs or editing only the device check is insufficient.

On another GPU, collect a fresh baseline and a fresh A/B sequence. Report device name, compute capability, memory, driver/CUDA, IREE version, compiler flags, checkpoint revision, S/C, precision, warmups, samples, timer boundary, output checks and raw per-request data. Compare variants **on that same GPU**. Absolute A100 latencies and even the best tile configuration need not transfer. For non-NVIDIA GPUs, port the target backend and device-specific tracing code; the supplied CUDA scripts are not generic.

For analysis without GPU access, `python run.py summarize` reads the archived reports. This reproduces tables and ratios, not kernel executions. The full-model original and tuned paths retain a known final hidden-state discrepancy at two code fixtures; keep it visible alongside passing serving logits/top-1/count checks.
