# Results and findings

## Earlier work that motivated this study

The first [MoE/IREE article](https://www.linkedin.com/pulse/benchmarking-dynamic-moe-execution-iree-when-faster-result-waiz-khan-guusf/) tested narrower boundaries. On a Colab CPU, 40 randomized Layer-7 B64 pairs had 2.2533 s median for host grouping and 2.3792 s for IREE route/pack; pack and combine took about 5.7 and 5.2 ms, so expert calls dominated. In a single-thread CPU expert replay, IREE dynamic calls were about 22.67×, 6.87×, 3.51×, 3.77× and 4.86× PyTorch at M=1,2,6,23,47; the 51-expert replay took 1.9209 versus 0.4569 s (paired ratio 3.80×). On a G4, one expert's IREE dynamic time rose from 0.415 ms at M=1 to 2.867 ms at M=47; fixing M=47 cut it to 1.102 ms, a 2.60× improvement at that expert-call boundary. A synthetic padded CPU graph also favored regular work. These previous results prompted the real-layer and full-model experiments; they do **not** predict their end-to-end performance. Their original data are outside this A100 archive.

## A100 Layer-7 grouping

The prepared dynamic expert core took 37.64/37.80 ms in two runs, while grouped 50-expert core took 16.57/16.66 ms. This excluded routing, weight selection, and scatter/combine and omitted the busiest 47-row expert. Integrated primary-trace layer timings reversed: dynamic 86.73/86.57 ms, Active-51 272.44/273.31 ms, Stationary-64 332.17/333.25 ms. A second trace with 55 active experts repeated the direction (dynamic 83–87 ms; Active-55 about 292 ms; Stationary-64 about 332–333 ms). The integrated timer ended at GPU-resident combined output and excluded router projection and final host readback. In the generated trace, scatter-back took about 217–234 ms in Active-51 and 279 ms in Stationary-64, with a reported one-block, one-thread launch. The prepared core and routed layer cover different work; their times cannot be subtracted as an exact integration penalty. The later full-model graph used a different padded formulation and did not contain a comparable scatter dispatch in its profile.

## Full-model schedule tuning

The model has 16 layers. At S256, top-8 routing yields 2,048 real assignments per layer. S256/C256 allocates 64×256 = 16,384 expert row slots per layer, eight times the real assignment count. The selected compiler schedule changes altered gate/up and down contraction lowering, not the model arithmetic. These are independent current-versus-candidate A/B sessions; they are not an additive decomposition or the entire search space.

| Candidate | Current A (ms) | Candidate B (ms) | A/B |
|---|---:|---:|---:|
| Initial tiling | 1,657.74 | 995.74 | 1.665× |
| Gate/up rows2 | 1,003.00 | 780.02 | 1.286× |
| Down rows2 | 779.90 | 668.58 | 1.167× |
| Gate/up rows4 | 663.97 | 570.92 | 1.163× |
| Down rows4 | 570.87 | 520.69 | 1.096× |

The frozen three-way full-model comparison used grouped compiled PyTorch (`torch.compile` with CUDA graphs), original padded IREE and tuned padded IREE at the same completed-output boundary. It comprised 960 timed requests over eight prompt/length fixtures.

| Length | PyTorch (ms) | Original IREE (ms) | Tuned IREE (ms) | Original / tuned |
|---|---:|---:|---:|---:|
| S64/C64 | 97.41 | 473.13 | 308.15 | 1.54× |
| S256/C256 | 188.96 | 1,657.67 | 520.62 | 3.18× |

At S256 the schedule cut padded IREE latency by 68.6%; it remained 2.76× compiled PyTorch. A tuned S256 profile attributed about 251 ms to expert contractions, 193 ms to two dense projection families, 32 ms to router sorting, and about 19 ms to attention; the remaining kernels together took about 23 ms. These profile components are from a separate profiled request.

## Four-way S256 comparison and trace

The dynamic arm used an **untuned exact-row expert VMFB** and Python per-expert orchestration in a split full-model path; it still used padded routing and assembly and read counts back once per layer. Four prompts, two isolated processes per arm and 640 timed requests gave these means of prompt medians:

| Arm | Latency (ms) |
|---|---:|
| Compiled grouped PyTorch | 189.37 |
| Tuned padded IREE | 520.97 |
| Original padded IREE | 1,658.24 |
| Untuned dynamic-row IREE | 2,704.23 |

The dynamic arm ranged 2,685.9–2,733.6 ms by prompt; its two process medians agreed within 1.06% per prompt. It was 1.63× slower than original padded IREE and 5.19× slower than tuned padded IREE, though its expert GEMMs processed one-eighth as many row slots.

One `compiler_s256` CUDA trace recorded 869 expert calls, 32,768 routed rows across 16 layers, about 2,123 ms in gate/up and down expert kernels, about 263 ms in other kernels, 2,979 kernel events and 16 count readbacks. Paired expert durations fit `12.29 + 64.47 × M` microseconds per call (R² ≈ 0.999998); both contraction launch grids scaled as `2,048 × M` blocks. The trace supports strong row-proportional cost for this generated schedule. It lacks L2/DRAM byte and latency counters, so repeated weight reads, block scheduling and reductions remain possible contributors.

For scale, 251 ms divided by 16×16,384 padded static slots is about 0.96 µs per slot, versus 64.47 µs per real dynamic row from the separate fit, roughly 67×. The expert sums are about 8.5× apart despite eight times fewer dynamic rows. These are derived across different requests and generated schedules and do not isolate a physical cause. The trace kernel sum must not be subtracted from a different unprofiled four-prompt mean to infer Python overhead.

## Numerical qualification and conclusion

Original and tuned static VMFBs passed serving logits, all-position logits, expert counts and all 1,280 checked top-1 predictions against HF, but both had a single final hidden-state tolerance violation at token 0, channel 505 for each of the two code prompts. For `code_s256`, static IREE was 0.152906135 versus HF 0.153474331: absolute difference 0.000568196 against allowed 0.000346042 (`rtol=atol=3e-4`). The dynamic S256 path passed the hidden-state gate on all four prompts; at this coordinate it returned 0.153373942. The final rows4 edit did not introduce the shared static deviation. Padding as its cause remains unproven; graph partitioning, lowering and arithmetic order also differ. An FP64 reference and layer-by-layer comparison remain future numerical work.

The supported result is implementation-specific: full-model schedule tuning improved padded prefill from about 1,658 to 521 ms, while the tested untuned dynamic split path took about 2,704 ms. A redesigned tuned bucketed full-model path is an open experiment, since the earlier grouped layer graph suffered costly scatter-back. Absolute latencies belong to the pinned A100 environment and stated timing boundary.
