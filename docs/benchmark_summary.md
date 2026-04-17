# Mode Benchmark Summary

| Mode | Target path | Notes |
| --- | --- | --- |
| NORMAL | CPU scalar / Java post-processing | Baseline correctness reference |
| SIMD | CPU + NEON intrinsics | Lower latency through vector preprocessing/postprocessing |
| HYBRID | GPU delegate + NEON | Best available runtime on current app build |
| GPU | Dashboard fallback routed to hybrid path | UI still exposes GPU selection for comparison |

Dashboard labels were updated to show live processing latency, end-to-end latency, and mode status for side-by-side comparison.
