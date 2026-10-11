# E18 measured tables

These tables describe E18-small-v1 with its batch-control addendum. FP32, L≤512, raw logits. Ratios above one favor the candidate. Confidence intervals are unadjusted exploratory session-bootstrap intervals; no confirmatory significance family is claimed. Accuracy-failing paths remain diagnostic. Missing observations are pending, not zero.

## Accuracy

| Device/path | Status | Strict numerical | Application | Max / P99 absolute score error | Largest top-1 / top-10 membership changes per cell |
|---|---|---|---|---|---|
| cuda_compiled | VERIFIED | True | True | 1e-05 / 2.86e-06 | 0.00% / 0.00% |
| cuda_cuda | VERIFIED | True | True | 1.55e-05 / 2.86e-06 | 0.00% / 0.00% |
| cuda_explicit | VERIFIED | True | True | 0 / 0 | 0.00% / 0.00% |
| cuda_stock | VERIFIED | True | True | 0 / 0 | 0.00% / 0.00% |
| cuda_triton | VERIFIED | True | True | 1.45e-05 / 2.86e-06 | 0.00% / 0.00% |
| mps_compiled | VERIFIED | True | True | 7.15e-06 / 2.86e-06 | 0.00% / 0.00% |
| mps_metal | VERIFIED | True | True | 1.67e-05 / 9.06e-06 | 0.00% / 0.00% |
| mps_metal_compiled | VERIFIED | True | True | 1.67e-05 / 9.06e-06 | 0.00% / 0.00% |
| mps_mlx | VERIFIED | True | True | 1.74e-05 / 9.06e-06 | 0.00% / 0.00% |
| mps_mlx_compiled | VERIFIED | True | True | 1.74e-05 / 9.06e-06 | 0.00% / 0.00% |
| mps_stock | VERIFIED | True | True | 0 / 0 | 0.00% / 0.00% |

Full error distributions, per-query rankings and nDCG/MRR differences with paired query intervals are in each path’s accuracy JSON and checksummed score artifact. Stock is the practical SDPA baseline; its self-comparison is not an independent full CPU audit. The CPU diagnostic uses four padded pairs.

## Primary complete direct scoring

| Device | Path | Accuracy | Pairs/s median [IQR] | Sessions / repetitions | Stock / candidate time ratio [95% CI] |
|---|---|---|---|---|---|
| cuda | compiled | VERIFIED | 736.8 [733.8, 737.5] | 5 / 15 | 1.033 [1.032, 1.034] |
| cuda | cuda | VERIFIED | 715.0 [713.6, 715.6] | 5 / 15 | 1.003 [1.002, 1.006] |
| cuda | explicit | VERIFIED | 712.7 [711.0, 713.7] | 5 / 15 | 0.999 [0.994, 1.002] |
| cuda | stock | VERIFIED | 712.4 [710.9, 713.4] | 5 / 15 | Baseline |
| cuda | triton | VERIFIED | 717.5 [716.3, 719.0] | 5 / 15 | 1.007 [1.006, 1.009] |
| mps | compiled | VERIFIED | 102.1 [83.1, 115.5] | 6 / 18 | 1.137 [1.081, 1.198] |
| mps | metal | VERIFIED | 37.1 [34.4, 42.7] | 6 / 18 | 0.428 [0.382, 0.472] |
| mps | metal_compiled | VERIFIED | 52.5 [43.4, 55.2] | 6 / 18 | 0.569 [0.485, 0.697] |
| mps | mlx | VERIFIED | 40.7 [34.0, 47.1] | 6 / 18 | 0.471 [0.440, 0.494] |
| mps | mlx_compiled | VERIFIED | 48.9 [41.3, 68.3] | 6 / 18 | 0.589 [0.550, 0.644] |
| mps | stock | VERIFIED | 87.4 [77.3, 96.3] | 6 / 18 | Baseline |

## Local loopback HTTP

| K | C | Coordinator/worker | P50 / P95 / P99 ms (median across repetitions) | req/s [IQR] | pairs/s | Successful / failed / timed-out requests | Python / Rust time ratio [95% CI] |
|---|---|---|---|---|---|---|---|
| 10 | 1 | python/mlx | 161.5 / 407.7 / 468.4 | 4.922 [4.340, 5.641] | 49.2 | 256 / 0 / 0 | — |
| 10 | 1 | python/stock | 73.9 / 78.6 / 79.0 | 13.608 [12.998, 14.200] | 136.1 | 256 / 0 / 0 | — |
| 10 | 1 | rust/mlx | 158.2 / 488.6 / 605.0 | 4.397 [3.823, 4.935] | 44.0 | 256 / 0 / 0 | 0.868 [0.802, 0.938] |
| 10 | 1 | rust/stock | 71.0 / 75.7 / 76.9 | 14.235 [13.907, 14.318] | 142.4 | 256 / 0 / 0 | 1.036 [1.011, 1.064] |
| 10 | 4 | python/mlx | 809.7 / 1195.2 / 1257.1 | 4.495 [3.951, 4.850] | 45.0 | 256 / 0 / 0 | — |
| 10 | 4 | python/stock | 285.2 / 299.8 / 303.6 | 13.854 [13.462, 14.237] | 138.5 | 256 / 0 / 0 | — |
| 10 | 4 | rust/mlx | 807.5 / 1119.6 / 1200.4 | 4.545 [4.149, 5.067] | 45.5 | 256 / 0 / 0 | 1.011 [0.913, 1.113] |
| 10 | 4 | rust/stock | 278.3 / 291.5 / 294.2 | 14.265 [13.898, 14.412] | 142.6 | 256 / 0 / 0 | 1.016 [0.995, 1.038] |
| 10 | 8 | python/mlx | 1357.8 / 1850.7 / 1921.9 | 4.594 [4.381, 5.369] | 45.9 | 256 / 0 / 0 | — |
| 10 | 8 | python/stock | 556.7 / 579.5 / 583.5 | 13.960 [12.857, 14.251] | 139.6 | 256 / 0 / 0 | — |
| 10 | 8 | rust/mlx | 1564.0 / 2227.1 / 2245.0 | 4.453 [3.789, 4.794] | 44.5 | 256 / 0 / 0 | 0.895 [0.800, 1.002] |
| 10 | 8 | rust/stock | 546.6 / 568.5 / 572.9 | 14.329 [13.642, 14.491] | 143.3 | 256 / 0 / 0 | 1.041 [1.008, 1.082] |
| 50 | 1 | python/mlx | 785.2 / 1509.0 / 1969.6 | 1.143 [0.718, 1.258] | 57.2 | 256 / 0 / 0 | — |
| 50 | 1 | python/stock | 374.1 / 409.3 / 420.9 | 2.643 [2.547, 2.745] | 132.2 | 256 / 0 / 0 | — |
| 50 | 1 | rust/mlx | 835.0 / 2015.6 / 2393.6 | 0.845 [0.594, 1.265] | 42.3 | 256 / 0 / 0 | 0.907 [0.779, 1.021] |
| 50 | 1 | rust/stock | 367.5 / 384.2 / 393.6 | 2.731 [2.706, 2.757] | 136.6 | 256 / 0 / 0 | 1.019 [0.997, 1.039] |
| 50 | 4 | python/mlx | 3178.2 / 4656.2 / 4871.7 | 1.115 [0.886, 1.220] | 55.7 | 256 / 0 / 0 | — |
| 50 | 4 | python/stock | 1471.1 / 1559.7 / 1562.4 | 2.650 [2.569, 2.743] | 132.5 | 256 / 0 / 0 | — |
| 50 | 4 | rust/mlx | 3359.5 / 4703.5 / 4854.4 | 1.101 [0.836, 1.225] | 55.0 | 256 / 0 / 0 | 1.020 [0.948, 1.097] |
| 50 | 4 | rust/stock | 1457.3 / 1579.2 / 1580.5 | 2.702 [2.636, 2.754] | 135.1 | 256 / 0 / 0 | 1.014 [0.996, 1.036] |
| 50 | 8 | python/mlx | 6414.0 / 7167.4 / 7179.9 | 1.152 [0.941, 1.229] | 57.6 | 256 / 0 / 0 | — |
| 50 | 8 | python/stock | 2968.0 / 3128.4 / 3129.0 | 2.623 [2.546, 2.681] | 131.1 | 256 / 0 / 0 | — |
| 50 | 8 | rust/mlx | 6535.8 / 7881.1 / 8080.8 | 1.113 [0.866, 1.191] | 55.6 | 256 / 0 / 0 | 0.962 [0.867, 1.068] |
| 50 | 8 | rust/stock | 2878.0 / 2987.5 / 2990.8 | 2.736 [2.594, 2.756] | 136.8 | 256 / 0 / 0 | 1.020 [1.003, 1.038] |
| 100 | 1 | python/mlx | 1526.5 / 2527.3 / 2663.2 | 0.611 [0.571, 0.626] | 61.1 | 256 / 0 / 0 | — |
| 100 | 1 | python/stock | 762.1 / 811.1 / 839.6 | 1.320 [1.236, 1.364] | 132.0 | 256 / 0 / 0 | — |
| 100 | 1 | rust/mlx | 1524.2 / 2585.0 / 3127.6 | 0.607 [0.553, 0.627] | 60.7 | 256 / 0 / 0 | 0.965 [0.920, 1.018] |
| 100 | 1 | rust/stock | 739.4 / 807.5 / 821.8 | 1.343 [1.305, 1.370] | 134.3 | 256 / 0 / 0 | 1.022 [1.012, 1.034] |
| 100 | 4 | python/mlx | 6081.4 / 7651.1 / 7990.4 | 0.619 [0.591, 0.635] | 61.9 | 256 / 0 / 0 | — |
| 100 | 4 | python/stock | 2997.2 / 3168.0 / 3178.3 | 1.319 [1.225, 1.345] | 131.9 | 256 / 0 / 0 | — |
| 100 | 4 | rust/mlx | 6303.6 / 7367.1 / 7493.2 | 0.613 [0.592, 0.639] | 61.3 | 256 / 0 / 0 | 0.987 [0.952, 1.025] |
| 100 | 4 | rust/stock | 2915.7 / 3030.0 / 3054.0 | 1.364 [1.315, 1.376] | 136.4 | 256 / 0 / 0 | 1.040 [1.010, 1.072] |
| 100 | 8 | python/mlx | 12079.7 / 13790.3 / 13818.3 | 0.607 [0.568, 0.654] | 60.7 | 256 / 0 / 0 | — |
| 100 | 8 | python/stock | 5916.9 / 6201.0 / 6214.8 | 1.316 [1.224, 1.361] | 131.6 | 256 / 0 / 0 | — |
| 100 | 8 | rust/mlx | 12463.1 / 14273.3 / 14487.9 | 0.598 [0.533, 0.624] | 59.8 | 256 / 0 / 0 | 0.967 [0.903, 1.026] |
| 100 | 8 | rust/stock | 5789.7 / 5967.4 / 5982.9 | 1.357 [1.328, 1.378] | 135.7 | 256 / 0 / 0 | 1.034 [1.006, 1.063] |

The 16-request repetition gives only two waves at C8. P99 is a near-maximum of a short wave, not a production tail-latency estimate. The load is fixed-concurrency closed-loop; it does not test open-loop overload. Rust coordinates a shared Python inference worker.

## HTTP queue and processing times

Each entry is the median across repetitions of that repetition’s request percentile. Queue and processing timers are server-side; request latency above is client-observed.

| K / C | Coordinator/worker | Queue P50 / P95 / P99 ms | Processing P50 / P95 / P99 ms |
|---|---|---|---|
| K10/C1 | python/mlx | 0.0 / 0.1 / 0.1 | 157.2 / 405.7 / 466.4 |
| K10/C1 | python/stock | 0.0 / 0.0 / 0.0 | 72.9 / 77.6 / 78.0 |
| K10/C1 | rust/mlx | 0.0 / 0.0 / 0.0 | 153.9 / 487.1 / 602.9 |
| K10/C1 | rust/stock | 0.0 / 0.0 / 0.0 | 70.3 / 74.8 / 76.2 |
| K10/C4 | python/mlx | 602.2 / 967.2 / 1003.0 | 170.2 / 454.0 / 495.3 |
| K10/C4 | python/stock | 213.4 / 227.8 / 228.0 | 72.9 / 77.5 / 77.9 |
| K10/C4 | rust/mlx | 576.8 / 863.8 / 926.6 | 171.1 / 452.4 / 487.8 |
| K10/C4 | rust/stock | 206.8 / 220.3 / 221.7 | 70.9 / 75.2 / 75.6 |
| K10/C8 | python/mlx | 1174.1 / 1589.8 / 1703.9 | 150.8 / 409.4 / 458.3 |
| K10/C8 | python/stock | 484.6 / 509.0 / 513.4 | 71.9 / 76.9 / 77.4 |
| K10/C8 | rust/mlx | 1276.6 / 1962.3 / 2075.6 | 162.1 / 462.9 / 539.1 |
| K10/C8 | rust/stock | 478.3 / 501.2 / 501.6 | 69.9 / 75.4 / 76.7 |
| K50/C1 | python/mlx | 0.0 / 0.1 / 0.1 | 781.2 / 1503.6 / 1962.6 |
| K50/C1 | python/stock | 0.0 / 0.0 / 0.0 | 372.5 / 407.7 / 419.3 |
| K50/C1 | rust/mlx | 0.0 / 0.0 / 0.0 | 833.4 / 2008.3 / 2384.4 |
| K50/C1 | rust/stock | 0.0 / 0.0 / 0.0 | 366.4 / 382.9 / 392.4 |
| K50/C4 | python/mlx | 2306.5 / 3750.8 / 3953.4 | 755.5 / 1573.7 / 2205.1 |
| K50/C4 | python/stock | 1102.7 / 1174.0 / 1183.9 | 374.0 / 401.8 / 406.4 |
| K50/C4 | rust/mlx | 2424.8 / 3722.1 / 3897.6 | 788.1 / 1576.0 / 1964.3 |
| K50/C4 | rust/stock | 1091.8 / 1193.7 / 1202.8 | 367.7 / 403.5 / 412.8 |
| K50/C8 | python/mlx | 5590.1 / 6299.2 / 6427.8 | 793.8 / 1198.6 / 1411.7 |
| K50/C8 | python/stock | 2582.5 / 2742.8 / 2747.0 | 374.3 / 414.9 / 423.2 |
| K50/C8 | rust/mlx | 5574.6 / 6997.0 / 7055.0 | 810.8 / 1558.1 / 2057.3 |
| K50/C8 | rust/stock | 2515.4 / 2627.7 / 2633.5 | 367.7 / 398.9 / 402.9 |
| K100/C1 | python/mlx | 0.1 / 0.2 / 0.2 | 1519.5 / 2520.3 / 2656.5 |
| K100/C1 | python/stock | 0.0 / 0.0 / 0.0 | 760.2 / 809.2 / 837.6 |
| K100/C1 | rust/mlx | 0.0 / 0.0 / 0.0 | 1518.7 / 2581.0 / 3124.4 |
| K100/C1 | rust/stock | 0.0 / 0.0 / 0.0 | 738.0 / 806.1 / 820.4 |
| K100/C4 | python/mlx | 4557.4 / 6032.7 / 6206.6 | 1452.9 / 2424.0 / 2563.3 |
| K100/C4 | python/stock | 2239.7 / 2380.3 / 2393.1 | 754.8 / 812.8 / 848.1 |
| K100/C4 | rust/mlx | 4601.8 / 5662.8 / 5881.3 | 1491.1 / 2342.3 / 2508.7 |
| K100/C4 | rust/stock | 2185.2 / 2299.7 / 2310.4 | 734.8 / 795.4 / 829.8 |
| K100/C8 | python/mlx | 10418.8 / 12193.2 / 12279.0 | 1509.4 / 2340.1 / 2562.6 |
| K100/C8 | python/stock | 5163.1 / 5445.5 / 5461.2 | 756.9 / 807.8 / 849.8 |
| K100/C8 | rust/mlx | 10918.4 / 12564.5 / 12801.8 | 1512.8 / 2402.3 / 2602.4 |
| K100/C8 | rust/stock | 5052.4 / 5231.7 / 5248.0 | 735.1 / 815.4 / 848.1 |

## Service setup and process memory

Readiness includes model loading and all HTTP shape warm-up. RSS values are after-run snapshots, not peak memory; shared pages may be counted twice.

| Study / coordinator / worker | Starts | Readiness seconds median [IQR] | Three HTTP warm-up requests, seconds median | Worker / coordinator RSS median MiB |
|---|---|---|---|---|
| optimized_mlx/python/mlx | 8 | 14.6 [14.2, 15.1] | 2.3 | 90.8 / 29.6 |
| optimized_mlx/python/mlx_compiled | 8 | 14.1 [13.5, 14.5] | 2.0 | 138.8 / 29.6 |
| optimized_stock/python/compiled | 8 | 8.9 [8.8, 9.0] | 1.2 | 984.6 / 88.5 |
| optimized_stock/python/stock | 8 | 7.0 [6.9, 7.0] | 1.2 | 751.8 / 88.5 |
| primary/python/mlx | 8 | 13.9 [13.7, 14.2] | 2.1 | 100.7 / 28.2 |
| primary/python/stock | 8 | 8.5 [8.1, 9.0] | 1.3 | 753.6 / 88.7 |
| primary/rust/mlx | 8 | 13.7 [13.0, 15.7] | 2.2 | 55.5 / 5.4 |
| primary/rust/stock | 8 | 6.4 [6.3, 7.6] | 1.2 | 751.9 / 9.1 |

## Follow-up optimized HTTP

Each family uses newly started matched baseline workers. These are conditional follow-ups selected from direct scoring; compare within this family, rather than with the earlier coordinator study.

| Study / workload | Worker (Python coordinator) | P50 / P95 / P99 ms | Requests/s | Matched baseline / candidate time [95% CI] |
|---|---|---|---|---|
| optimized_mlx/K100/C1 | python/mlx | 1453.1 / 2454.3 / 2614.3 | 0.618 | Baseline |
| optimized_mlx/K100/C1 | python/mlx_compiled | 1292.5 / 2243.3 / 2296.4 | 0.658 | 1.112 [1.062, 1.171] |
| optimized_mlx/K100/C4 | python/mlx | 5987.4 / 7385.5 / 7447.3 | 0.637 | Baseline |
| optimized_mlx/K100/C4 | python/mlx_compiled | 5555.2 / 6474.2 / 6601.6 | 0.689 | 1.100 [1.023, 1.172] |
| optimized_mlx/K100/C8 | python/mlx | 11789.0 / 14282.9 / 14378.2 | 0.617 | Baseline |
| optimized_mlx/K100/C8 | python/mlx_compiled | 10766.1 / 12315.1 / 12384.6 | 0.694 | 1.157 [1.103, 1.221] |
| optimized_stock/K100/C1 | python/compiled | 731.3 / 765.6 / 794.9 | 1.378 | 0.995 [0.985, 1.005] |
| optimized_stock/K100/C1 | python/stock | 734.3 / 764.7 / 792.2 | 1.373 | Baseline |
| optimized_stock/K100/C4 | python/compiled | 2890.1 / 2971.7 / 3001.4 | 1.377 | 1.003 [0.990, 1.016] |
| optimized_stock/K100/C4 | python/stock | 2889.9 / 2996.2 / 3020.4 | 1.366 | Baseline |
| optimized_stock/K100/C8 | python/compiled | 5721.6 / 5856.8 / 5860.1 | 1.376 | 1.010 [1.000, 1.021] |
| optimized_stock/K100/C8 | python/stock | 5763.9 / 5890.0 / 5943.2 | 1.370 | Baseline |

## Resident operation

| Device / rows × 384 | Path | Median µs [IQR] | Native / candidate time ratio [95% CI] |
|---|---|---|---|
| cuda/2048 | cuda | 20.1 [15.2, 25.8] | 0.862 [0.837, 0.887] |
| cuda/2048 | native | 17.9 [13.2, 22.3] | Baseline |
| cuda/2048 | triton | 18.2 [17.4, 29.9] | 0.795 [0.710, 0.971] |
| cuda/4096 | cuda | 15.5 [15.3, 25.7] | 0.865 [0.839, 0.891] |
| cuda/4096 | native | 13.7 [13.7, 21.1] | Baseline |
| cuda/4096 | triton | 17.6 [17.4, 29.1] | 0.765 [0.742, 0.788] |
| cuda/8192 | cuda | 15.6 [15.4, 25.9] | 1.226 [1.009, 1.476] |
| cuda/8192 | native | 22.7 [22.7, 22.8] | Baseline |
| cuda/8192 | triton | 17.6 [17.4, 29.2] | 1.100 [0.920, 1.316] |
| mps/2048 | metal | 254.5 [253.5, 258.2] | 1.406 [1.196, 1.812] |
| mps/2048 | native | 324.5 [323.1, 358.0] | Baseline |
| mps/4096 | metal | 343.1 [337.0, 346.1] | 1.377 [1.362, 1.393] |
| mps/4096 | native | 467.2 [464.8, 477.9] | Baseline |
| mps/8192 | metal | 501.3 [499.3, 510.8] | 1.499 [1.460, 1.525] |
| mps/8192 | native | 759.7 [756.9, 767.4] | Baseline |

Apple operation measurements include dispatch and synchronized amortized wall time. CUDA operation measurements use device events over 100 launches and include GPU-visible launch gaps. These scopes differ and must not be interpreted as cross-hardware kernel speedups.

## Complete secondary direct matrix

| Device | Scope | K / B / L / group | Path | Pairs/s median [IQR] | Accuracy |
|---|---|---|---|---|---|
| cuda | forward | 10 / 16 / 512 / sweep | compiled | 1734.0 [1667.9, 1749.7] | VERIFIED |
| cuda | forward | 10 / 16 / 512 / sweep | cuda | 1549.4 [1503.3, 1559.5] | VERIFIED |
| cuda | forward | 10 / 16 / 512 / sweep | explicit | 1539.4 [1496.6, 1553.4] | VERIFIED |
| cuda | forward | 10 / 16 / 512 / sweep | stock | 1533.0 [1492.1, 1551.7] | VERIFIED |
| cuda | forward | 10 / 16 / 512 / sweep | triton | 1568.9 [1528.9, 1574.6] | VERIFIED |
| cuda | forward | 100 / 1 / 512 / sweep | compiled | 1058.2 [1047.5, 1059.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 1 / 512 / sweep | cuda | 617.9 [613.8, 624.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 1 / 512 / sweep | explicit | 711.6 [703.2, 718.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 1 / 512 / sweep | stock | 682.5 [679.2, 684.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 1 / 512 / sweep | triton | 598.8 [586.7, 602.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 16 / 128 / sweep | compiled | 7125.0 [7117.4, 7130.5] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 16 / 128 / sweep | cuda | 6600.3 [6580.7, 6607.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 16 / 128 / sweep | explicit | 6638.2 [6626.7, 6648.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 16 / 128 / sweep | stock | 6591.7 [6584.6, 6597.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 16 / 128 / sweep | triton | 6584.7 [6577.1, 6602.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 16 / 256 / sweep | compiled | 3719.7 [3674.5, 3751.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 16 / 256 / sweep | cuda | 3067.7 [3049.2, 3107.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 16 / 256 / sweep | explicit | 2969.5 [2918.7, 3026.3] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 16 / 256 / sweep | stock | 2989.0 [2947.5, 3010.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 16 / 256 / sweep | triton | 3115.1 [3054.1, 3136.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 16 / 512 / primary | compiled | 1446.4 [1443.0, 1447.6] | VERIFIED |
| cuda | forward | 100 / 16 / 512 / primary | cuda | 1390.0 [1388.3, 1392.2] | VERIFIED |
| cuda | forward | 100 / 16 / 512 / primary | explicit | 1377.9 [1372.1, 1381.3] | VERIFIED |
| cuda | forward | 100 / 16 / 512 / primary | stock | 1377.6 [1372.5, 1380.5] | VERIFIED |
| cuda | forward | 100 / 16 / 512 / primary | triton | 1397.1 [1392.0, 1400.1] | VERIFIED |
| cuda | forward | 100 / 16 / 512 / sweep | compiled | 1479.7 [1464.3, 1485.2] | VERIFIED |
| cuda | forward | 100 / 16 / 512 / sweep | cuda | 1417.8 [1404.0, 1419.7] | VERIFIED |
| cuda | forward | 100 / 16 / 512 / sweep | explicit | 1403.0 [1391.3, 1412.3] | VERIFIED |
| cuda | forward | 100 / 16 / 512 / sweep | stock | 1401.5 [1389.0, 1409.2] | VERIFIED |
| cuda | forward | 100 / 16 / 512 / sweep | triton | 1419.7 [1409.0, 1424.7] | VERIFIED |
| cuda | forward | 100 / 32 / 512 / sweep | compiled | 1507.1 [1490.5, 1517.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 32 / 512 / sweep | cuda | 1493.2 [1482.7, 1497.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 32 / 512 / sweep | explicit | 1461.3 [1456.6, 1475.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 32 / 512 / sweep | stock | 1468.7 [1460.0, 1474.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 32 / 512 / sweep | triton | 1496.1 [1486.8, 1505.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 64 / 512 / sweep | compiled | 1563.2 [1544.6, 1570.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 64 / 512 / sweep | cuda | 1512.5 [1508.6, 1518.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 64 / 512 / sweep | explicit | 1488.5 [1475.1, 1494.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 64 / 512 / sweep | stock | 1481.4 [1474.5, 1485.5] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 64 / 512 / sweep | triton | 1507.3 [1502.9, 1518.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 8 / 512 / sweep | compiled | 1538.7 [1516.0, 1542.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 8 / 512 / sweep | cuda | 1387.9 [1380.0, 1389.3] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 8 / 512 / sweep | explicit | 1368.9 [1361.3, 1377.9] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 8 / 512 / sweep | stock | 1363.8 [1361.4, 1381.3] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 100 / 8 / 512 / sweep | triton | 1404.6 [1393.5, 1411.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | forward | 50 / 16 / 512 / sweep | compiled | 1510.9 [1464.5, 1518.8] | VERIFIED |
| cuda | forward | 50 / 16 / 512 / sweep | cuda | 1416.4 [1396.9, 1424.6] | VERIFIED |
| cuda | forward | 50 / 16 / 512 / sweep | explicit | 1410.4 [1384.7, 1416.3] | VERIFIED |
| cuda | forward | 50 / 16 / 512 / sweep | stock | 1411.7 [1396.2, 1417.8] | VERIFIED |
| cuda | forward | 50 / 16 / 512 / sweep | triton | 1429.6 [1406.0, 1435.8] | VERIFIED |
| cuda | scoring | 10 / 16 / 512 / sweep | compiled | 740.8 [731.8, 749.0] | VERIFIED |
| cuda | scoring | 10 / 16 / 512 / sweep | cuda | 711.3 [703.8, 715.0] | VERIFIED |
| cuda | scoring | 10 / 16 / 512 / sweep | explicit | 708.8 [701.5, 714.7] | VERIFIED |
| cuda | scoring | 10 / 16 / 512 / sweep | stock | 709.3 [690.9, 712.3] | VERIFIED |
| cuda | scoring | 10 / 16 / 512 / sweep | triton | 720.7 [716.6, 723.9] | VERIFIED |
| cuda | scoring | 100 / 1 / 512 / sweep | compiled | 449.2 [447.3, 451.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 1 / 512 / sweep | cuda | 366.4 [364.6, 368.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 1 / 512 / sweep | explicit | 402.6 [401.0, 403.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 1 / 512 / sweep | stock | 389.5 [387.8, 390.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 1 / 512 / sweep | triton | 359.4 [355.0, 362.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 16 / 128 / sweep | compiled | 1217.3 [1215.4, 1218.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 16 / 128 / sweep | cuda | 1202.4 [1196.4, 1205.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 16 / 128 / sweep | explicit | 1207.0 [1204.5, 1212.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 16 / 128 / sweep | stock | 1200.0 [1199.0, 1203.5] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 16 / 128 / sweep | triton | 1204.3 [1197.3, 1205.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 16 / 256 / sweep | compiled | 1059.4 [1048.9, 1065.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 16 / 256 / sweep | cuda | 1009.4 [1005.8, 1015.5] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 16 / 256 / sweep | explicit | 1000.2 [985.9, 1005.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 16 / 256 / sweep | stock | 1000.9 [998.7, 1003.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 16 / 256 / sweep | triton | 1015.1 [1009.8, 1016.5] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 16 / 512 / primary | compiled | 736.8 [733.8, 737.5] | VERIFIED |
| cuda | scoring | 100 / 16 / 512 / primary | cuda | 715.0 [713.6, 715.6] | VERIFIED |
| cuda | scoring | 100 / 16 / 512 / primary | explicit | 712.7 [711.0, 713.7] | VERIFIED |
| cuda | scoring | 100 / 16 / 512 / primary | stock | 712.4 [710.9, 713.4] | VERIFIED |
| cuda | scoring | 100 / 16 / 512 / primary | triton | 717.5 [716.3, 719.0] | VERIFIED |
| cuda | scoring | 100 / 16 / 512 / sweep | compiled | 750.8 [748.3, 753.7] | VERIFIED |
| cuda | scoring | 100 / 16 / 512 / sweep | cuda | 727.1 [725.0, 730.8] | VERIFIED |
| cuda | scoring | 100 / 16 / 512 / sweep | explicit | 726.2 [722.9, 728.8] | VERIFIED |
| cuda | scoring | 100 / 16 / 512 / sweep | stock | 723.9 [722.9, 726.0] | VERIFIED |
| cuda | scoring | 100 / 16 / 512 / sweep | triton | 730.5 [729.4, 731.0] | VERIFIED |
| cuda | scoring | 100 / 32 / 512 / sweep | compiled | 752.7 [591.3, 758.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 32 / 512 / sweep | cuda | 750.9 [550.7, 754.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 32 / 512 / sweep | explicit | 749.8 [737.6, 754.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 32 / 512 / sweep | stock | 744.9 [585.7, 749.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 32 / 512 / sweep | triton | 754.2 [573.3, 758.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 64 / 512 / sweep | compiled | 774.2 [771.7, 783.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 64 / 512 / sweep | cuda | 768.3 [765.2, 769.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 64 / 512 / sweep | explicit | 758.8 [752.7, 760.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 64 / 512 / sweep | stock | 759.6 [757.7, 761.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 64 / 512 / sweep | triton | 766.7 [765.1, 767.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 8 / 512 / sweep | compiled | 737.6 [736.1, 740.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 8 / 512 / sweep | cuda | 703.2 [701.1, 704.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 8 / 512 / sweep | explicit | 702.5 [695.1, 703.3] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 8 / 512 / sweep | stock | 698.3 [696.9, 699.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 100 / 8 / 512 / sweep | triton | 708.3 [706.9, 708.9] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| cuda | scoring | 50 / 16 / 512 / sweep | compiled | 737.9 [734.5, 739.9] | VERIFIED |
| cuda | scoring | 50 / 16 / 512 / sweep | cuda | 716.4 [713.4, 719.3] | VERIFIED |
| cuda | scoring | 50 / 16 / 512 / sweep | explicit | 715.2 [713.2, 716.9] | VERIFIED |
| cuda | scoring | 50 / 16 / 512 / sweep | stock | 714.6 [712.3, 715.9] | VERIFIED |
| cuda | scoring | 50 / 16 / 512 / sweep | triton | 718.0 [716.3, 721.2] | VERIFIED |
| mps | forward | 10 / 16 / 512 / sweep | compiled | 102.3 [79.3, 126.2] | VERIFIED |
| mps | forward | 10 / 16 / 512 / sweep | metal | 56.8 [38.7, 62.9] | VERIFIED |
| mps | forward | 10 / 16 / 512 / sweep | metal_compiled | 66.3 [58.3, 80.9] | VERIFIED |
| mps | forward | 10 / 16 / 512 / sweep | mlx | 58.7 [46.8, 62.7] | VERIFIED |
| mps | forward | 10 / 16 / 512 / sweep | mlx_compiled | 67.6 [58.7, 80.2] | VERIFIED |
| mps | forward | 10 / 16 / 512 / sweep | stock | 90.1 [74.1, 107.0] | VERIFIED |
| mps | forward | 100 / 1 / 512 / sweep | compiled | 156.8 [132.0, 171.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 1 / 512 / sweep | metal | 43.5 [32.1, 64.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 1 / 512 / sweep | metal_compiled | 83.9 [56.9, 109.5] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 1 / 512 / sweep | mlx | 44.5 [31.0, 63.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 1 / 512 / sweep | mlx_compiled | 78.0 [70.8, 92.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 1 / 512 / sweep | stock | 127.4 [85.3, 129.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 128 / sweep | compiled | 499.6 [362.4, 560.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 128 / sweep | metal | 291.5 [257.1, 358.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 128 / sweep | metal_compiled | 386.4 [310.2, 485.9] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 128 / sweep | mlx | 276.2 [226.7, 321.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 128 / sweep | mlx_compiled | 388.1 [249.8, 477.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 128 / sweep | stock | 451.7 [414.9, 564.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 256 / sweep | compiled | 203.0 [159.2, 252.9] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 256 / sweep | metal | 122.1 [109.5, 132.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 256 / sweep | metal_compiled | 136.4 [111.6, 158.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 256 / sweep | mlx | 128.6 [113.5, 147.5] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 256 / sweep | mlx_compiled | 142.5 [133.3, 166.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 256 / sweep | stock | 205.6 [176.9, 251.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 16 / 512 / primary | compiled | 99.4 [87.3, 119.1] | VERIFIED |
| mps | forward | 100 / 16 / 512 / primary | metal | 39.3 [38.3, 45.5] | VERIFIED |
| mps | forward | 100 / 16 / 512 / primary | metal_compiled | 52.1 [46.0, 58.1] | VERIFIED |
| mps | forward | 100 / 16 / 512 / primary | mlx | 43.5 [36.5, 50.1] | VERIFIED |
| mps | forward | 100 / 16 / 512 / primary | mlx_compiled | 49.9 [45.3, 69.2] | VERIFIED |
| mps | forward | 100 / 16 / 512 / primary | stock | 99.0 [82.4, 102.4] | VERIFIED |
| mps | forward | 100 / 16 / 512 / sweep | compiled | 105.6 [72.5, 115.2] | VERIFIED |
| mps | forward | 100 / 16 / 512 / sweep | metal | 42.9 [38.7, 50.3] | VERIFIED |
| mps | forward | 100 / 16 / 512 / sweep | metal_compiled | 47.7 [41.3, 59.6] | VERIFIED |
| mps | forward | 100 / 16 / 512 / sweep | mlx | 48.5 [41.3, 54.2] | VERIFIED |
| mps | forward | 100 / 16 / 512 / sweep | mlx_compiled | 51.7 [45.9, 62.7] | VERIFIED |
| mps | forward | 100 / 16 / 512 / sweep | stock | 94.1 [74.6, 119.7] | VERIFIED |
| mps | forward | 100 / 32 / 512 / sweep | compiled | 100.1 [88.9, 116.9] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 32 / 512 / sweep | metal | 48.8 [44.2, 57.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 32 / 512 / sweep | metal_compiled | 55.9 [49.5, 65.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 32 / 512 / sweep | mlx | 55.5 [46.1, 61.3] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 32 / 512 / sweep | mlx_compiled | 67.7 [51.5, 77.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 32 / 512 / sweep | stock | 92.9 [78.9, 105.5] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 64 / 512 / sweep | compiled | 100.4 [74.4, 109.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 64 / 512 / sweep | metal | 47.6 [45.6, 53.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 64 / 512 / sweep | metal_compiled | 55.4 [49.9, 68.9] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 64 / 512 / sweep | mlx | 57.3 [47.6, 66.9] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 64 / 512 / sweep | mlx_compiled | 59.6 [54.1, 84.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 64 / 512 / sweep | stock | 93.1 [76.0, 106.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 8 / 512 / sweep | compiled | 104.3 [91.9, 131.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 8 / 512 / sweep | metal | 27.3 [23.8, 31.3] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 8 / 512 / sweep | metal_compiled | 29.2 [26.7, 31.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 8 / 512 / sweep | mlx | 33.6 [25.3, 37.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 8 / 512 / sweep | mlx_compiled | 29.3 [24.3, 48.5] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 100 / 8 / 512 / sweep | stock | 106.4 [100.3, 122.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | forward | 50 / 16 / 512 / sweep | compiled | 95.3 [81.5, 115.2] | VERIFIED |
| mps | forward | 50 / 16 / 512 / sweep | metal | 39.0 [34.8, 44.8] | VERIFIED |
| mps | forward | 50 / 16 / 512 / sweep | metal_compiled | 53.1 [46.1, 64.1] | VERIFIED |
| mps | forward | 50 / 16 / 512 / sweep | mlx | 45.9 [36.0, 52.9] | VERIFIED |
| mps | forward | 50 / 16 / 512 / sweep | mlx_compiled | 49.5 [43.2, 69.2] | VERIFIED |
| mps | forward | 50 / 16 / 512 / sweep | stock | 99.3 [81.6, 114.9] | VERIFIED |
| mps | scoring | 10 / 16 / 512 / sweep | compiled | 103.4 [78.4, 125.6] | VERIFIED |
| mps | scoring | 10 / 16 / 512 / sweep | metal | 55.6 [46.1, 65.1] | VERIFIED |
| mps | scoring | 10 / 16 / 512 / sweep | metal_compiled | 63.4 [50.0, 77.2] | VERIFIED |
| mps | scoring | 10 / 16 / 512 / sweep | mlx | 50.0 [40.0, 58.7] | VERIFIED |
| mps | scoring | 10 / 16 / 512 / sweep | mlx_compiled | 67.7 [51.6, 85.1] | VERIFIED |
| mps | scoring | 10 / 16 / 512 / sweep | stock | 87.1 [67.6, 105.8] | VERIFIED |
| mps | scoring | 100 / 1 / 512 / sweep | compiled | 98.1 [86.4, 120.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 1 / 512 / sweep | metal | 42.8 [24.8, 56.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 1 / 512 / sweep | metal_compiled | 68.5 [55.6, 77.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 1 / 512 / sweep | mlx | 39.6 [29.8, 59.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 1 / 512 / sweep | mlx_compiled | 64.5 [51.3, 72.9] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 1 / 512 / sweep | stock | 106.2 [68.4, 108.3] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 128 / sweep | compiled | 409.2 [341.9, 442.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 128 / sweep | metal | 238.2 [210.8, 288.3] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 128 / sweep | metal_compiled | 388.5 [320.5, 424.5] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 128 / sweep | mlx | 278.1 [229.6, 331.9] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 128 / sweep | mlx_compiled | 351.6 [310.5, 412.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 128 / sweep | stock | 370.2 [282.7, 409.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 256 / sweep | compiled | 206.8 [148.7, 260.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 256 / sweep | metal | 118.6 [114.8, 130.9] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 256 / sweep | metal_compiled | 129.0 [113.1, 147.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 256 / sweep | mlx | 130.2 [117.0, 140.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 256 / sweep | mlx_compiled | 152.5 [128.7, 199.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 256 / sweep | stock | 212.7 [167.6, 241.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 16 / 512 / primary | compiled | 102.1 [83.1, 115.5] | VERIFIED |
| mps | scoring | 100 / 16 / 512 / primary | metal | 37.1 [34.4, 42.7] | VERIFIED |
| mps | scoring | 100 / 16 / 512 / primary | metal_compiled | 52.5 [43.4, 55.2] | VERIFIED |
| mps | scoring | 100 / 16 / 512 / primary | mlx | 40.7 [34.0, 47.1] | VERIFIED |
| mps | scoring | 100 / 16 / 512 / primary | mlx_compiled | 48.9 [41.3, 68.3] | VERIFIED |
| mps | scoring | 100 / 16 / 512 / primary | stock | 87.4 [77.3, 96.3] | VERIFIED |
| mps | scoring | 100 / 16 / 512 / sweep | compiled | 94.8 [65.9, 117.1] | VERIFIED |
| mps | scoring | 100 / 16 / 512 / sweep | metal | 42.8 [38.8, 46.8] | VERIFIED |
| mps | scoring | 100 / 16 / 512 / sweep | metal_compiled | 44.3 [39.9, 52.6] | VERIFIED |
| mps | scoring | 100 / 16 / 512 / sweep | mlx | 46.7 [40.7, 51.2] | VERIFIED |
| mps | scoring | 100 / 16 / 512 / sweep | mlx_compiled | 50.7 [44.7, 61.3] | VERIFIED |
| mps | scoring | 100 / 16 / 512 / sweep | stock | 94.1 [71.7, 107.0] | VERIFIED |
| mps | scoring | 100 / 32 / 512 / sweep | compiled | 91.3 [80.0, 119.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 32 / 512 / sweep | metal | 47.6 [44.8, 54.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 32 / 512 / sweep | metal_compiled | 55.5 [46.9, 62.8] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 32 / 512 / sweep | mlx | 49.7 [43.6, 59.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 32 / 512 / sweep | mlx_compiled | 68.0 [49.3, 74.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 32 / 512 / sweep | stock | 90.4 [77.6, 101.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 64 / 512 / sweep | compiled | 99.9 [65.5, 107.9] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 64 / 512 / sweep | metal | 46.8 [43.1, 52.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 64 / 512 / sweep | metal_compiled | 58.3 [52.4, 75.1] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 64 / 512 / sweep | mlx | 51.5 [46.2, 66.3] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 64 / 512 / sweep | mlx_compiled | 63.2 [52.3, 86.6] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 64 / 512 / sweep | stock | 91.9 [83.8, 98.7] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 8 / 512 / sweep | compiled | 109.6 [86.7, 121.2] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 8 / 512 / sweep | metal | 24.0 [22.6, 28.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 8 / 512 / sweep | metal_compiled | 24.1 [22.3, 29.4] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 8 / 512 / sweep | mlx | 28.6 [21.9, 32.5] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 8 / 512 / sweep | mlx_compiled | 23.4 [20.8, 29.3] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 100 / 8 / 512 / sweep | stock | 99.7 [89.2, 116.0] | DIAGNOSTIC_B_OR_L_NOT_FULLY_AUDITED |
| mps | scoring | 50 / 16 / 512 / sweep | compiled | 110.1 [87.2, 119.3] | VERIFIED |
| mps | scoring | 50 / 16 / 512 / sweep | metal | 40.9 [38.9, 43.4] | VERIFIED |
| mps | scoring | 50 / 16 / 512 / sweep | metal_compiled | 49.6 [44.2, 58.4] | VERIFIED |
| mps | scoring | 50 / 16 / 512 / sweep | mlx | 39.2 [33.1, 48.7] | VERIFIED |
| mps | scoring | 50 / 16 / 512 / sweep | mlx_compiled | 46.6 [39.8, 61.0] | VERIFIED |
| mps | scoring | 50 / 16 / 512 / sweep | stock | 94.6 [69.2, 101.6] | VERIFIED |
