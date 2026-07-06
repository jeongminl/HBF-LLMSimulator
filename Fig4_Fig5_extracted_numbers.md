# Extracted Numbers — Figures 4 & 5

Source: Kyung, Moon, Cho, Ahn, "High-Bandwidth Flash for KV Caches: Endurance and Performance Implications," IEEE Computer Architecture Letters, 2026 (DOI 10.1109/LCA.2026.3695938).

---

## Figure 4 — HBF Lifetime Heatmaps (Llama 4 Maverick & DeepSeek-R1)

Each cell = **HBF lifetime in years** with **average TPOT (ms) in parentheses**, for a given average context length (8K/16K/32K) and average `L_in:L_out` ratio, under low dispersion (CV=0.1, κ=90) and high dispersion (CV=0.3, κ=30).

Column-header row = **normalized lifetime** relative to the 8K case (per model/dispersion setting).

### Llama 4 Maverick — CV = 0.1, κ = 90
Normalized lifetime (8K/16K/32K): **1.00 / 2.04 / 4.14**

| L_in:L_out | 8K | 16K | 32K |
|---|---|---|---|
| 1:3 | 6.2 (52) | 12.8 (52) | 26.3 (53) |
| 1:1 | 4.7 (60) | 9.8 (59) | 19.1 (59) |
| 3:1 | 2.7 (68) | 5.3 (67) | 11.4 (65) |
| 7:1 | 1.4 (75) | 2.9 (73) | 5.6 (71) |
| 15:1 | 0.8 (80) | 1.5 (79) | 3.0 (75) |

### Llama 4 Maverick — CV = 0.3, κ = 30
Normalized lifetime (8K/16K/32K): **1.00 / 1.92 / 4.22**

| L_in:L_out | 8K | 16K | 32K |
|---|---|---|---|
| 1:3 | 6.6 (54) | 13.1 (54) | 26.7 (55) |
| 1:1 | 5.1 (61) | 9.7 (61) | 22.3 (61) |
| 3:1 | 2.9 (72) | 5.2 (70) | 13.1 (69) |
| 7:1 | 1.5 (79) | 3.0 (77) | 6.0 (74) |
| 15:1 | 0.8 (84) | 1.6 (82) | 3.4 (79) |

### DeepSeek-R1 — CV = 0.1, κ = 90
Normalized lifetime (8K/16K/32K): **1.00 / 1.70 / 3.37**

| L_in:L_out | 8K | 16K | 32K |
|---|---|---|---|
| 1:3 | 10.0 (70) | 17.5 (59) | 34.3 (57) |
| 1:1 | 7.4 (78) | 12.0 (65) | 24.2 (61) |
| 3:1 | 4.0 (82) | 6.9 (76) | 13.7 (72) |
| 7:1 | 2.0 (87) | 3.5 (76) | 7.2 (72) |
| 15:1 | 1.1 (91) | 1.8 (80) | 3.5 (75) |

### DeepSeek-R1 — CV = 0.3, κ = 30
Normalized lifetime (8K/16K/32K): **1.00 / 1.76 / 3.31**

| L_in:L_out | 8K | 16K | 32K |
|---|---|---|---|
| 1:3 | 10.1 (72) | 17.9 (62) | 34.1 (58) |
| 1:1 | 7.4 (78) | 13.4 (67) | 25.5 (64) |
| 3:1 | 4.2 (85) | 7.0 (74) | 12.3 (70) |
| 7:1 | 2.1 (89) | 3.8 (79) | 7.0 (75) |
| 15:1 | 1.1 (94) | 1.9 (83) | 3.8 (78) |

Target warranty period referenced in the paper: **5 years**.

---

## Figure 5 — Token Generation Throughput (Llama 4 Maverick & DeepSeek-R1)

All cases use `L_in:L_out = 1:3`. Device configurations per group: **NVLink5.0**, **NVLink6.0** (both device_HBM, differing only in NVLink-C2C generation to CPU memory), **HBF** (full bandwidth), **1/2-HBF** (half bandwidth).

- Bars outlined in red = **SLO-adjusted throughput** (TPOT was capped/throttled to meet the 200 ms SLO; the paper does not print a distinct TPOT number for these bars since it is pinned at the SLO).
- Number printed above a non-SLO-bound bar = **TPOT (ms)** for that configuration.
- The two-decimal numbers under NVLink5.0/NVLink6.0 (e.g. "1.6", "3.2") = **CPU memory KV cache capacity relative to HBM-resident KV cache** (marked `***` in the figure).

### Baseline throughputs (unnormalized)
| Model | Context length | Baseline throughput |
|---|---|---|
| Llama 4 Maverick | 8K | 9,843 tokens/s |
| Llama 4 Maverick | 32K | 2,467 tokens/s |
| DeepSeek-R1 | 8K | 20,483 tokens/s |
| DeepSeek-R1 | 32K | 5,566 tokens/s |

### SRAM size per device (fixed weights+KV capacity cases)
| Model | Weights+reserved-KV capacity | Context length | SRAM/device |
|---|---|---|---|
| Llama 4 Maverick | 512 GB | 8K | 206 MB |
| Llama 4 Maverick | 768 GB | 8K | 308 MB |
| Llama 4 Maverick | 512 GB | 32K | 89 MB |
| Llama 4 Maverick | 768 GB | 32K | 114 MB |
| DeepSeek-R1 | 512 GB | 8K | 265 MB |
| DeepSeek-R1 | 768 GB | 8K | 431 MB |
| DeepSeek-R1 | 512 GB | 32K | 104 MB |
| DeepSeek-R1 | 768 GB | 32K | 145 MB |

### Per-bar data

**Llama 4 Maverick, 512 GB, Context length = 8K** (CPU-KV ratios: NVLink5.0=1.6, NVLink6.0=1.8)
| Device | SLO-bound (red outline)? | TPOT (ms) |
|---|---|---|
| NVLink5.0 | Yes | — (capped at SLO) |
| NVLink6.0 | No | 124 |
| HBF | No | 51 |
| 1/2-HBF | No | 100 |

**Llama 4 Maverick, 768 GB, Context length = 8K** (CPU-KV ratios: NVLink5.0=1.6, NVLink6.0=3.2)
| Device | SLO-bound? | TPOT (ms) |
|---|---|---|
| NVLink5.0 | Yes | — |
| NVLink6.0 | Yes | — |
| HBF | No | 75 |
| 1/2-HBF | No | 144 |

**Llama 4 Maverick, 512 GB, Context length = 32K** (CPU-KV ratios: NVLink5.0=1.7, NVLink6.0=1.8)
| Device | SLO-bound? | TPOT (ms) |
|---|---|---|
| NVLink5.0 | Yes | — |
| NVLink6.0 | No | 122 |
| HBF | No | 49 |
| 1/2-HBF | No | 97 |

**Llama 4 Maverick, 768 GB, Context length = 32K** (CPU-KV ratios: NVLink5.0=1.7, NVLink6.0=3.3)
| Device | SLO-bound? | TPOT (ms) |
|---|---|---|
| NVLink5.0 | Yes | — |
| NVLink6.0 | Yes | — |
| HBF | No | 72 |
| 1/2-HBF | No | 142 |

**DeepSeek-R1, 512 GB, Context length = 8K** (CPU-KV ratios: NVLink5.0=2.7, NVLink6.0=3.4)
| Device | SLO-bound? | TPOT (ms) |
|---|---|---|
| NVLink5.0 | Yes | — |
| NVLink6.0 | No | 153 |
| HBF | No | 81 |
| 1/2-HBF | No | 130 |

**DeepSeek-R1, 768 GB, Context length = 8K** (CPU-KV ratios: NVLink5.0=2.7, NVLink6.0=4.8)
| Device | SLO-bound? | TPOT (ms) |
|---|---|---|
| NVLink5.0 | Yes | — |
| NVLink6.0 | Yes | — |
| HBF | No | 129 |
| 1/2-HBF | **Yes** | — |

**DeepSeek-R1, 512 GB, Context length = 32K** (CPU-KV ratios: NVLink5.0=3.1, NVLink6.0=3.5)
| Device | SLO-bound? | TPOT (ms) |
|---|---|---|
| NVLink5.0 | Yes | — |
| NVLink6.0 | No | 130 |
| HBF | No | 57 |
| 1/2-HBF | No | 108 |

**DeepSeek-R1, 768 GB, Context length = 32K** (CPU-KV ratios: NVLink5.0=3.1, NVLink6.0=6.0)
| Device | SLO-bound? | TPOT (ms) |
|---|---|---|
| NVLink5.0 | Yes | — |
| NVLink6.0 | Yes | — |
| HBF | No | 83 |
| 1/2-HBF | No | 155 |

**Notes on Figure 5:**
- Exact numeric values for the bar heights themselves ("Normalized throughput," y-axis 0–6) are not printed in the figure — only visually encoded. The TPOT (ms) numbers above bars and the tables above are the only precisely labeled quantities.
- The 768 GB case renders NVLink6.0 SLO-bound for both models at both context lengths (matches paper text: "increased KV cache offloading again renders the system SLO-bound").
- DeepSeek-R1 at 768 GB / 8K context is the one case where even 1/2-HBF becomes SLO-bound (red-outlined), unlike all other HBF/1-2-HBF bars in the figure.
