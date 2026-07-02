# HBF memory model simulation results

This report presents the findings from the automated sweeps evaluating HBF memory models.

## 1. Maximum Per-GPU Batch Size (Figure 3 Replication)

| Model | Workload | Memory Config | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU |
|---|---|---|---|---|---|---|---|
| llama3_405B | SHORT | HBM4 | 0.0 (0.00x) | 0.0 (0.00x) | 99.2 (0.51x) | 194.8 (1.00x) | 242.6 (1.25x) |
| llama3_405B | SHORT | HBF | 97.0 (0.50x) | 387.0 (1.99x) | 374.5 (1.92x) | 329.8 (1.69x) | 184.1 (0.95x) |
| llama3_405B | SHORT | HBF+ | 235.0 (1.21x) | 491.0 (2.52x) | 490.0 (2.52x) | 454.5 (2.33x) | 375.2 (1.93x) |
| llama3_405B | SHORT | CONV | 0.0 (0.00x) | 0.0 (0.00x) | 19.0 (0.10x) | 83.0 (0.43x) | 91.0 (0.47x) |
| llama3_405B | SHORT | CONV+ | 0.0 (0.00x) | 0.0 (0.00x) | 46.8 (0.24x) | 114.8 (0.59x) | 121.6 (0.62x) |
| llama3_405B | MID | HBM4 | 0.0 (0.00x) | 0.0 (0.00x) | 31.5 (0.51x) | 61.9 (1.00x) | 77.1 (1.25x) |
| llama3_405B | MID | HBF | 39.0 (0.63x) | 159.0 (2.57x) | 191.8 (3.10x) | 167.5 (2.71x) | 167.8 (2.71x) |
| llama3_405B | MID | HBF+ | 81.0 (1.31x) | 217.5 (3.52x) | 235.2 (3.80x) | 221.8 (3.58x) | 176.0 (2.84x) |
| llama3_405B | MID | CONV | 0.0 (0.00x) | 0.0 (0.00x) | 7.0 (0.11x) | 34.0 (0.55x) | 42.6 (0.69x) |
| llama3_405B | MID | CONV+ | 0.0 (0.00x) | 0.0 (0.00x) | 16.5 (0.27x) | 44.2 (0.72x) | 53.1 (0.86x) |
| llama3_405B | LONG | HBM4 | 0.0 (0.00x) | 0.0 (0.00x) | 1.8 (0.47x) | 3.8 (1.00x) | 4.7 (1.25x) |
| llama3_405B | LONG | HBF | 2.0 (0.53x) | 11.0 (2.93x) | 14.5 (3.87x) | 15.9 (4.23x) | 16.2 (4.33x) |
| llama3_405B | LONG | HBF+ | 5.0 (1.33x) | 13.5 (3.60x) | 17.0 (4.53x) | 18.6 (4.97x) | 19.0 (5.07x) |
| llama3_405B | LONG | CONV | 0.0 (0.00x) | 0.0 (0.00x) | 0.2 (0.07x) | 2.1 (0.57x) | 3.1 (0.82x) |
| llama3_405B | LONG | CONV+ | 0.0 (0.00x) | 0.0 (0.00x) | 1.0 (0.27x) | 2.8 (0.73x) | 3.6 (0.97x) |
| llama4_maverick | SHORT | HBM4 | 0.0 (0.00x) | 0.0 (0.00x) | 269.5 (0.52x) | 514.9 (1.00x) | 535.1 (1.04x) |
| llama4_maverick | SHORT | HBF | 512.0 (0.99x) | 1130.5 (2.20x) | 1530.5 (2.97x) | 1024.0 (1.99x) | 653.4 (1.27x) |
| llama4_maverick | SHORT | HBF+ | 730.0 (1.42x) | 1557.5 (3.03x) | 1847.5 (3.59x) | 1024.5 (1.99x) | 1744.2 (3.39x) |
| llama4_maverick | SHORT | CONV | 40.0 (0.08x) | 61.5 (0.12x) | 123.8 (0.24x) | 310.2 (0.60x) | 410.4 (0.80x) |
| llama4_maverick | SHORT | CONV+ | 48.0 (0.09x) | 76.0 (0.15x) | 171.2 (0.33x) | 397.6 (0.77x) | 491.6 (0.95x) |
| llama4_maverick | MID | HBM4 | 0.0 (0.00x) | 0.0 (0.00x) | 85.5 (0.52x) | 164.8 (1.00x) | 203.7 (1.24x) |
| llama4_maverick | MID | HBF | 281.0 (1.71x) | 447.5 (2.72x) | 593.0 (3.60x) | 670.4 (4.07x) | 649.1 (3.94x) |
| llama4_maverick | MID | HBF+ | 351.0 (2.13x) | 565.5 (3.43x) | 727.0 (4.41x) | 793.2 (4.81x) | 542.8 (3.29x) |
| llama4_maverick | MID | CONV | 33.0 (0.20x) | 46.0 (0.28x) | 68.8 (0.42x) | 110.4 (0.67x) | 147.9 (0.90x) |
| llama4_maverick | MID | CONV+ | 39.0 (0.24x) | 55.5 (0.34x) | 84.2 (0.51x) | 136.6 (0.83x) | 175.8 (1.07x) |
| llama4_maverick | LONG | HBM4 | 0.0 (0.00x) | 0.0 (0.00x) | 16.8 (0.51x) | 33.0 (1.00x) | 40.8 (1.23x) |
| llama4_maverick | LONG | HBF | 86.0 (2.61x) | 99.0 (3.00x) | 112.0 (3.39x) | 150.9 (4.57x) | 157.5 (4.77x) |
| llama4_maverick | LONG | HBF+ | 100.0 (3.03x) | 116.0 (3.52x) | 130.8 (3.96x) | 176.1 (5.34x) | 182.8 (5.54x) |
| llama4_maverick | LONG | CONV | 16.0 (0.48x) | 19.5 (0.59x) | 22.2 (0.67x) | 31.4 (0.95x) | 35.3 (1.07x) |
| llama4_maverick | LONG | CONV+ | 19.0 (0.58x) | 22.5 (0.68x) | 26.0 (0.79x) | 37.0 (1.12x) | 41.8 (1.27x) |

## 2. System Throughput (Figure 4 Replication)

| Model | Workload | Memory Config | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU |
|---|---|---|---|---|---|---|---|
| llama3_405B | SHORT | HBM4 | 0.00 (0.00x) | 0.00 (0.00x) | 3492.04 (0.90x) | 3884.99 (1.00x) | 2846.05 (0.73x) |
| llama3_405B | SHORT | HBF | 970.42 (0.25x) | 3870.51 (1.00x) | 3816.23 (0.98x) | 3434.63 (0.88x) | 2625.76 (0.68x) |
| llama3_405B | SHORT | HBF+ | 2351.25 (0.61x) | 4910.85 (1.26x) | 4910.20 (1.26x) | 4885.14 (1.26x) | 4373.63 (1.13x) |
| llama3_405B | SHORT | CONV | 0.00 (0.00x) | 0.00 (0.00x) | 190.11 (0.05x) | 830.08 (0.21x) | 910.21 (0.23x) |
| llama3_405B | SHORT | CONV+ | 0.00 (0.00x) | 0.00 (0.00x) | 467.55 (0.12x) | 1147.84 (0.30x) | 1216.03 (0.31x) |
| llama3_405B | MID | HBM4 | 0.00 (0.00x) | 0.00 (0.00x) | 1234.47 (0.60x) | 2046.43 (1.00x) | 1841.07 (0.90x) |
| llama3_405B | MID | HBF | 391.39 (0.19x) | 1590.77 (0.78x) | 1918.05 (0.94x) | 1851.36 (0.90x) | 1852.12 (0.91x) |
| llama3_405B | MID | HBF+ | 811.55 (0.40x) | 2175.23 (1.06x) | 2354.32 (1.15x) | 2344.68 (1.15x) | 2300.67 (1.12x) |
| llama3_405B | MID | CONV | 0.00 (0.00x) | 0.00 (0.00x) | 70.05 (0.03x) | 340.12 (0.17x) | 426.35 (0.21x) |
| llama3_405B | MID | CONV+ | 0.00 (0.00x) | 0.00 (0.00x) | 165.47 (0.08x) | 442.74 (0.22x) | 531.30 (0.26x) |
| llama3_405B | LONG | HBM4 | 0.00 (0.00x) | 0.00 (0.00x) | 72.37 (0.51x) | 142.10 (1.00x) | 166.13 (1.17x) |
| llama3_405B | LONG | HBF | 20.81 (0.15x) | 110.82 (0.78x) | 145.01 (1.02x) | 159.49 (1.12x) | 162.65 (1.14x) |
| llama3_405B | LONG | HBF+ | 50.44 (0.35x) | 137.07 (0.96x) | 171.77 (1.21x) | 187.05 (1.32x) | 190.40 (1.34x) |
| llama3_405B | LONG | CONV | 0.00 (0.00x) | 0.00 (0.00x) | 2.61 (0.02x) | 21.81 (0.15x) | 30.86 (0.22x) |
| llama3_405B | LONG | CONV+ | 0.00 (0.00x) | 0.00 (0.00x) | 10.05 (0.07x) | 27.90 (0.20x) | 36.59 (0.26x) |
| llama4_maverick | SHORT | HBM4 | 0.00 (0.00x) | 0.00 (0.00x) | 11054.43 (0.56x) | 19866.61 (1.00x) | 19943.04 (1.00x) |
| llama4_maverick | SHORT | HBF | 5123.77 (0.26x) | 11306.94 (0.57x) | 15306.38 (0.77x) | 16135.70 (0.81x) | 15246.12 (0.77x) |
| llama4_maverick | SHORT | HBF+ | 7307.63 (0.37x) | 15575.45 (0.78x) | 19404.41 (0.98x) | 19786.59 (1.00x) | 18638.03 (0.94x) |
| llama4_maverick | SHORT | CONV | 403.00 (0.02x) | 617.05 (0.03x) | 1239.44 (0.06x) | 3102.51 (0.16x) | 4151.35 (0.21x) |
| llama4_maverick | SHORT | CONV+ | 484.10 (0.02x) | 762.57 (0.04x) | 1714.88 (0.09x) | 3976.69 (0.20x) | 5028.94 (0.25x) |
| llama4_maverick | MID | HBM4 | 0.00 (0.00x) | 0.00 (0.00x) | 4206.79 (0.61x) | 6938.23 (1.00x) | 8232.37 (1.19x) |
| llama4_maverick | MID | HBF | 2813.57 (0.41x) | 4478.48 (0.65x) | 5930.02 (0.85x) | 6703.92 (0.97x) | 6778.60 (0.98x) |
| llama4_maverick | MID | HBF+ | 3517.72 (0.51x) | 5658.05 (0.82x) | 7271.23 (1.05x) | 8003.78 (1.15x) | 7944.34 (1.15x) |
| llama4_maverick | MID | CONV | 334.17 (0.05x) | 463.62 (0.07x) | 687.97 (0.10x) | 1104.27 (0.16x) | 1478.78 (0.21x) |
| llama4_maverick | MID | CONV+ | 397.13 (0.06x) | 559.91 (0.08x) | 845.12 (0.12x) | 1366.33 (0.20x) | 1769.01 (0.25x) |
| llama4_maverick | LONG | HBM4 | 0.00 (0.00x) | 0.00 (0.00x) | 1206.66 (0.67x) | 1811.67 (1.00x) | 2086.56 (1.15x) |
| llama4_maverick | LONG | HBF | 862.61 (0.48x) | 993.48 (0.55x) | 1121.12 (0.62x) | 1508.94 (0.83x) | 1575.27 (0.87x) |
| llama4_maverick | LONG | HBF+ | 1003.07 (0.55x) | 1160.33 (0.64x) | 1308.87 (0.72x) | 1761.70 (0.97x) | 1832.90 (1.01x) |
| llama4_maverick | LONG | CONV | 163.29 (0.09x) | 195.19 (0.11x) | 223.15 (0.12x) | 314.48 (0.17x) | 353.24 (0.19x) |
| llama4_maverick | LONG | CONV+ | 191.50 (0.11x) | 227.14 (0.13x) | 260.37 (0.14x) | 370.10 (0.20x) | 417.65 (0.23x) |

## 3. Runtime Performance Breakdown (Figure 5 Replication)

Fractions of decode execution time spent on key components under a 0.1s TPOT SLO:

| Model | Workload | Memory | GPUs | Attention | FFN | KV Write | Communication | Others |
|---|---|---|---|---|---|---|---|---|
| llama3_405B | SHORT | HBM4 | 4 | 42.4% | 48.2% | 0.0% | 7.1% | 2.3% |
| llama3_405B | SHORT | HBM4 | 8 | 41.8% | 30.5% | 0.0% | 22.6% | 5.0% |
| llama3_405B | SHORT | HBM4 | 16 | 31.1% | 22.7% | 0.0% | 38.8% | 7.5% |
| llama3_405B | SHORT | HBF | 4 | 41.9% | 39.8% | 13.5% | 0.0% | 4.8% |
| llama3_405B | SHORT | HBF | 8 | 37.4% | 34.6% | 13.0% | 6.5% | 8.6% |
| llama3_405B | SHORT | HBF | 16 | 31.2% | 26.4% | 13.9% | 15.3% | 13.2% |
| llama3_405B | SHORT | HBF+ | 4 | 47.7% | 37.0% | 14.2% | 0.0% | 1.1% |
| llama3_405B | SHORT | HBF+ | 8 | 47.5% | 36.8% | 14.6% | 0.0% | 1.1% |
| llama3_405B | SHORT | HBF+ | 16 | 42.5% | 33.0% | 14.4% | 8.3% | 1.8% |
| llama3_405B | SHORT | CONV | 4 | 23.4% | 68.4% | 6.9% | 0.6% | 0.8% |
| llama3_405B | SHORT | CONV | 8 | 43.0% | 36.3% | 8.8% | 5.2% | 6.8% |
| llama3_405B | SHORT | CONV | 16 | 43.1% | 19.8% | 9.0% | 13.2% | 15.0% |
| llama3_405B | SHORT | CONV+ | 4 | 30.8% | 59.5% | 7.5% | 1.1% | 1.2% |
| llama3_405B | SHORT | CONV+ | 8 | 49.4% | 30.0% | 9.2% | 6.9% | 4.4% |
| llama3_405B | SHORT | CONV+ | 16 | 49.0% | 15.4% | 9.5% | 17.1% | 9.0% |
| llama3_405B | MID | HBM4 | 4 | 44.6% | 51.5% | 0.0% | 3.1% | 0.8% |
| llama3_405B | MID | HBM4 | 8 | 60.7% | 23.4% | 0.0% | 13.3% | 2.6% |
| llama3_405B | MID | HBM4 | 16 | 53.3% | 14.6% | 0.0% | 27.3% | 4.8% |
| llama3_405B | MID | HBF | 4 | 56.0% | 19.7% | 15.7% | 3.7% | 4.8% |
| llama3_405B | MID | HBF | 8 | 54.6% | 21.1% | 16.0% | 3.6% | 4.6% |
| llama3_405B | MID | HBF | 16 | 54.6% | 21.1% | 16.0% | 3.6% | 4.6% |
| llama3_405B | MID | HBF+ | 4 | 60.5% | 17.8% | 16.2% | 4.5% | 1.0% |
| llama3_405B | MID | HBF+ | 8 | 60.2% | 17.7% | 16.5% | 4.5% | 1.0% |
| llama3_405B | MID | HBF+ | 16 | 59.2% | 17.3% | 17.9% | 4.5% | 1.1% |
| llama3_405B | MID | CONV | 4 | 24.2% | 68.1% | 6.9% | 0.3% | 0.5% |
| llama3_405B | MID | CONV | 8 | 50.7% | 35.1% | 9.0% | 2.5% | 2.8% |
| llama3_405B | MID | CONV | 16 | 58.0% | 18.5% | 9.6% | 6.9% | 7.0% |
| llama3_405B | MID | CONV+ | 4 | 31.8% | 59.6% | 7.4% | 0.5% | 0.7% |
| llama3_405B | MID | CONV+ | 8 | 55.7% | 30.0% | 9.3% | 3.0% | 1.9% |
| llama3_405B | MID | CONV+ | 16 | 62.4% | 15.3% | 9.9% | 8.3% | 4.2% |
| llama3_405B | LONG | HBM4 | 4 | 45.7% | 53.3% | 0.0% | 1.0% | 0.0% |
| llama3_405B | LONG | HBM4 | 8 | 72.2% | 24.6% | 0.0% | 3.1% | 0.2% |
| llama3_405B | LONG | HBM4 | 16 | 80.8% | 11.7% | 0.0% | 7.1% | 0.4% |
| llama3_405B | LONG | HBF | 4 | 72.7% | 15.3% | 11.2% | 0.5% | 0.4% |
| llama3_405B | LONG | HBF | 8 | 78.0% | 8.0% | 11.7% | 1.5% | 0.8% |
| llama3_405B | LONG | HBF | 16 | 78.8% | 4.3% | 11.8% | 3.5% | 1.6% |
| llama3_405B | LONG | HBF+ | 4 | 74.9% | 13.2% | 11.2% | 0.5% | 0.2% |
| llama3_405B | LONG | HBF+ | 8 | 79.8% | 6.7% | 11.6% | 1.6% | 0.3% |
| llama3_405B | LONG | HBF+ | 16 | 80.6% | 3.4% | 11.7% | 3.9% | 0.4% |
| llama3_405B | LONG | CONV | 4 | 21.8% | 70.9% | 6.7% | 0.2% | 0.4% |
| llama3_405B | LONG | CONV | 8 | 55.9% | 35.2% | 7.7% | 0.7% | 0.5% |
| llama3_405B | LONG | CONV | 16 | 71.8% | 17.6% | 8.1% | 1.8% | 0.6% |
| llama3_405B | LONG | CONV+ | 4 | 32.8% | 59.7% | 6.8% | 0.2% | 0.4% |
| llama3_405B | LONG | CONV+ | 8 | 60.6% | 30.4% | 7.7% | 0.8% | 0.5% |
| llama3_405B | LONG | CONV+ | 16 | 73.9% | 15.4% | 8.1% | 1.9% | 0.6% |
| llama4_maverick | SHORT | HBM4 | 4 | 36.0% | 63.6% | 0.0% | 0.0% | 0.4% |
| llama4_maverick | SHORT | HBM4 | 8 | 64.9% | 34.4% | 0.0% | 0.0% | 0.8% |
| llama4_maverick | SHORT | HBM4 | 16 | 65.1% | 19.4% | 0.0% | 14.7% | 0.8% |
| llama4_maverick | SHORT | HBF | 4 | 55.7% | 29.2% | 12.8% | 0.0% | 2.3% |
| llama4_maverick | SHORT | HBF | 8 | 58.8% | 25.6% | 13.2% | 0.0% | 2.4% |
| llama4_maverick | SHORT | HBF | 16 | 55.5% | 21.9% | 13.0% | 7.3% | 2.3% |
| llama4_maverick | SHORT | HBF+ | 4 | 60.8% | 22.3% | 15.4% | 1.0% | 0.5% |
| llama4_maverick | SHORT | HBF+ | 8 | 62.1% | 21.0% | 15.4% | 1.0% | 0.5% |
| llama4_maverick | SHORT | HBF+ | 16 | 58.5% | 11.4% | 14.9% | 14.3% | 0.9% |
| llama4_maverick | SHORT | CONV | 4 | 20.6% | 76.5% | 2.7% | 0.0% | 0.3% |
| llama4_maverick | SHORT | CONV | 8 | 50.1% | 44.9% | 4.3% | 0.0% | 0.8% |
| llama4_maverick | SHORT | CONV | 16 | 67.0% | 24.5% | 5.5% | 2.0% | 1.0% |
| llama4_maverick | SHORT | CONV+ | 4 | 24.7% | 72.1% | 2.9% | 0.0% | 0.3% |
| llama4_maverick | SHORT | CONV+ | 8 | 56.0% | 38.9% | 4.7% | 0.0% | 0.5% |
| llama4_maverick | SHORT | CONV+ | 16 | 70.8% | 20.5% | 5.7% | 2.4% | 0.6% |
| llama4_maverick | MID | HBM4 | 4 | 40.6% | 59.2% | 0.0% | 0.0% | 0.2% |
| llama4_maverick | MID | HBM4 | 8 | 66.4% | 33.3% | 0.0% | 0.0% | 0.3% |
| llama4_maverick | MID | HBM4 | 16 | 78.9% | 18.1% | 0.0% | 2.3% | 0.6% |
| llama4_maverick | MID | HBF | 4 | 62.7% | 23.3% | 13.1% | 0.0% | 0.9% |
| llama4_maverick | MID | HBF | 8 | 70.9% | 14.0% | 14.1% | 0.0% | 1.0% |
| llama4_maverick | MID | HBF | 16 | 71.7% | 9.8% | 14.3% | 3.2% | 1.0% |
| llama4_maverick | MID | HBF+ | 4 | 67.0% | 18.9% | 13.9% | 0.0% | 0.2% |
| llama4_maverick | MID | HBF+ | 8 | 73.7% | 10.4% | 15.2% | 0.4% | 0.2% |
| llama4_maverick | MID | HBF+ | 16 | 73.2% | 8.0% | 14.8% | 3.8% | 0.2% |
| llama4_maverick | MID | CONV | 4 | 33.6% | 62.9% | 3.3% | 0.0% | 0.2% |
| llama4_maverick | MID | CONV | 8 | 53.1% | 42.6% | 4.0% | 0.0% | 0.3% |
| llama4_maverick | MID | CONV | 16 | 70.9% | 22.8% | 5.2% | 0.3% | 0.7% |
| llama4_maverick | MID | CONV+ | 4 | 35.9% | 60.4% | 3.5% | 0.0% | 0.2% |
| llama4_maverick | MID | CONV+ | 8 | 57.4% | 38.1% | 4.3% | 0.0% | 0.2% |
| llama4_maverick | MID | CONV+ | 16 | 74.1% | 19.7% | 5.4% | 0.4% | 0.4% |
| llama4_maverick | LONG | HBM4 | 4 | 62.6% | 37.3% | 0.0% | 0.0% | 0.0% |
| llama4_maverick | LONG | HBM4 | 8 | 70.1% | 29.8% | 0.0% | 0.0% | 0.1% |
| llama4_maverick | LONG | HBM4 | 16 | 80.5% | 18.6% | 0.0% | 0.7% | 0.2% |
| llama4_maverick | LONG | HBF | 4 | 63.2% | 17.3% | 19.3% | 0.0% | 0.2% |
| llama4_maverick | LONG | HBF | 8 | 64.1% | 10.6% | 25.1% | 0.0% | 0.2% |
| llama4_maverick | LONG | HBF | 16 | 66.9% | 5.9% | 26.2% | 0.8% | 0.2% |
| llama4_maverick | LONG | HBF+ | 4 | 64.6% | 15.7% | 19.7% | 0.0% | 0.1% |
| llama4_maverick | LONG | HBF+ | 8 | 65.3% | 9.0% | 25.6% | 0.0% | 0.1% |
| llama4_maverick | LONG | HBF+ | 16 | 68.0% | 4.9% | 26.6% | 0.4% | 0.1% |
| llama4_maverick | LONG | CONV | 4 | 58.1% | 35.1% | 6.8% | 0.0% | 0.1% |
| llama4_maverick | LONG | CONV | 8 | 61.2% | 30.2% | 8.5% | 0.0% | 0.1% |
| llama4_maverick | LONG | CONV | 16 | 68.6% | 21.7% | 9.5% | 0.1% | 0.2% |
| llama4_maverick | LONG | CONV+ | 4 | 59.3% | 33.8% | 6.8% | 0.0% | 0.1% |
| llama4_maverick | LONG | CONV+ | 8 | 63.0% | 28.3% | 8.7% | 0.0% | 0.1% |
| llama4_maverick | LONG | CONV+ | 16 | 70.9% | 19.2% | 9.7% | 0.1% | 0.1% |

## 4. SLO Sensitivity Analysis (Figure 6 Replication)

| Model | Memory | SLO | Metric | 4 GPU | 8 GPU | 16 GPU |
|---|---|---|---|---|---|---|
| llama3_405B | HBM4 | 0.05s | Batch Size | 1.8 (0.47x) | 3.8 (1.00x) | 4.7 (1.25x) |
| llama3_405B | HBM4 | 0.05s | TPS/GPU | 72.37 (0.51x) | 142.10 (1.00x) | 166.13 (1.17x) |
| llama3_405B | HBM4 | 0.1s | Batch Size | 1.8 (0.47x) | 3.8 (1.00x) | 4.7 (1.25x) |
| llama3_405B | HBM4 | 0.1s | TPS/GPU | 72.37 (0.51x) | 142.10 (1.00x) | 166.13 (1.17x) |
| llama3_405B | HBM4 | 0.2s | Batch Size | 1.8 (0.47x) | 3.8 (1.00x) | 4.7 (1.25x) |
| llama3_405B | HBM4 | 0.2s | TPS/GPU | 72.37 (0.51x) | 142.10 (1.00x) | 166.13 (1.17x) |
| llama3_405B | HBM4 | Offline (24h) | Batch Size | 1.8 (0.47x) | 3.8 (1.00x) | 4.7 (1.25x) |
| llama3_405B | HBM4 | Offline (24h) | TPS/GPU | 72.37 (0.51x) | 142.10 (1.00x) | 166.13 (1.17x) |
| llama3_405B | HBF | 0.05s | Batch Size | 4.8 (1.27x) | 6.4 (1.70x) | 6.9 (1.85x) |
| llama3_405B | HBF | 0.05s | TPS/GPU | 95.94 (0.68x) | 128.19 (0.90x) | 139.04 (0.98x) |
| llama3_405B | HBF | 0.1s | Batch Size | 14.5 (3.87x) | 15.9 (4.23x) | 16.2 (4.33x) |
| llama3_405B | HBF | 0.1s | TPS/GPU | 145.01 (1.02x) | 159.49 (1.12x) | 162.65 (1.14x) |
| llama3_405B | HBF | 0.2s | Batch Size | 33.8 (9.00x) | 35.0 (9.33x) | 34.8 (9.27x) |
| llama3_405B | HBF | 0.2s | TPS/GPU | 168.99 (1.19x) | 175.18 (1.23x) | 173.78 (1.22x) |
| llama3_405B | HBF | Offline (24h) | Batch Size | 68.0 (18.13x) | 70.0 (18.67x) | 70.9 (18.92x) |
| llama3_405B | HBF | Offline (24h) | TPS/GPU | 180.30 (1.27x) | 182.59 (1.28x) | 178.09 (1.25x) |
| llama3_405B | HBF+ | 0.05s | Batch Size | 6.0 (1.60x) | 7.6 (2.03x) | 8.2 (2.18x) |
| llama3_405B | HBF+ | 0.05s | TPS/GPU | 120.83 (0.85x) | 153.08 (1.08x) | 163.93 (1.15x) |
| llama3_405B | HBF+ | 0.1s | Batch Size | 17.0 (4.53x) | 18.6 (4.97x) | 19.0 (5.07x) |
| llama3_405B | HBF+ | 0.1s | TPS/GPU | 171.77 (1.21x) | 187.05 (1.32x) | 190.40 (1.34x) |
| llama3_405B | HBF+ | 0.2s | Batch Size | 39.5 (10.53x) | 40.8 (10.87x) | 40.7 (10.85x) |
| llama3_405B | HBF+ | 0.2s | TPS/GPU | 197.66 (1.39x) | 204.10 (1.44x) | 203.68 (1.43x) |
| llama3_405B | HBF+ | Offline (24h) | Batch Size | 77.5 (20.67x) | 79.5 (21.20x) | 80.4 (21.45x) |
| llama3_405B | HBF+ | Offline (24h) | TPS/GPU | 209.35 (1.47x) | 212.02 (1.49x) | 208.30 (1.47x) |
| llama3_405B | CONV | 0.05s | Batch Size | 0.0 (0.00x) | 0.0 (0.00x) | 0.9 (0.23x) |
| llama3_405B | CONV | 0.05s | TPS/GPU | 0.00 (0.00x) | 0.00 (0.00x) | 17.60 (0.12x) |
| llama3_405B | CONV | 0.1s | Batch Size | 0.2 (0.07x) | 2.1 (0.57x) | 3.1 (0.82x) |
| llama3_405B | CONV | 0.1s | TPS/GPU | 2.61 (0.02x) | 21.81 (0.15x) | 30.86 (0.22x) |
| llama3_405B | CONV | 0.2s | Batch Size | 4.8 (1.27x) | 6.6 (1.77x) | 7.5 (2.00x) |
| llama3_405B | CONV | 0.2s | TPS/GPU | 24.13 (0.17x) | 33.33 (0.23x) | 37.52 (0.26x) |
| llama3_405B | CONV | Offline (24h) | Batch Size | 68.0 (18.13x) | 70.0 (18.67x) | 70.9 (18.92x) |
| llama3_405B | CONV | Offline (24h) | TPS/GPU | 42.02 (0.30x) | 42.99 (0.30x) | 43.23 (0.30x) |
| llama3_405B | CONV+ | 0.05s | Batch Size | 0.0 (0.00x) | 0.2 (0.07x) | 1.1 (0.30x) |
| llama3_405B | CONV+ | 0.05s | TPS/GPU | 0.00 (0.00x) | 5.06 (0.04x) | 22.68 (0.16x) |
| llama3_405B | CONV+ | 0.1s | Batch Size | 1.0 (0.27x) | 2.8 (0.73x) | 3.6 (0.97x) |
| llama3_405B | CONV+ | 0.1s | TPS/GPU | 10.05 (0.07x) | 27.90 (0.20x) | 36.59 (0.26x) |
| llama3_405B | CONV+ | 0.2s | Batch Size | 6.0 (1.60x) | 7.9 (2.10x) | 8.7 (2.32x) |
| llama3_405B | CONV+ | 0.2s | TPS/GPU | 30.36 (0.21x) | 39.50 (0.28x) | 43.61 (0.31x) |
| llama3_405B | CONV+ | Offline (24h) | Batch Size | 77.5 (20.67x) | 79.5 (21.20x) | 80.4 (21.45x) |
| llama3_405B | CONV+ | Offline (24h) | TPS/GPU | 48.43 (0.34x) | 49.43 (0.35x) | 49.69 (0.35x) |
| llama4_maverick | HBM4 | 0.05s | Batch Size | 16.8 (0.51x) | 33.0 (1.00x) | 40.8 (1.23x) |
| llama4_maverick | HBM4 | 0.05s | TPS/GPU | 1206.66 (0.67x) | 1811.67 (1.00x) | 2086.56 (1.15x) |
| llama4_maverick | HBM4 | 0.1s | Batch Size | 16.8 (0.51x) | 33.0 (1.00x) | 40.8 (1.23x) |
| llama4_maverick | HBM4 | 0.1s | TPS/GPU | 1206.66 (0.67x) | 1811.67 (1.00x) | 2086.56 (1.15x) |
| llama4_maverick | HBM4 | 0.2s | Batch Size | 16.8 (0.51x) | 33.0 (1.00x) | 40.8 (1.23x) |
| llama4_maverick | HBM4 | 0.2s | TPS/GPU | 1206.66 (0.67x) | 1811.67 (1.00x) | 2086.56 (1.15x) |
| llama4_maverick | HBM4 | Offline (24h) | Batch Size | 16.8 (0.51x) | 33.0 (1.00x) | 40.8 (1.23x) |
| llama4_maverick | HBM4 | Offline (24h) | TPS/GPU | 1206.66 (0.67x) | 1811.67 (1.00x) | 2086.56 (1.15x) |
| llama4_maverick | HBF | 0.05s | Batch Size | 49.5 (1.50x) | 68.4 (2.07x) | 73.9 (2.24x) |
| llama4_maverick | HBF | 0.05s | TPS/GPU | 993.55 (0.55x) | 1369.60 (0.76x) | 1483.28 (0.82x) |
| llama4_maverick | HBF | 0.1s | Batch Size | 112.0 (3.39x) | 150.9 (4.57x) | 157.5 (4.77x) |
| llama4_maverick | HBF | 0.1s | TPS/GPU | 1121.12 (0.62x) | 1508.94 (0.83x) | 1575.27 (0.87x) |
| llama4_maverick | HBF | 0.2s | Batch Size | 245.0 (7.42x) | 318.8 (9.66x) | 323.4 (9.80x) |
| llama4_maverick | HBF | 0.2s | TPS/GPU | 1225.68 (0.68x) | 1594.25 (0.88x) | 1620.94 (0.89x) |
| llama4_maverick | HBF | Offline (24h) | Batch Size | 580.2 (17.58x) | 596.4 (18.07x) | 520.8 (15.78x) |
| llama4_maverick | HBF | Offline (24h) | TPS/GPU | 1300.16 (0.72x) | 1633.73 (0.90x) | 1637.81 (0.90x) |
| llama4_maverick | HBF+ | 0.05s | Batch Size | 58.0 (1.76x) | 80.2 (2.43x) | 86.6 (2.62x) |
| llama4_maverick | HBF+ | 0.05s | TPS/GPU | 1160.55 (0.64x) | 1605.44 (0.89x) | 1736.24 (0.96x) |
| llama4_maverick | HBF+ | 0.1s | Batch Size | 130.8 (3.96x) | 176.1 (5.34x) | 182.8 (5.54x) |
| llama4_maverick | HBF+ | 0.1s | TPS/GPU | 1308.87 (0.72x) | 1761.70 (0.97x) | 1832.90 (1.01x) |
| llama4_maverick | HBF+ | 0.2s | Batch Size | 285.2 (8.64x) | 370.1 (11.22x) | 376.1 (11.40x) |
| llama4_maverick | HBF+ | 0.2s | TPS/GPU | 1427.26 (0.79x) | 1850.98 (1.02x) | 1881.35 (1.04x) |
| llama4_maverick | HBF+ | Offline (24h) | Batch Size | 660.8 (20.02x) | 676.4 (20.50x) | 580.8 (17.60x) |
| llama4_maverick | HBF+ | Offline (24h) | TPS/GPU | 1504.89 (0.83x) | 1528.61 (0.84x) | 1535.06 (0.85x) |
| llama4_maverick | CONV | 0.05s | Batch Size | 9.8 (0.30x) | 13.4 (0.41x) | 14.2 (0.43x) |
| llama4_maverick | CONV | 0.05s | TPS/GPU | 195.32 (0.11x) | 269.17 (0.15x) | 284.72 (0.16x) |
| llama4_maverick | CONV | 0.1s | Batch Size | 22.2 (0.67x) | 31.4 (0.95x) | 35.3 (1.07x) |
| llama4_maverick | CONV | 0.1s | TPS/GPU | 223.15 (0.12x) | 314.48 (0.17x) | 353.24 (0.19x) |
| llama4_maverick | CONV | 0.2s | Batch Size | 50.5 (1.53x) | 72.8 (2.20x) | 80.6 (2.44x) |
| llama4_maverick | CONV | 0.2s | TPS/GPU | 253.14 (0.14x) | 363.86 (0.20x) | 403.09 (0.22x) |
| llama4_maverick | CONV | Offline (24h) | Batch Size | 580.2 (17.58x) | 596.4 (18.07x) | 520.8 (15.78x) |
| llama4_maverick | CONV | Offline (24h) | TPS/GPU | 335.35 (0.19x) | 441.01 (0.24x) | 446.01 (0.25x) |
| llama4_maverick | CONV+ | 0.05s | Batch Size | 11.2 (0.34x) | 15.8 (0.48x) | 16.9 (0.51x) |
| llama4_maverick | CONV+ | 0.05s | TPS/GPU | 226.58 (0.13x) | 315.25 (0.17x) | 339.08 (0.19x) |
| llama4_maverick | CONV+ | 0.1s | Batch Size | 26.0 (0.79x) | 37.0 (1.12x) | 41.8 (1.27x) |
| llama4_maverick | CONV+ | 0.1s | TPS/GPU | 260.37 (0.14x) | 370.10 (0.20x) | 417.65 (0.23x) |
| llama4_maverick | CONV+ | 0.2s | Batch Size | 59.0 (1.79x) | 85.1 (2.58x) | 93.8 (2.84x) |
| llama4_maverick | CONV+ | 0.2s | TPS/GPU | 295.51 (0.16x) | 426.08 (0.24x) | 469.10 (0.26x) |
| llama4_maverick | CONV+ | Offline (24h) | Batch Size | 660.8 (20.02x) | 676.4 (20.50x) | 580.8 (17.60x) |
| llama4_maverick | CONV+ | Offline (24h) | TPS/GPU | 386.31 (0.21x) | 394.79 (0.22x) | 398.00 (0.22x) |

## 5. Write Traffic and Endurance Assessment (Figure 7 Replication)

Assuming continuous batching in steady state over a 3-year warranty lifespan:

| Model | Workload | Memory | Batch Size | TPS/GPU | Write rate/GPU (MB/s) | 3-Year PEC | Lifetime Status |
|---|---|---|---|---|---|---|---|
| llama3_405B | SHORT | HBF | 329.8 | 3434.63 | 9661.4 | 235157.2 | FAIL (Wear-out) |
| llama3_405B | SHORT | HBF+ | 454.5 | 4885.14 | 13741.6 | 295599.6 | FAIL (Wear-out) |
| llama3_405B | SHORT | CONV | 83.0 | 830.08 | 2335.0 | 56832.7 | PASS (Safe) |
| llama3_405B | SHORT | CONV+ | 114.8 | 1147.84 | 3228.8 | 69456.0 | PASS (Safe) |
| llama3_405B | MID | HBF | 167.5 | 1851.36 | 12252.8 | 298230.9 | FAIL (Wear-out) |
| llama3_405B | MID | HBF+ | 221.8 | 2344.68 | 15517.6 | 333804.7 | FAIL (Wear-out) |
| llama3_405B | MID | CONV | 34.0 | 340.12 | 2251.0 | 54788.6 | PASS (Safe) |
| llama3_405B | MID | CONV+ | 44.2 | 442.74 | 2930.1 | 63031.2 | PASS (Safe) |
| llama3_405B | LONG | HBF | 15.9 | 159.49 | 7827.3 | 190516.2 | FAIL (Wear-out) |
| llama3_405B | LONG | HBF+ | 18.6 | 187.05 | 9179.7 | 197466.8 | FAIL (Wear-out) |
| llama3_405B | LONG | CONV | 2.1 | 21.81 | 1070.3 | 26050.2 | PASS (Safe) |
| llama3_405B | LONG | CONV+ | 2.8 | 27.90 | 1369.3 | 29454.6 | PASS (Safe) |
| llama4_maverick | SHORT | HBF | 1024.0 | 16135.70 | 17290.9 | 420858.8 | FAIL (Wear-out) |
| llama4_maverick | SHORT | HBF+ | 1024.5 | 19786.59 | 21203.2 | 456108.7 | FAIL (Wear-out) |
| llama4_maverick | SHORT | CONV | 310.2 | 3102.51 | 3324.6 | 80921.0 | PASS (Safe) |
| llama4_maverick | SHORT | CONV+ | 397.6 | 3976.69 | 4261.4 | 91668.4 | PASS (Safe) |
| llama4_maverick | MID | HBF | 670.4 | 6703.92 | 16902.1 | 411396.3 | FAIL (Wear-out) |
| llama4_maverick | MID | HBF+ | 793.2 | 8003.78 | 20179.4 | 434085.7 | FAIL (Wear-out) |
| llama4_maverick | MID | CONV | 110.4 | 1104.27 | 2784.1 | 67765.1 | PASS (Safe) |
| llama4_maverick | MID | CONV+ | 136.6 | 1366.33 | 3444.8 | 74103.1 | PASS (Safe) |
| llama4_maverick | LONG | HBF | 150.9 | 1508.94 | 28210.6 | 686643.5 | FAIL (Wear-out) |
| llama4_maverick | LONG | HBF+ | 176.1 | 1761.70 | 32936.1 | 708499.1 | FAIL (Wear-out) |
| llama4_maverick | LONG | CONV | 31.4 | 314.48 | 5879.4 | 143104.0 | FAIL (Wear-out) |
| llama4_maverick | LONG | CONV+ | 37.0 | 370.10 | 6919.2 | 148840.9 | FAIL (Wear-out) |
