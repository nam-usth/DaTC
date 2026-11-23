Table I. Retrieval recall (%), ties at their expected rank.

| upper block: reviewed settings | FB top-1 | FB top-5 | PRESIOUS top-1 | PRESIOUS top-5 |
|---|---|---|---|---|
| DaTC + local alignment | 55.0 | 66.7 | 36.2 | 80.9 |
| DaTC + cyclic DTW | 53.3 | 61.7 | 38.3 | 55.3 |
| (kappa,tau) + cyclic dDTW | 4.2 | 21.7 | 23.4 | 57.4 |
| rim length | 3.8 | 21.2 | 42.7 | 67.6 |
| rim planarity | 10.0 | 31.7 | 17.0 | 53.2 |

| lower block: tuned alike | test top-1 | test top-5 | held-out top-1 | PRESIOUS top-1 |
|---|---|---|---|---|
| DaTC rank code + local al. | 76.7 | 83.3 | 56.5 | 57.4 |
| (kappa,tau) tuned + cyclic DTW | 75.0 | 90.0 | 57.6 | 68.1 |
| (kappa,tau) tuned + local al. | 80.0 | 93.3 | 58.7 | 74.5 |

Object-level sign tests (wins:losses, objects with equal counts dropped)
- FB pool, DaTC vs (kappa,tau) + cyclic dDTW: 45:7, p = 0.0000
- FB pool, DaTC vs rim length: 36:12, p = 0.0007
- FB pool, DaTC vs rim planarity: 35:12, p = 0.0011
- test half, (kappa,tau) tuned + cyclic DTW vs DaTC: 7:3, p = 0.34
- test half, (kappa,tau) tuned + local al. vs DaTC: 6:2, p = 0.29
- held-out, DaTC vs (kappa,tau) tuned + cyclic DTW: 11:9, p = 0.82

Pair detection on the Fantastic Breaks pool
- DaTC + local alignment: ROC AUC 0.753, average precision 0.303
- rim length: ROC AUC 0.808, average precision 0.037

TABLE I REPRODUCED
