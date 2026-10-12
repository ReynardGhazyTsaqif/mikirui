# Laporan Fase 5 / pratinjau Fase 6 (2026-10-12 09:37)

Semua angka memakai OOF rata-rata 3 seed (aturan dari pelajaran P4).

## Model tunggal (bag 3 seed)

| Model | Mean-fold AUC | OOF KS |
|---|---|---|
| T_lgb_b3_4f3c7f | 0.83788 | 0.5138 |
| T_xgb_b3_565cf6 | 0.83700 | 0.5125 |
| T_lgb_selb2_4f3c7f | 0.83687 | 0.5140 |
| T_cat_b3_a6abab | 0.83666 | 0.5135 |
| E81 | 0.83559 | 0.5114 |
| E80 | 0.83544 | 0.5101 |
| E87 | 0.83537 | 0.5088 |
| E71b | 0.83499 | 0.5097 |
| E71c | 0.83490 | 0.5076 |
| E83 | 0.82910 | 0.4960 |
| E82 | 0.82314 | 0.4887 |

## Ensemble

- Nested CV Caruana: mean-fold AUC 0.83792, KS 0.5150 (vs tunggal terbaik T_lgb_b3_4f3c7f: 0.83788 / 0.5138)
- Bobot Caruana (semua fold): E71c 0.10, T_cat_b3_a6abab 0.20, T_xgb_b3_565cf6 0.10, T_lgb_b3_4f3c7f 0.50, T_lgb_selb2_4f3c7f 0.10
- Stabilitas KS nested blend - T_lgb_b3_4f3c7f: potongan 10% rata-rata -0.0001 (menang 50%), potongan 30% +0.0004 (menang 63%)

## File submission kandidat

- `P5_single_T_lgb_b3_4f3c7f.csv` (md5 e180da83b81212223ef969fb12a017b1): model tunggal terbaik, bag 3 seed
- `P5_caruana_nested0.83792.csv` (md5 d715936b96d8e127cc86e731583d1fb3): blend Caruana