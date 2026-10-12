# Laporan Fase 5 / pratinjau Fase 6 (2026-10-11 22:27)

Semua angka memakai OOF rata-rata 3 seed (aturan dari pelajaran P4).

## Model tunggal (bag 3 seed)

| Model | Mean-fold AUC | OOF KS |
|---|---|---|
| T_lgb_b3_effea5 | 0.82987 | 0.5124 |
| T_lgb_selb2_effea5 | 0.82962 | 0.5092 |
| T_xgb_b3_dada8a | 0.82836 | 0.5141 |
| T_cat_b3_230dba | 0.82688 | 0.5025 |

## Ensemble

- Nested CV Caruana: mean-fold AUC 0.83052, KS 0.5140 (vs tunggal terbaik T_lgb_b3_effea5: 0.82987 / 0.5124)
- Bobot Caruana (semua fold): T_cat_b3_230dba 0.25, T_xgb_b3_dada8a 0.25, T_lgb_b3_effea5 0.25, T_lgb_selb2_effea5 0.25
- Stabilitas KS nested blend - T_lgb_b3_effea5: potongan 10% rata-rata +0.0011 (menang 53%), potongan 30% -0.0000 (menang 51%)