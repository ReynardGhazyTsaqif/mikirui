# Log Submission Kaggle

Isi `public_lb` setelah submit. Hanya file dengan MD5 yang tercatat di sini yang boleh dipilih sebagai final.

| Tanggal | Exp | File | MD5 | OOF AUC | Public LB | Catatan |
|---|---|---|---|---|---|---|
| 2026-10-10 | E00 | E00_constant.csv | 689dfa152a24710ff85854101f753962 | – | 0 | Deteksi metrik (konstanta 0,1358) → metrik berbasis ranking, bukan AUC; kandidat KS/Gini |
| 2026-10-10 | E01 | E01_lgb_oof0.64712.csv | 0f7654e7dc6c35b87722b46a91b6a6ae | 0.64712 | | B0: aplikasi + profil |
| 2026-10-10 | E02 | E02_lgb_oof0.64924.csv | b8051b7c32ca3e941eba5146f7ce8843 | 0.64924 | | + rasio (ditolak) |
| 2026-10-10 | E03 | E03_lgb_oof0.82554.csv | 80f70f7bdc19a98824b5e39bb65e90bc | 0.82554 | 0.52436 | B1: + agregat naif semua tabel (submit serius pertama) |
| 2026-10-10 | E31 | E31_lgb_oof0.83132_ks0.5001.csv | e059cbf896c3c5cf6666e144191343e6 | 0.83132 (KS 0.5001) | 0.53801 | B2: E03 + rp_core + ec_window + pa_burden (hasil Fase 2) |
| 2026-10-11 | E71b | E71b_lgb_176f_oof0.83249_ks0.5051.csv | a200fbd704ef50f1843b0f03e749254d | 0.83249 (KS 0.5051; 3 seed 0.5066) | 0.54082 | B3: 176 fitur hasil null importance dari semua grup |
| 2026-10-11 | P4 | P4_blend_gbdt5_cb2_oof0.83644_ks0.5118.csv | 6506f3409fdbb4718bee552eab221ade | 0.83644 mean-fold (KS 0.5118) | 0.53867 | Pratinjau blend rank: LGB B3 + LGB sel_b2 + LGB extra_trees + XGB + CatBoost x2, bobot tidak dioptimasi |
| 2026-10-11 | E80 | E80_catboost_oof0.83462_ks0.5092.csv | 5d13d173b4de4e413314aa6c7dd71851 | 0.83462 (KS 0.5092) | 0.54487 | Diagnostik: CatBoost tunggal, untuk menafsirkan P4 < E71b di public |
| 2026-10-12 | P5 single | P5_single_T_lgb_b3_4f3c7f.csv | e180da83b81212223ef969fb12a017b1 | 0.83788 mean-fold bag 3 seed (KS 0.5138) | 0.54685 | LightGBM tuned Optuna (S2), lr 0.02, bag 3 seed, B3 — public terbaik |
| 2026-10-12 | P5 caruana | P5_caruana_nested0.83792.csv | d715936b96d8e127cc86e731583d1fb3 | 0.83792 mean-fold nested (KS 0.5150) | 0.54435 | Caruana: LGB tuned 0.5, Cat tuned 0.2, XGB tuned 0.1, LGB selb2 0.1, E71c 0.1 |
| 2026-10-12 | E121 | E121_lgb_ecm_bag3_auc0.84261.csv | 19f57efb82c2eeb4c93e929d1cb4898b | 0.84261 mean-fold bag 3 seed (KS 0.5207) | 0.55567 | LGB tuned (batas iterasi 15000) + 7 fitur model level order e-commerce nested; audit label acak lolos. File LF |
| 2026-10-13 | E130a+E141 | E130a_E141_rank50_auc0.84353.csv | 4368b2155c965c1506f359b129d4d50a | 0.84353 mean-fold (KS 0.5230) | 0.55858 | Blend rank 50:50: LGB tuned set E130a (528 kolom + ecm, bag 3) + LGB DART 5000 ronde (bag 3); Spearman test 0.9985. File LF |
