# PROMPT 3 — Menggabungkan kenaikan kecil (dikerjakan mulai 18 Okt sore, sesi eksplorasi)

Sumber perintah: `PROMPT_3_gabungan_kenaikan_kecil.md` (versi 18 Okt 15.15: kuota 3 submission/hari untuk kandidat kuat; aturan pemilihan final berbasis `nilai_harapan`). Aturan keputusan: bootstrap berpasangan berstrata per fold (1000×), Diterima P ≥ 0,95 dan Δ ≥ +0,0003, Kandidat P 0,80–0,95, Ditolak P < 0,80; ensemble hanya dengan OOF rata-rata ≥ 3 seed.

## Langkah 1 — Inventaris komponen (diterima / kandidat)

Δ solo = selisih mean-fold AUC (OOF rata-rata 3 seed kecuali disebut lain) terhadap acuan saat diukur; keputusan menurut aturan baru (penilaian ulang 18 Okt, `experiments/explore_17okt/reevaluasi_aturan_baru.csv`).

| komponen | jenis | Δ solo | acuan saat diukur | seed | fold naik | file | repro |
|---|---|---|---|---|---|---|---|
| Set 528 kolom (E130a) | fitur | 0,84314 (+0,00053 vs E121 basis B3 176 kolom) | E121 | 3 | – | `features/*_v1.parquet` (GROUPS) | CPU-deterministik |
| ecm, model level order e-commerce (E121) | fitur nested | +0,0045 (0,84261 vs 0,83806) | T_lgb_b3_4f3c7f_r15k | 3 vs 5 | – | `features/e121_ordermodel_v1.npz` | CPU-deterministik |
| Seleksi null importance 333 kolom (E130b) | seleksi | +0,00036 (P 0,996) — Diterima | E130a | 3 | 3/5 | featset E130b | CPU-deterministik (seleksi memakai target seluruh train → sedikit optimis) |
| seq_gru/bigru/tf persentil (E122 → E122c) | fitur nested | E122 +0,00097 vs E130a; E122c +0,00044 vs E122 (P 0,999) — Diterima | E130a / E122 | 3 | 5/5 | `e122_gru_v1r.npz`, `_bigrur`, `_tfr` | **GPU** |
| seq persentil 1 kolom / rata-rata 3 varian (E122r, E122m) | fitur nested | +0,00019 / +0,00023 — Kandidat | E122 | 3 | 4/5 | `e122_gru_v1r.npz` | **GPU** |
| seqx_rp/ab/pa (E122x1) | fitur nested | +0,00041 (P 0,992) — Diterima | E122c | 3 | 4/5 | `e122x_{rp,ab,pa}_v1r.npz` | **GPU** |
| seqx_all (E122x2) | fitur nested | +0,00033 (P 0,985) — Diterima; E122x3 (keduanya) +0,00041 | E122c | 3 | 5/5 | `e122x_all_v1r.npz` | **GPU** |
| DART set E130a (E141) | model | +0,00038 (P 0,973) — Diterima | E130a | 3 | 5/5 | `oof/E141_s*` | CPU-deterministik |
| DART set E122x1 (E141x) | model | +0,00020 (P 0,857) — Kandidat | E122x1 | 3 | 3/5 | `oof/E141x_s*` | GPU (fitur) |
| XGB tuned set E130a (E150_xgb) | model | 0,84230 (−0,00084 vs E130a); blend +0,0001 | E130a | 3 | – | `oof/E150_xgb_s*` | CPU (cek determinisme hist) |
| Optuna ulang fold 0–2 (E146) | model/parameter | −0,00001 — Ditolak (disimpan untuk ensemble) | E130a | 3 | 2/5 | `oof/E146_s*` | CPU-deterministik |
| Batas iterasi 15000 (E140a) | pelatihan | B3 +0,00009 (Kandidat), sel_b2 +0,00041 (Diterima); sudah default | T_lgb_* | 3 | 3/5, 4/5 | – | CPU-deterministik |
| 5 seed (E144) | pelatihan | +0,0001 | E140a | 5 | – | – | CPU-deterministik |
| NN RealMLP / TabM versi `_iv` (E142, fitur E130a + seq + seqx) | model | TabM 0,84476, RealMLP 0,84424 (sendiri); blend 50% dengan E122x1 +0,0008 di CV tetapi lebih rendah di public | – | 5 | – | `oof/E142_*_seqx_iv_s*` | **GPU** |
| **rp_ev rantai bayar event (E200)** | fitur | **+0,00163** (P 1,000) — Diterima | E122x1 | 3 | 5/5 | `features/rp_events_v1.parquet` | CPU-deterministik |
| **10-fold bersarang (E202)** | pelatihan | **+0,00051** (P 0,999) — Diterima | E200 | 3 | 4/5 | `oof/E202_s*` | CPU-deterministik |
| ab_nonin, inflow non-gaji vs income (E204, H21) | fitur | +0,00019 (P 0,884) — Kandidat | E200 | 3 | 4/5 | `features/ab_nonin_v1.parquet` | CPU-deterministik |
| Set CPU saja: 528 + ecm + rp_ev (E205) | set fitur | +0,00203 vs E130a (P 1,000); = E122x1 (+0,00022, P 0,75) | E130a | 3 | – | `oof/E205_s*` | CPU-deterministik |
| DART set E205 (E207) | model | berjalan (seed 42: 0,84496 ≈ GBDT E205 seed 42 0,84494) | E205 | 3 | – | `oof/E207_s*` | CPU-deterministik |

Ditolak sejak 17 Okt (tidak masuk gabungan): E201 (model bantu PD pinjaman lama), H07 stacking per sumber, H08/H09, H14 sementara, H27 kalibrasi per segmen, semua probe putaran 2 (bagian 9–10 laporan eksplorasi).

## Langkah 2 — rencana

F* = set E122x1 (528 + ecm + seq_gru/bigru/tf + seqx_rp/ab/pa) + rp_ev (E200) + ab_nonin (E204). LGB GBDT tuned 5-fold 3 seed (= E208, sebanding dengan E200/E122x1), lalu versi 10-fold untuk model akhir.
