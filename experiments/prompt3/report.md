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

**Langkah 2 — hasil**: F* sudah terukur sebagai **E204** (5-fold, 3 seed): **0,84677** vs E122x1 0,84495 → Δ **+0,00182**, CI 95% [+0,00133; +0,00232], P 1,000, naik 5/5; KS mean-fold 0,5266 → 0,5293 (`cmp_Fstar_E204_vs_E122x1.json`). Versi 10-fold (E210) dan F*-CPU (E209/E211) dijadwalkan.

## Langkah 3 — Ablation (rencana dan sumber screen)

Screen leave-one-group-out 1 seed = `s33` (kontribusi grup rp, rp_ev, bu+bp, pa, ab, ec, app+prof, x, ecm, seq_ec, seqx pada 28k dan 56k baris; set E200). ab_nonin: membuangnya = E204 → E200, turun −0,00019 dengan P(turun) 0,884 ≥ 0,80 → **dipertahankan**. Grup dengan penurunan screen < 0,0005 dicek ulang 3 seed + bootstrap.

## Langkah 4 — antrean (launcher, masing-masing < 3 jam, berurutan)

| ID | model | set | skema | perkiraan |
|---|---|---|---|---|
| E210 | LGB GBDT tuned | F* | 10-fold bersarang, 3 seed | 40 mnt |
| E209 | LGB GBDT tuned | F*-CPU (528 + ecm + rp_ev + ab_nonin) | 5-fold, 3 seed | 15 mnt |
| E211 | LGB GBDT tuned | F*-CPU | 10-fold, 3 seed | 30 mnt |
| E212 | LGB DART 5000 ronde (setting E141) | F* | 5-fold, 3 seed | 2,3 jam |
| E213 | XGB tuned (setting E150_xgb) | F* | 5-fold, 3 seed | 1 jam |
| H22 | LGB/XGB/CatBoost tuned, fitur hampir mentah (239 kolom naif) | — | 5-fold, 1 seed | 1,5 jam |
| E214 | CatBoost tuned (setting E150c) | F* | 5-fold, 3 seed | 2,8 jam |
| E207 | LGB DART 5000 ronde | E205 (CPU, tanpa ab_nonin) | 5-fold, 3 seed | berjalan |

AutoGluon `best_quality` (H22, mentor ide 14): `kaggle/e215_autogluon_fstar.ipynb` (2 jam/fold) dan `kaggle/e215_autogluon_naive.ipynb` (1 jam/fold), CPU Kaggle, OOF selaras fold; perlu dataset versi baru (`rp_events_v1.parquet`, `ab_nonin_v1.parquet`, `naive_cols.json` sudah disalin ke `kaggle/dataset/`).

## Langkah 7 — alat pemilihan (pendahuluan)

`experiments/prompt3/p3_select.py` → `nilai_harapan.md`. Regresi diperbarui dengan 22 submission: **public = −0,693 + 1,479 × AUC** (sd residual 0,0026); koreksi jenis (disusutkan n/(n+3)): blend dengan DART +0,0008, blend dengan NN −0,0011. Teratas sekarang: E202 + E141 (nilai harapan 0,50292), E202 tunggal (0,50273), E122x1 + E141 (0,50248) — selisih ≤ 0,0005, dan **semuanya GPU**. Kandidat CPU terbaik yang sudah punya public: E130a + E141 (public 0,55858, nilai harapan 0,50121). → prioritas submission berikutnya: kandidat CPU baru (E205 + E207 DART, E211) supaya kandidat yang bisa direproduksi punya public terukur.

**E207 (DART di set CPU E205)**: seed 0,84496 / 0,84507 / 0,84500, **bag 0,84536** vs E205 GBDT 0,84517 → Δ +0,00018, CI [−0,00019; +0,00057], P 0,834 → **Kandidat** (Spearman 0,9974). Submission kandidat A: `E205_E207_rank50_auc0.84545.csv` (MD5 afc487f6…, KS OOF 0,5242) — menunggu diunggah user.

---

## STATUS & CARA MELANJUTKAN (disimpan 18 Okt 16.15 — sesi dihentikan karena limit Claude)

**Berjalan di laptop (mandiri, tanpa Claude)**: `experiments/run_s33.cmd` (s32 ulang → s33; log `experiments/explore_17okt/s33_stdout.log`) lalu `experiments/run_laptop_queue.cmd` (menunggu "launcher s33 selesai", lalu berurutan `run_p3_chain1.cmd` → `run_p3_step8.cmd` → `run_p3_step6a.cmd` → `run_h22.cmd`; perkiraan selesai ±22.30). Log: `experiments/e121_stdout.log` (E210, E209, E211, E210n, E122x1n), `experiments/p3_stdout.log` (refit check, E216 pseudo-label, NF_gbdt, penanda "antrean laptop selesai"), `experiments/h22_stdout.log`.

**Menunggu user (Kaggle, CPU, Internet On)**: unggah `kaggle/dataset/` sebagai versi baru dataset privat `dsc-logika-exp` (baru: `rp_events_v1.parquet`, `ab_nonin_v1.parquet`, `naive_cols.json`, `p3_params.json`); jalankan paralel `kaggle/p3_E214.ipynb` (CatBoost F*, ±3 jam), `kaggle/p3_E212.ipynb` (DART F*, ±2,5 jam), `kaggle/p3_NF_dart.ipynb` (DART fold baru, ±2,5 jam), `kaggle/p3_E213.ipynb` (XGB F*, ±1 jam), `kaggle/e215_autogluon_fstar.ipynb` (±10 jam) dan `kaggle/e215_autogluon_naive.ipynb` (±5 jam); taruh semua `*.npy` keluaran di `kaggle/output/`. Runner sudah diuji lokal: fold 0 AUC 0,84596 = E204 seed 42 (0,845957). Submission `E205_E207_rank50_auc0.84545.csv` belum diunggah (isi public di `submissions/LOG.md` setelah keluar).

**Langkah berikutnya saat sesi dilanjutkan** (urut):
1. Baca hasil s32 ulang (H14, tier) dan s33 (`experiments/explore_17okt/group_learning.csv` + `s33_stdout.log`): grup dengan penurunan screen < 0,0005 → cek 3 seed + bootstrap (Langkah 3) → tetapkan F** (dugaan: F** = F*).
2. `python kaggle/import_results.py` setelah file Kaggle ada → `compare_oof.py` untuk E210 (vs E204/E202), E209/E211 (F*-CPU), E212/E213/E214 (vs E204), E216 (pseudo-label vs E204), E210n/E122x1n (seed baru: Δ harus tetap positif), NF_gbdt vs NF_dart (fold baru), H22, E215.
3. Langkah 5: `python experiments/prompt3/p3_ensemble.py --models E210 E212 E213 E214 --tag gbdt` (+ versi CPU: E211 E207); NN hanya terpisah, bobot ≤ 0,2.
4. Langkah 6: simulasi public (`experiments/explore_17okt/s34_public_noise.py`, tambah kandidat), CI bootstrap kandidat final.
5. Langkah 7: `python experiments/prompt3/p3_select.py` → `nilai_harapan.md`; kandidat A (CPU) dan B (GPU, hanya jika reproducible); usul 3 submission/hari via `experiments/make_sub.py` + `verify_submission.py --log`.
6. Langkah 8: refit 100% (`e145_refit.py --extra ...`) bila cek aturan iterasi lolos (|selisih| ≤ 0,0002); pseudo-label bila E216 diterima.


---

## 19 Okt — public E205 + E207 = 0,56428 (rekor baru) dan arah baru

**Public**: `E205_E207_rank50_auc0.84545.csv` (CPU saja, GBDT + DART, set 528 + ecm + rp_ev) **0,56428** — +0,0048 dari rekor lama E122x1 + E141 (0,55951) dan +0,0067 dari E202 (0,55754) walau CV-nya lebih rendah (0,84545 vs 0,84711).

**Diagnosis** (semua dari OOF/preds, tanpa melatih ulang):
- Fitur GRU (seq_*, seqx_*) sehat di test: Spearman antar model di test selalu > di OOF (GRU vs tanpa GRU +0,0018 s.d. +0,0028; tanpa vs tanpa +0,0010 s.d. +0,0013); korelasi fitur GRU dengan prediksi E205 di test ≥ di OOF (seq_bigru 0,735 vs 0,726). Tidak ada salah urutan atau pergeseran.
- Notebook GRU (`e122_gru.py`, `e122x_seq.py`) bersih: early stopping di holdout 10% data latih, nested benar → tidak ada optimisme CV seperti kasus NN E142.
- Simulasi 2000 subset 9.000 baris (s34): E205 + E207 − E202: harapan −0,0030, sd 0,0037, teramati +0,0067 → **z +2,62** (P 0,005); − (E202 + E141) z +2,27; − (E122x1 + E141) z +1,16; − (E130a + E141) z +1,06.
- Tafsiran: E205 + E207 adalah maksimum dari 23 submission (efek pemenang) → sebagian lonjakan = keberuntungan; sebagian nyata (rp_ev di set CPU +0,0019 CV vs E130a + E141). E202 kurang beruntung di subset public (pola "model ber-GRU di bawah garis regresi" = satu undian subset public yang sama, bukan bukti independen berulang).

**Keputusan arah (pakai aturan Langkah 7)**: nilai_harapan E205 + E207 = 0,45 × 0,56428 + 0,45 × ±0,5574 ≈ **0,5048** > E202 ±0,5029, dan E205 + E207 **CPU-deterministik** → **jalur utama = keluarga CPU (kandidat A)**: perkuat (ab_nonin, 10-fold, pasangan DART 10-fold, model ketiga), buat beberapa varian kuat untuk portofolio public, dan bangun notebook final kandidat A sekarang (jalur kritis). Kandidat B (set GPU) tetap pembanding.

**Antrean 19 Okt**: laptop `experiments/run_cpu_19okt.cmd` (10.47): E209 → E211 → cek refit set CPU → E216c (pseudo-label set CPU) → E210 (set GPU 10-fold) → H22. Kaggle (user): `kaggle/p3_E219.ipynb` (DART set CPU 10-fold, ±5 jam), `p3_E218` (DART set CPU 5-fold, ±2,5 jam), `p3_E221` (CatBoost set CPU, ±3 jam), `p3_E220` (XGB set CPU, ±1 jam), `p3_NF_dart` (DART fold baru, ±2,5 jam); notebook set GPU (E212–E214) dihapus. Catatan: antrean laptop 18 Okt tidak jalan (laptop berhenti, s32 ulang terputus di fold 2).

### 19 Okt siang — hasil set CPU

| ID | isi | bag 3 seed | vs acuan | keputusan |
|---|---|---|---|---|
| E209 | GBDT F*-CPU (528 + ecm + rp_ev + ab_nonin), 5-fold | 0,84537 (KS OOF 0,5229) | vs E205 +0,00019, CI [−0,00016; +0,00052], P 0,854 | Kandidat (H21 konsisten dengan set GPU) |
| E211 | GBDT F*-CPU, 10-fold bersarang | **0,84595** (KS OOF 0,5243) | vs E209 **+0,00059**, CI [+0,00026; +0,00092], P 1,000 | **Diterima** |

Submission baru (CPU-deterministik, lolos `verify_submission.py`, tercatat di LOG): **`E211_E207_rank50_auc0.84601.csv`** (MD5 9a88b413…, KS OOF 0,5247) = usulan utama; `E209_E207_rank50_auc0.84562.csv` (MD5 c06ad87d…, KS OOF 0,5244) = cadangan. Pasangan DART yang setara kuat (E219 = DART F*-CPU 10-fold, Kaggle) diharapkan menaikkan blend lebih jauh (E207 5-fold 0,84535 lebih lemah dari E211).

**Notebook final kandidat A** dirakit: `final/build_notebook.py` → `final/NomorPeserta_NamaTim_Notebook.ipynb` (43 sel, kode sumber yang sama; target MD5 E205 + E207 afc487f6…). Verifikasi persis terhadap cache (`final/verify_notebook.py --train-check`) **belum dijalankan** (laptop dimatikan user).

## STATUS & CARA MELANJUTKAN (diperbarui 19 Okt ±12.00 — laptop dimatikan user)

Tidak ada job yang berjalan. Belum dijalankan dari antrean `experiments/run_cpu_19okt_b.cmd`: verifikasi notebook, cek aturan refit set CPU, E216c (pseudo-label set CPU), E210 (set GPU 10-fold), H22. Untuk melanjutkan semuanya cukup jalankan launcher baru yang menghapus baris tunggu E209/E211 (E209 dan E211 sudah selesai), atau langsung:

1. `python final/verify_notebook.py final/NomorPeserta_NamaTim_Notebook.ipynb --train-check` (±30 mnt) — harus "identik" untuk semua grup, matriks, ecm, dan prediksi fold 0. Bila beda: perbaiki bundler, lalu Run All penuh (±3,5 jam) dan cocokkan MD5 dengan afc487f6….
2. Notebook final perlu diperbarui ke konfigurasi kandidat terbaik saat itu (mis. E211 + E219: tambah `ab_nonin`, 10-fold untuk GBDT dan DART) — parameterkan `build_notebook.py`.
3. Kaggle (user): unggah `kaggle/dataset/` versi baru; jalankan `kaggle/p3_E219.ipynb`, `p3_E218`, `p3_E221`, `p3_E220`, `p3_NF_dart` (CPU); keluaran `*.npy` ke `kaggle/output/` → `python kaggle/import_results.py` → `compare_oof.py` (E219 vs E207, E218 vs E207, E220/E221 vs E209) → `experiments/prompt3/p3_ensemble.py --models E211 E219 ...`.
4. Submission: usulan hari ini E211 + E207 (utama) dan E209 + E207 (cadangan); isi public di `submissions/LOG.md`; perbarui `experiments/prompt3/p3_select.py` (nilai_harapan).
5. Sisa Langkah 6/8: refit check + E216c set CPU, seed baru 101–103 (E211n), fold baru GBDT/DART.


### 19 Okt sore — public keluarga CPU dan arahan user

**Public**: `E209_E207_rank50` **0,56351**, `E211_E207_rank50` **0,56302** (rekor tetap `E205_E207_rank50` 0,56428). Keluarga CPU (GBDT + DART tanpa fitur GRU) kini 3 dari 3 submission di 0,5630–0,5643, sedangkan keluarga ber-GRU 0,5553–0,5595 (8+ submission) — berlawanan dengan CV (fitur GRU +0,0014). Ini **pola konsisten yang belum terjelaskan** dan menjadi petunjuk utama sesi berikutnya.

**Arahan user (19 Okt)**: (1) notebook final **jangan dikerjakan dulu** — user menyelesaikannya sendiri dalam 1 hari menjelang tenggat; (2) target public **minimal 0,58**; (3) jangan menjelaskan hasil sebagai "keberuntungan" — cari penyebab dan cara menaikkan; pertanyaan user: kenapa tim lain bisa 0,57+.

**Rencana sesi berikutnya** (belum dikerjakan):
1. Bedah pola public keluarga CPU vs GRU: submission diagnostik yang memisahkan komponen (mis. set CPU + seq e-commerce saja, set CPU + seqx saja, GBDT CPU tanpa DART) untuk menemukan komponen yang menurunkan public; cek ulang cara pembuatan fitur GRU untuk baris test (persentil per blok, model mana yang dipakai untuk te{k}) dan pertimbangkan membuang/merekonstruksi fitur yang merugikan.
2. Perkuat keluarga CPU yang terbukti di public: pasangan DART 10-fold (E219), CatBoost/XGB set CPU (E220/E221), ensemble nested — notebook Kaggle sudah siap (`kaggle/p3_*.ipynb`).
3. Cari sumber kenaikan besar baru dengan sudut pandang tim lain (pipeline agregat sederhana + ensemble kuat): bandingkan model "sederhana kuat" (mis. CatBoost/LGB di fitur mentah + target encoding) dengan pipeline kita di public, karena public keluarga sederhana (tanpa GRU) justru lebih tinggi.
4. Sisa antrean lokal: cek aturan refit + E216c (pseudo-label) set CPU, E210, H22, AutoGluon.


### 19 Okt malam — hasil Kaggle set CPU (diimpor `kaggle/import_results.py`)

| ID | model (set F*-CPU = 528 + ecm + rp_ev + ab_nonin) | bag 3 seed | pembanding | keputusan |
|---|---|---|---|---|
| **E219** | LGB DART 5000 ronde, **10-fold** | **0,84623** (KS OOF 0,5293) | vs E207 +0,00087 (CI [+0,00046; +0,00128], P 1,000); vs E211 +0,00027 (P 0,928) | **Diterima** vs E207; model tunggal CPU terbaik |
| E218 | LGB DART 5000 ronde, 5-fold | 0,84565 | vs E207 +0,00029, P 0,957 | Kandidat |
| E220 | XGB tuned, 5-fold | 0,84505 | vs E209 −0,00032, P 0,118 | Ditolak (sendiri) |
| E221 | CatBoost tuned, 5-fold | 0,84290 | vs E209 −0,00247 | Ditolak (sendiri; Spearman 0,980 = paling beragam) |
| NF_dart | DART fold baru seed 2026, non-nested | (3 seed tersimpan) | menunggu NF_gbdt (laptop) | Langkah 6 |

Ensemble (Langkah 5, `p3_ensemble.py --models E211 E219 E220 E221`): rata-rata sederhana 0,84615, bobot nested 0,84621 (E219 0,60 / E211 0,26 / XGB 0,09 / Cat 0,05), Caruana 0,84621 → tidak lebih baik dari E219 sendiri (0,84623) → pilih yang paling sederhana. Submission: **`E211_E219_rank50_auc0.84628.csv`** (GBDT + DART, keduanya 10-fold; MD5 a8a95dbe…; KS OOF 0,5260 tertinggi di keluarga CPU) — menunggu public.
