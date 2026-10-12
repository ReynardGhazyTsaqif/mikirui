# CLAUDE.md — DSC LOGIKA UI 2026 (Credit Default Prediction)

Konteks proyek untuk Claude Code. Bahasa komunikasi dengan user: **Bahasa Indonesia**.

## 1. Tentang Kompetisi

- **Lomba**: Data Science Competition (DSC) LOGIKA UI 2026, Departemen Matematika FMIPA UI. Tim 2–3 mahasiswa.
- **Babak penyisihan** (online, Kaggle + notebook): **11–30 Oktober 2026**.
  - Deadline notebook: **30 Oktober 2026 pukul 23.59 WIB**. Telat = **−1 poin per menit**.
  - Nama tim di Kaggle **harus sama** dengan nama tim saat registrasi.
  - Penamaan notebook: `NomorPeserta_NamaTim_Notebook.ipynb` (contoh `DSC0001_RandomSeed_Notebook.ipynb`).
- **Penilaian penyisihan**: Public LB 45% + Private LB 45% + notebook 10% (*reproduksibilitas & kelancaran kode*).
- **Syarat lolos**: 8 tim teratas, **dengan syarat output notebook = output submission yang dipilih di Kaggle**. Kalau tidak sama → digantikan peringkat berikutnya.
- **Final**: pengumuman 11 Nov, registrasi ulang 11–13 Nov, TM final 14 Nov, submit laporan + presentasi (.pdf) 14–19 Nov 23.59 WIB
  (`NomorPeserta_NamaTim_laporan.pdf`, `NomorPeserta_NamaTim_presentasi.pdf`), final luring di UI Depok 21 Nov (10 menit presentasi + 10 menit QnA).
- **Aturan penting**:
  - **Wajib Python.**
  - **Dilarang memakai data selain yang disediakan panitia** (tidak boleh data eksternal / pretrained berbasis data luar).
  - Tidak boleh kerja sama antartim; plagiarisme = diskualifikasi.
- **Metrik = KS statistic** (Kolmogorov–Smirnov, `scipy.stats.ks_2samp` antara prediksi kelas 1 dan 0) — dikonfirmasi dari tab Evaluation Kaggle. KS = max(TPR − FPR), hanya bergantung pada urutan prediksi (rank blend aman, kalibrasi tidak perlu). Implementasi: `src.train.ks_score`. KS jauh lebih noisy dari AUC (sd public ±0,012–0,02 tergantung porsi public), jadi keputusan tetap pakai mean-fold AUC; KS dicatat di log (`oof_ks`, `mean_fold_ks`). Kuota 3 submission/hari, reset 05.00. Kompetisi Kaggle dibuka 10 Okt, tutup 30 Okt.

## 2. Problem

Prediksi probabilitas **default** untuk aplikasi pinjaman: `target = 1` jika DPD (days past due) pinjaman mencapai **90 hari dalam 12 bulan** setelah pencairan. Semua pinjaman tenor 12 bulan.

- Output: `submission.csv` dengan kolom `application_id,default_probability` (30.000 baris, semua `application_id` dari test).
- Urutan `sample_submission.csv` **berbeda** dari `test_applications.csv` — selalu merge berdasarkan `application_id`, jangan andalkan urutan baris.

## 3. Dataset (`Dataset/`, ±905 MB, lihat `Dataset/DATA_DICTIONARY.md`)

Semua nominal dalam IDR bulat, waktu Asia/Jakarta. Kunci join utama: `customer_id` (1 customer = 1 aplikasi train/test).

| File | Baris | Grain | Join | Catatan |
|---|---|---|---|---|
| `train_applications.csv` | 70.000 | aplikasi | `customer_id` | target rate **13,58%** |
| `test_applications.csv` | 30.000 | aplikasi | `customer_id` | |
| `customer_profiles.csv` | 100.000 | customer | `customer_id` | demografi, income deklarasi |
| `previous_applications.csv` | 201.984 | aplikasi lama | `customer_id`; `loan_id` → repayment | hanya 70% customer punya riwayat |
| `repayment_history.csv` | 1.361.341 | cicilan | `loan_id`, `customer_id` | ±66% customer |
| `bureau_observations.csv` | 797.563 | laporan biro × aplikasi | `application_id` (current **dan** previous) | maks 1 laporan per provider per aplikasi |
| `account_behavior.csv` | 1.084.418 | customer × bulan | `customer_id` | 70% customer, 8–23 bulan |
| `ecommerce_transactions.csv` | 4.359.817 | item order | `customer_id`, `order_id` | 529 MB; 100% customer |

### Temuan EDA awal (sudah diverifikasi)

- **Split train/test acak, bukan temporal**: keduanya mencakup aplikasi 27 Des 2024 – 28 Des 2025; customer train & test tidak overlap. → **StratifiedKFold (5-fold)** valid sebagai CV. Proporsi coverage tabel pendukung sama di train & test (70/30).
- **Tidak ada kebocoran waktu di data mentah**: semua timestamp tabel pendukung (`decision_datetime` previous apps, `due_date`, `last_payment_datetime`, `available_datetime` biro, bulan account behavior, transaksi e-commerce) **< `decision_datetime`** aplikasi saat ini. Tetap gunakan `decision_datetime` sebagai cutoff saat membuat fitur berbasis waktu (recency, window 3/6/12 bulan).
- `bureau_observations` memuat laporan untuk aplikasi saat ini (100% aplikasi, 1–3 provider) **dan** untuk previous applications (67% baris). Pisahkan keduanya saat agregasi. `report_datetime` vs `available_datetime` lag 0–120 hari.
- `ecommerce_transactions`: `order_total` diulang per item → **dedup per `order_id`** sebelum agregasi order. Order `cancelled` bukan spending.
- `loan_purpose`, `region`, `employment_type`, `housing_status` hampir tidak informatif sendirian (target rate 13–14% di semua level).
- **Sinyal univariat terkuat (AUC)**: `max_days_past_due` riwayat 0,744 · rasio bayar 0,69 · DPD saat ini 0,689 · skor biro (mean) 0,675 · `annual_interest_rate` 0,646 · utilisasi revolving 0,60. Account behavior & e-commerce lemah jika diagregasi mentah → perlu fitur tren/volatilitas/rasio.
- Customer tanpa riwayat (30%) punya target rate 11,7% (lebih rendah dari rata-rata) → missingness itu sendiri informatif; jangan imputasi sembarangan, biarkan NaN untuk GBDT + flag `has_history`.
- Missing: `score` biro 5,5% (unscored; target rate 13,1% ≈ rata-rata, jadi tidak informatif), `utilization` 59% (tanpa limit revolving).
- `annual_interest_rate` hanya 5 tier (0,16/0,20/0,24/0,28/0,32) dengan target rate monoton 5,5% → 10,6% → 17,8% → 30,4% → 50,3% = grade risiko dari lender.
- `proposed_monthly_installment` = rumus anuitas persis (selisih ≤ 1 rupiah) → fitur "deviasi anuitas" tidak berguna.
- Distribusi skor ketiga biro identik (mean 669, std 53,3) → normalisasi skor per provider tidak perlu.
- `repayment_history`: `days_past_due` = DPD saat cutoff (0 jika lunas); `max_days_past_due` = DPD terburuk cicilan itu (korelasi 1,0 dengan keterlambatan bayar). Jumlah pinjaman lama dengan max DPD ≥ 90: 0 → 9,3%, 1 → 30,5%, 2 → 41,8%, ≥3 → 49,8% target rate.
- 3.422 pinjaman lama `approved` tanpa baris repayment = pinjaman yang cair ≤ 31 hari sebelum cutoff (belum jatuh tempo). Customer seperti ini target rate 19,1% vs 13,4% → fitur "pinjaman baru belum jatuh tempo".

- `ec_days_since_last_order` (hari dari order e-commerce terakhir ke cutoff) adalah fitur tunggal terkuat (AUC 0,689): desil teratas ≤1,5 hari → 28,4% target, >60 hari → 3,4%. Distribusi train = test, jadi sah (bukan leakage). Fitur e-commerce jendela pendek (7/14/30 hari) layak diprioritaskan.

## Cara Menjalankan

```
python run.py E00                    # submission konstan (deteksi metrik)
python run.py E03 --submit           # CV LightGBM dev + tulis submissions/<exp>.csv (+MD5)
python run.py E03 --seeds 42 43 44   # ukur noise antar seed
python run.py E04                    # adversarial validation
python run.py E03 --rebuild          # bangun ulang cache features/*.parquet
```

- Eksperimen baru: tambahkan builder ke `GROUPS` dan konfigurasi ke `EXPERIMENTS` di `run.py`. Ganti nama versi cache (`_v2`) setiap logika fitur berubah.
- Log otomatis: `experiments/log.csv` (aturan terima memakai **rata-rata AUC per fold**, bukan pooled OOF), `oof/<exp>.npy`, `preds/<exp>.npy`, `experiments/importance_<exp>.csv`. Catat submission manual di `submissions/LOG.md`.
- Hasil Fase 1: E01 0,6471 → E02 0,6492 (ditolak; mean-fold turun) → **E03 0,8255 (B1, acuan saat ini)**. Noise antar seed E03: std 0,00013. Adversarial E04: AUC 0,503 (aman).

- Hasil Fase 2 (LightGBM dev, 1 seed): diterima E10 `rp_core` (+0,0010), E50 `ec_window` (+0,0039), E31 `pa_burden` (+0,0014) → **B2 = E31: mean-fold AUC 0,8318, OOF KS 0,5001, 331 fitur** (acuan saat ini). Ditolak: E11, E12, E13, E20, E21, E30, E40, E41, E51 (−0,0021), E60, E61. Pola: grup yang tumpang tindih dengan agregat naif cenderung sedikit menurunkan skor (dilusi), bukan tanpa sinyal. Ablation E62: semua grup di B2 berguna; kontributor terbesar `bu_naive` (−0,0065 jika dibuang), `ec_naive` (−0,0053), `ec_window` (−0,0037).
- Hasil Fase 3: **B3 = E71b** (`experiments/B3.json`): 176 kolom featset `sel_full` (null importance 40 run atas 528 kolom semua grup, termasuk 59 kolom dari grup yang ditolak di Fase 2). Rata-rata 3 seed: mean-fold AUC 0,8329, OOF KS 0,5066 (E31: 0,8312 / 0,4987). Set kedua untuk ensemble: E71c (`sel_b2`, 117 kolom, KS 0,5043). Ditolak: class weight (E72), monotone (E73, setelah uji 3 seed), encoding kategori alternatif (E74). Catatan: seleksi null importance memakai target seluruh train, jadi CV bisa sedikit optimis. Featset ada di `experiments/featsets/*.json`; eksperimen merujuknya lewat `cols=` dan `base="B3"`.
- Noise antar seed pada set besar (E31) std ±0,0006 AUC, lebih besar dari E03. Keputusan tipis (< 0,001) wajib dikonfirmasi 3 seed (contoh: `experiments/seed_check_b3.py`).
- Segmen terlemah (E75): customer tanpa riwayat (30%, KS 0,466 vs 0,518) dan yang belanja ≤2 hari sebelum cutoff (KS dalam segmen 0,428).
- Public LB: E03 0,5244 → E31 0,5380 → E71b 0,5408 (searah dengan CV; public konsisten ±0,035 di atas OOF KS).
- Hasil Fase 4 (semua di B3, seed 42; `src/models.py`, eksperimen `model=` + `blend_with=`): CatBoost E80 model tunggal terbaik (mean-fold AUC 0,8346, KS 0,5092, ~10 mnt/CV); LGB extra_trees E87 0,8337; XGBoost E81 0,8331; LogReg+binning E82 0,8231; MLP sklearn E83 0,8218; ExtraTrees E84 0,8190. Semua "kandidat" (blend dengan E71b naik ≥ 0,0003). Korelasi Spearman antar GBDT 0,97–0,98; LogReg/MLP ~0,92 (paling beragam). Pratinjau blend rank GBDT (CatBoost bobot 2): mean-fold AUC 0,8364, KS 0,5118 → `submissions/P4_blend_*.csv`. torch/autogluon tidak terpasang (E85 ditunda).
- Public: P4 blend 0,5387 (< E71b 0,5408 dan < E80 CatBoost 0,5449, padahal CV P4 tertinggi; peluang keduanya karena noise ≤2,5%). E80 CatBoost tunggal = public terbaik, kenaikannya searah CV.
- **Pelajaran metodologi (diverifikasi, `experiments/seed_bag_check.py`)**: prediksi test tiap model = rata-rata 5 model fold (sudah "di-bagging"), sedangkan OOF = 1 model per baris (lebih noisy). Akibatnya korelasi antar-model di test ~0,015 lebih tinggi dan **kenaikan blend di OOF 1 seed ~2× terlalu optimis** (blend 3 LGB: +0,00185 dengan OOF 1 seed vs +0,00087 dengan OOF rata-rata 3 seed). Aturan: **evaluasi blend/ensemble hanya dengan OOF rata-rata ≥3 seed.** Rata-rata 3 seed sendiri menaikkan tiap model ~+0,002 AUC / +0,003–0,005 KS (E71b 3 seed: AUC 0,8350, KS 0,5097). OOF/preds per seed: `oof/<exp>_s<seed>.npy`.
- **Fase 5 selesai 12 Okt 09.37 (`tune.py`, laporan `experiments/p5/report.md`)**: Optuna LGB 100 trial, CatBoost 40, XGB 41/60 (anggaran habis). Final lr kecil, bag 3 seed (mean-fold AUC / OOF KS): **T_lgb_b3_4f3c7f 0,83788 / 0,5138** (terbaik; vs E71b bag 0,83499), T_xgb_b3_565cf6 0,83700, T_lgb_selb2_4f3c7f 0,83687, T_cat_b3_a6abab 0,83666 (vs E80 bag 0,83544). Caruana nested 0,83792 / 0,5150 (hampir sama dengan tunggal). Submission kandidat: `P5_single_T_lgb_b3_4f3c7f.csv`, `P5_caruana_nested0.83792.csv`. Catatan: iterasi LGB & CatBoost di lr final banyak mentok batas 5000 → naikkan batas iterasi bisa menambah skor. Pipeline panjang dijalankan terpisah (`experiments/p5/run_p5.cmd` via Start-Process), bukan background task Claude, supaya tidak dimatikan saat RAM sempit. Public P5 single 0,54685 (terbaik), P5 Caruana 0,54435. Leaderboard 12 Okt: #1 = 0,58 (di bawahnya rapat), kita #30; target akhir user 0,65.
- Kolom `diterima` di log berisi True/False/"kandidat" → pandas membacanya sebagai teks. Selalu filter lewat `run.accepted_rows()` (bandingkan string "True"), jangan `== True`.
- Registry grup fitur: `src/features/registry.py`. Eksperimen dengan `add=[...]` otomatis memakai set eksperimen diterima terbaik (mean-fold AUC) sebagai basis.

## Planning

Rencana lengkap ada di `Planning Model DSC LOGIKA UI 2026.md` (protokol eksperimen E00–E112, aturan terima, jadwal, pembagian tugas). Riset peningkatan skor (12 Okt, eksperimen E120+, hasil uji kebocoran & screen) ada di file terpisah `Riset Peningkatan Skor DSC LOGIKA UI 2026.md` — jangan digabung ke file lain. Ikuti ID eksperimen dan prefix fitur (`app_`, `prof_`, `pa_`, `rp_`, `bu_`, `bp_`, `ab_`, `ec_`, `x_`) dari dokumen itu.
- **Notebook final harus self-contained**: panitia hanya menerima `.ipynb`, jadi semua kode `src/` harus di-inline dan fitur dihitung dari data mentah (tidak bergantung pada cache `features/`).

## 4. Aturan Kerja / Konvensi Kode

- **Jangan pernah** memakai `*_id` sebagai fitur numerik (opaque identifier).
- **Jangan** memakai data di luar folder `Dataset/` (melanggar aturan lomba).
- Semua fitur dari tabel pendukung **wajib** dihitung relatif terhadap `decision_datetime` aplikasi saat ini.
- **Reproduksibilitas (10% nilai + syarat lolos)**: `SEED = 42` global (numpy, random, model), pinned versi library, notebook final harus bisa *Run All* dari nol tanpa error dan menghasilkan `submission.csv` yang **identik** dengan submission yang dipilih di Kaggle. Simpan hash/nama file setiap submission ke `submissions/LOG.md`.
- Notebook final harus **berisi penjelasan & analisis** (markdown) — bukan hanya kode.
- Fit encoder/target-encoding **di dalam fold CV** untuk menghindari leakage.
- Jangan overfit public LB (private = 45% nilai). Percayai CV lokal; pilih 2 submission akhir: satu best-CV, satu robust (blend).
- File besar: baca `ecommerce_transactions.csv` dengan `usecols`/`dtype` hemat memori atau `chunksize`; cache fitur ke `.parquet` di `features/`.

## 5. Struktur Proyek (rencana)

```
Dataset/            # data mentah panitia (read-only, jangan diubah)
Guidebook/          # guidebook PDF (berupa gambar, bukan teks)
notebooks/          # eksplorasi (01_eda.ipynb, 02_features.ipynb, ...)
src/                # fungsi fitur & training yang dipakai ulang
features/           # cache fitur .parquet (bisa dihapus & dibangun ulang)
submissions/        # submission .csv + LOG.md (CV score, LB score, catatan)
final/              # notebook final untuk diserahkan
```

## 6. Environment

- Windows 11, Python 3.12. Panggil pip lewat `python -m pip` (perintah `pip` tidak ada di PATH).
- Terpasang: pandas 2.3.2, scikit-learn 1.8.0, lightgbm 4.7.0, xgboost, catboost, pymupdf, pypdf. **polars belum terpasang.**
- Jangan beri nama script yang sama dengan modul standar (mis. `signal.py`, `random.py`) — memicu circular import.
- Notebook final kemungkinan dijalankan ulang oleh panitia / di Kaggle → pastikan path data mudah diganti (variabel `DATA_DIR` di sel pertama).
