# Laporan kerja eksplorasi 17 Okt+ — DSC LOGIKA UI 2026

> Sumber perintah: `PROMPT_UTAMA.md` di root repo (17 Okt 21.11). `LAMPIRAN_ide_eksternal.md` sengaja tidak dibuka sebelum Langkah 3 selesai.
>
> Lokasi: file ini ditulis di worktree `.claude/worktrees/explore-17okt` (branch `worktree-explore-17okt`), karena sesi latar ini wajib terisolasi dari checkout utama. Data, cache `features/`, `oof/`, `preds/` dibaca (hanya baca) dari checkout utama `D:\Unand\Sems 7\Lomba\DSC\dsc-logika`. Script analisis ada di folder yang sama dengan file ini.

## 0. Ringkasan satu halaman (selalu diperbarui)

*Status 18 Okt sore: Langkah 1–3 selesai; diff aturan baru disetujui dan dipakai; Langkah 4 berjalan (bagian 9): E200 diterima, E201 ditolak, E202 diterima, E203 (parameter tuned tetap optimal), E204 kandidat, E205 (set CPU saja) diterima vs E130a; s27 = model jenuh; s29 = masih lapar data. Berjalan: s31 (H07), s32 (H14), E207 (DART set CPU), lalu s33. Pekerjaan di checkout utama, branch `explore-17okt`.*

- **Posisi**: public terbaik tercatat 0,55951 (`E122x1_E141_rank50`, 16 Okt). CV terbaik sekarang: **E202 mean-fold AUC 0,84710** (set E200 = E122x1 + rantai bayar event, dilatih 10-fold bersarang); sebelum eksplorasi: E122x1 0,84495.
- **Target terverifikasi**: public 0,58 ≈ mean-fold AUC 0,857–0,860 → butuh **+0,012 s.d. +0,015**.
- **Temuan terpenting**:
  1. **Cara data dibangkitkan**: tiap customer punya risiko default laten π yang **stabil**; target sekarang berperilaku persis seperti "satu pinjaman lagi" (ICC label antarpinjaman 0,40). Pembayaran hanya terjadi di ulang bulan due (rantai Markov bulanan); laju lewat-bayar datar 23 bulan (0,35 vs 0,15). E-commerce & rekening punya **ramp khusus sebelum keputusan sekarang yang bergantung langsung pada target**. Biro = laten stabil + noise per laporan.
  2. **Batas informasi bukan penghalang**: dengan π diketahui sempurna AUC ≈ 0,927. Kesalahan model (FN/FP) tidak bisa dibedakan dari kasus benar → gain harus dari **taksiran π yang lebih tajam**, bukan pola baru di kasus salah.
  3. **Model hampir jenuh** terhadap representasi sekarang: kurva belajar asimtot ≈ 0,848 (lr 0,05); 25+ kandidat sinyal baru |z| < 2. Satu-satunya sisa terukur: **rantai bayar level event** (z 6,3; screen +0,00069, 5/5).
  4. **Data berlabel tambahan**: 81.745 pinjaman lama lengkap berlabel dari mekanisme yang sama; bukti LOO AUC 0,825 vs 0,817 untuk target; 2× data ≈ +0,002. E126 gagal karena desain as-of.
  5. Kejanggalan proyek: `src/features/` tidak ter-version; penolakan Fase 2 berbasis 1 seed di dalam noise; parameter tuned berasal dari set 176 kolom; tidak ada kebocoran ID/urutan baris.
- **Hipotesis teratas** (bagian 5): H01 rantai bayar event (+0,0005–0,001), H02 model bantu PD pinjaman LOO (+0,001–0,003), H03 model sifat statis tanpa riwayat, H04 NN multi-tugas, H05 10-fold + refit. Jumlah jujur semua tuas berbukti ≈ +0,003–0,006.
- **Kenaikan CV terkumpul (18 Okt)**: **+0,00214** (0,84495 → 0,84710): E200 = H01 rantai bayar level event +0,00163 (P 1,000, 5/5) dan E202 = H05 10-fold bersarang +0,00051 (P 0,999, 4/5). Keduanya diterima; E200 diusulkan jadi acuan baru (menunggu konfirmasi user). E204 (H21 inflow non-gaji) kandidat (+0,00019, P 0,88). E201 (H02) ditolak (−0,00032); H08/H09 ditolak di screen; E203: parameter tuned tetap optimal. **Kandidat CPU-saja E205 = 0,84517** (setara E122x1 GPU 0,84495; Spearman 0,989). Target 0,857–0,860 → sisa **+0,010 s.d. +0,013**.
- **Diagnosis jenuh (18 Okt)**: residual boosting dengan pohon depth 8 di atas OOF E200 berhenti di iterasi 1 (s27) → tidak ada interaksi tersisa; kurva belajar masih naik (+0,0030 dari 28k → 56k; asimtot 0,847–0,857) → kenaikan berikutnya hanya dari representasi yang lebih hemat data (seperti rp_ev, setara > 2× data), refit/ensemble, atau informasi baru.
- **Ide eksternal**: 13 ide dicek dengan data → 2 sudah dikerjakan (kurva belajar, kasus salah), 1 audit bersih (ID/urutan baris tabel pendukung), 5 terbantah (GMM, artefak angka, tujuan↔belanja, pinjam-tutup-utang, kalender/gajian), 5 masuk daftar (H17 hazard event, H18 objektif ranking, H19 fitur otomatis, H20 KNN, plus penggabungan ke H02/H05/H06/H15).
- **Rencana berikutnya**: s31 (H07 stacking per sumber) dan s32 (H14 mekanisme tier) ±14.00 → E207 DART di set CPU E205 (±16.00) → s33 kurva belajar per grup (±16.40; menunjuk grup tempat model masih belajar lambat) → rekayasa representasi untuk grup itu → PROMPT 3 Langkah 2–7 (±22/23–25 Okt).
- **Pertanyaan terbuka**: konfirmasi E200 sebagai acuan baru (dengan skema latih 10-fold E202 untuk model akhir).

---

## 1. Pemahaman awal (Langkah 1)

### 1.1 Masalah, data, aturan

- **Target**: `target = 1` jika DPD akhir hari pinjaman mencapai 90 dalam 12 bulan kalender setelah pencairan. Semua pinjaman tenor 12 bulan. Train 70.000 (13,58% positif), test 30.000. 1 customer = 1 aplikasi; customer train dan test tidak tumpang tindih.
- **Metrik**: KS dua sampel (`ks_2samp` prediksi kelas 1 vs 0) = max(TPR − FPR). Hanya bergantung urutan. Dicek ulang hari ini: `src.train.ks_score` = `ks_2samp` persis (0,524551 = 0,524551 pada OOF E122x1).
- **Data** (`Dataset/`, ±905 MB): 8 tabel. Kunci `customer_id`; biro memakai `application_id` (aplikasi sekarang dan aplikasi lama); cicilan memakai `loan_id` → aplikasi lama. Semua timestamp tabel pendukung < `decision_datetime` (diverifikasi tim 12 Okt). 30% customer tidak punya aplikasi lama sekaligus tidak punya data rekening (persis 21.000 dari 70.000 di train).
- **Aturan (guidebook hal. 14–18, dibaca ulang)**: wajib Python; dilarang data selain dari panitia; skor penyisihan = Public 45% + Private 45% + notebook 10% (reproduksibilitas dan kelancaran kode); 8 besar lolos hanya jika output notebook = output submisi terpilih di Kaggle; telat −1 poin/menit (30 Okt 23.59 WIB); tidak boleh kerja sama antartim; plagiarisme = diskualifikasi. Larangan "model pretrained berbasis data luar" berasal dari CLAUDE.md/keputusan tim, guidebook hanya menyebut larangan data selain dari panitia.
- **Kuota**: 3 submission/hari (reset 05.00). Jadwal internal: feature freeze 26 Okt, notebook 27–29 Okt, upload 30 Okt ≤ 12.00 WIB.

### 1.2 Pipeline saat ini

```
Dataset/ ──► 23 builder grup fitur (src/features, cache features/*_v1.parquet) ──► 528 kolom
        └──► fitur bertarget nested per fold (format npz tr{k}/va{k}/te{k}):
               ecm_* (7)    : LightGBM level order e-commerce, agregat per customer (e121.py --build, CPU)
               seq_* (3)    : GRU / biGRU / Transformer pada urutan order (Kaggle GPU), persentil per blok
               seqx_* (3)   : GRU pada urutan cicilan / rekening / aplikasi lama (Kaggle GPU), persentil per blok
   ──► LightGBM tuned (Optuna tune_lgb: leaves 35, depth 4, min_child 34, ff 0,129, bagging 0,926,
        L1 0,012, L2 7,66, min_gain 0,94, extra_trees, lr 0,02, ES 200 pada fold luar, batas 15000) × 5 fold × 3 seed
   ──► LightGBM DART (pohon sama, lr 0,05, 5000 ronde, drop 0,1, skip 0,5) × 5 fold × 3 seed
   ──► blend rank 50:50 ──► submission (round 6, LF)
```

**Perintah persis untuk mereproduksi submission public terbaik `E122x1_E141_rank50_auc0.84480.csv` (MD5 7da18bf0…)** — direkonstruksi hari ini dari `preds/` dan MD5-nya cocok (`experiments/explore_17okt/` belum berisi script ini; versi sementara ada di `$CLAUDE_JOB_DIR/tmp/repro_sub.py`):

```
# 0. cache 528 kolom (dibangun otomatis saat pertama dipakai) + fitur order e-commerce nested (CPU)
python e121.py --build                                   # -> features/e121_ordermodel_v1.npz
# 1. GPU Kaggle (dataset privat dsc-logika-exp dari `python kaggle/make_dataset.py`, tabel urutan dari
#    `python kaggle/make_seq_tables.py`): e122_gru.ipynb, e122_bigru.ipynb, e122_tf.ipynb, e122x_single.ipynb
python kaggle/import_results.py
python kaggle/rank_blocks.py features/e122_gru_v1.npz features/e122_gru_v1r.npz      # dst. untuk _bigru, _tf, e122x_rp/ab/pa
# 2. CV + prediksi test
python e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --basefs all --name E122x1
python e141.py --rounds 5000 --lr 0.05 --seeds 42 43 44 --name E141          # DART pada set E130a (hanya ecm)
# 3. blend: bag(x) = rank(rata-rata rank seed 42/43/44); sub = round(rank(0.5*bag(E122x1)+0.5*bag(E141))/30000, 6)
#    merge ke sample_submission, to_csv(lineterminator="\n")  (pola experiments/make_sub_e141x.py)
```

Komponen GPU (seq_*, seqx_*) dilatih di Kaggle T4 dengan `torch.use_deterministic_algorithms(True, warn_only=True)`: reproduksi bit-identik belum diuji → tag **butuh cek**. Komponen lain CPU-deterministik (LightGBM `deterministic=True`, `force_row_wise=True`, 4 thread).

### 1.3 Verifikasi hubungan public–CV (dari `submissions/LOG.md` + `experiments/log.csv`)

| Subset submission | Regresi public ~ mean-fold AUC | r | AUC untuk public 0,58 |
|---|---|---|---|
| 9 pertama (versi Riset bag. 10) | public = −0,913 + 1,742·AUC | 0,956 | **0,8568** |
| Semua 20 | public = −0,723 + 1,514·AUC | 0,958 | 0,8604 |
| GBDT saja (14) | public = −0,754 + 1,553·AUC | 0,978 | 0,8592 |
| GBDT dengan AUC > 0,84 (9) | tidak ada hubungan (r = −0,05) | — | — |

- Kesimpulan: target CV **0,857–0,860** terkonfirmasi (+0,012 s.d. +0,015 dari 0,845). Di pita 0,842–0,846 public sudah tidak bisa membedakan model (sd selisih berpasangan 0,003–0,004).
- Selisih **public − OOF KS** sangat stabil: GBDT rata-rata +0,0339 (sd 0,0021), blend lintas keluarga +0,027–0,031. Sumber yang masuk akal: prediksi test = rata-rata 5 model fold (lebih kuat dari OOF 1 model per baris), bias positif KS pada sampel kecil, dan keberuntungan subset public yang sama untuk semua submission. Implikasi: private kemungkinan lebih rendah dari public untuk semua tim; yang penting adalah kekuatan relatif.

### 1.4 Peta informasi (setiap kolom mentah)

Legenda: **D** = dipakai mendalam (banyak turunan + diuji model khusus), **K** = hanya agregat kasar/mentah, **B** = belum disentuh sebagai fitur. "B3" = masuk set 176 kolom terpilih; set acuan sekarang memakai semua 528 kolom.

| Tabel | Kolom | Status | Dipakai di |
|---|---|---|---|
| applications | application_datetime | K | app_hour/dow/month, process_hours (E02 ditolak; probe 12 Okt: lama proses AUC 0,502) |
| | decision_datetime | D sebagai cutoff, K sebagai fitur | dec_hour saja; tanggal/hari bulan/jarak ke akhir bulan belum |
| | loan_amount, proposed_monthly_installment | D | mentah, PTI, LTI, vs pinjaman lama, total PTI |
| | annual_interest_rate | D | mentah, interaksi, rate_pred/resid (E61 netral) |
| | term_months | — | konstan 12 |
| | loan_purpose | K | kategori native |
| customer_profiles | age, employment_tenure_months, dependents | K | mentah + 3 rasio |
| | declared_monthly_income | K–D | mentah, PTI/LTI, vs income rekening (E41), belanja vs income (E60), `total_vi` di model order |
| | region, employment_type, housing_status | K | kategori native (tidak informatif sendirian) |
| previous_applications | decision_datetime, decision | D | jumlah/porsi/jendela 30–365 hari, recency, keputusan terakhir, pinjaman aktif, GRU seqx_pa |
| | application_datetime | K | umur aplikasi, lama proses rata-rata |
| | requested_amount, approved_amount, term_months, annual_interest_rate, scheduled_monthly_installment | K–D | naive stats, beban aktif (E31 diterima), rasio vs pinjaman sekarang, GRU seqx_pa |
| | loan_id | K | join ke cicilan (rp_loan E12 ditolak 1 seed; pinjaman baru belum jatuh tempo) |
| repayment_history | due_date, amount_due, amount_paid, last_payment_datetime, days_past_due, max_days_past_due | D | rp_naive/core/window/loan/trend, GRU seqx_rp, model baris E121d (+0,0001) |
| | installment_id | — | ID (tidak dipakai) |
| | urutan cicilan dalam pinjaman, pola bayar parsial, selisih amount_due vs jadwal | B | hanya implisit di GRU seqx_rp (`inst_k`) |
| bureau_observations (sekarang) | score | D | pivot A/B/C, statistik, skor segar berbobot, delta vs laporan lama |
| | active_accounts, outstanding_balance, revolving_*, utilization, inquiries_last_6m, delinquent_accounts_last_12m | K–D | pivot + statistik + jumlah (bu_naive/bu_extra) |
| | report_datetime, available_datetime | K | umur laporan, avail_age_min; jeda report→available, laporan terbit setelah `application_datetime` (1%) belum |
| bureau_observations (aplikasi lama) | semua kolom | K | bp_naive (mean/max/min), bp_trend (E21 ditolak 1 seed), ringkasan per aplikasi lama di seqx_pa |
| account_behavior | closing/average/minimum_balance, inflows, outflows, income_like_credits, low_balance_days | K–D | naive, jendela 3/6/12 bln, slope, CV, vs income deklarasi (E40/E41 ditolak 1 seed), GRU seqx_ab, model baris (+0,0001) |
| | opening_balance | K | naive saja; kontinuitas opening(m) vs closing(m−1) belum dicek |
| | month | K | indeks bulan dari belakang |
| ecommerce_transactions | transaction_datetime, order_total, order_status | D | recency, jendela 3–365 hari, intensitas, gap, model order ecm_*, GRU seq_* |
| | product_category, item_total, quantity, payment_method | K–D | porsi kategori/metode, pergeseran (E51 ditolak −0,0021), fitur model order |
| | jam/hari order | K | share_night/weekend (E51), fitur model order |

Catatan: probe 16 Okt menyatakan semua kolom mentah dirujuk di `src/features`. Yang **belum** disentuh adalah relasi antarkolom dan antartabel yang spesifik (kontinuitas saldo, jadwal vs realisasi cicilan, waktu terbit laporan biro vs waktu aplikasi, kalender keputusan), bukan kolomnya.

### 1.5 Riwayat eksperimen dan kualitas bukti

Aturan lama: Δ mean-fold AUC ≥ +0,0005 dan ≥ 4/5 fold. Noise antar seed: E03 std 0,00013; set besar (E31) std ±0,0006; LGB tuned bag 3 seed jauh lebih stabil.

| ID | Isi | Hasil (mean-fold AUC) | Seed / acuan | Kualitas bukti | Alasan tolak yang sebenarnya |
|---|---|---|---|---|---|
| E01–E03 | baseline B0 → agregat naif | 0,6517 → 0,8256 | 1 | kuat (lompatan besar) | — |
| E02 | rasio aplikasi/profil | −0,0010 | 1, vs E01 | lemah | tidak terbukti; 4 kolomnya kemudian lolos null importance |
| E10 | rp_core | +0,0010 diterima | 1, vs E03 | sedang | — |
| E11 / E12 / E13 | rp jendela / per pinjaman / tren | −0,0000 / −0,0003 / +0,0001 | 1, vs E10 | **lemah** (di dalam noise ±0,0006) | tumpang tindih dengan rp_naive/rp_core; 15/9/3 kolomnya lolos null importance |
| E50 | ec jendela | +0,0039 diterima | 1, vs E10 | kuat | — |
| E51 | ec metode bayar/kategori/pola waktu | −0,0021 | 1, vs E50 | sedang (negatif besar untuk param dev) | dilusi (27 kolom, ff 0,5); hanya 1 kolom lolos null importance |
| E20 / E21 | biro ekstra / tren biro lama | −0,0004 / −0,0004 | 1 | lemah | tumpang tindih; 6/3 kolom lolos |
| E30 / E31 | pa jendela / beban aktif | −0,0007 / +0,0014 diterima | 1 | lemah / sedang | E30: tumpang tindih (13 kolom lolos) |
| E40 / E41 | ab jendela / income | −0,0006 / −0,0004 | 1 | lemah | 0 / 3 kolom lolos; rekening memang lemah (AUC univariat maks 0,576) |
| E60 / E61 | lintas tabel / residual bunga | −0,0001 / +0,0000 | 1 | lemah | tumpang tindih |
| E62 | ablation per grup | semua grup B2 berguna | 1 | sedang | — |
| E70 / E71 | filter redundan / null importance (target seluruh train) | +0,0005 / B3 = 176 kolom (3 seed 0,8329) | 3 | sedang; seleksi bisa sedikit optimis | — |
| E72–E74 | class weight / monotone / encoding | netral–negatif | 1–3 | sedang | — |
| E80–E87 | CatBoost/XGB/LR/MLP/ET/extra_trees | 0,8190–0,8346 | 1 → 3 | sedang | dipakai sebagai bahan blend |
| E90–E92 (P5) | Optuna LGB/Cat/XGB pada B3, fold 0–2 | LGB 0,83788 bag 3 | 3 | sedang | param tuned di set lama |
| E140a / E144 | 15.000 iterasi / 5 seed | +0,0001 / +0,0001 | 3–5 | kuat (memang kecil) | — |
| **E121** | model order e-commerce nested (ecm_*) | **+0,00455** (0,84261) | 3, vs T_lgb r15k | kuat, 5/5, audit label acak lolos | — |
| E121b/c/d/e | model order besar / level item / tabel lain / agregat waktu | −0,0001 … −0,0004 / screen +0,0001 | 3 / screen | sedang | informasi sama dengan ecm_* |
| E130a | semua 528 kolom + ecm | +0,00053 (0,84314) | 3, vs E121 | sedang (4/5) | — (membuktikan grup "ditolak" berguna dengan param tuned) |
| E130b | null importance ulang (333 kolom) | +0,00036, 3/5 | 3, vs E130a | sedang | ditolak aturan lama; **perlu dinilai ulang dengan aturan baru** |
| E146 | Optuna ulang di fold 0–2 | −0,00001 | 3 | sedang | overfit ke fold tuning; tuning 5 fold belum pernah |
| E125 | prediksi label bantu (bukan target) | −0,00019 | 3 | sedang | informasi sudah tersirat |
| E128 | model khusus segmen tanpa riwayat | +0,00004 | 3 | sedang | model global sudah menangkap segmen |
| E150_xgb / E150c | XGB / CatBoost di set besar | 0,84230 / 0,84222 (1 seed) | 3 / 1 | sedang | diversitas kecil (Spearman 0,98–0,99) |
| E141 | DART | +0,00038 vs E130a, 5/5, P 0,97 | 3 | sedang | "kandidat" aturan lama; **lolos aturan baru** (P ≥ 0,95 & Δ ≥ 0,0003) — perlu dihitung ulang |
| **E122** | GRU urutan order (seq_gru) | **+0,00097** (0,84411), 5/5 | 3, vs E130a | kuat, uji label acak lolos | — |
| E122r / E122c | persentil per blok / 3 varian urutan | +0,00019 (P 0,97) / +0,00044 (P 1,00) | 3 | sedang–kuat | E122c **lolos aturan baru**; E122r jadi "kandidat" |
| E122x1/2/3 | GRU cicilan/rekening/aplikasi lama | +0,00041 (P 0,99) / +0,00033 / +0,00041 | 3, vs E122c | sedang | E122x1 **lolos aturan baru** |
| E142 | RealMLP / TabM (Kaggle GPU) | 0,8428–0,8448 (seq/iv) | 5 | sedang | blend naik di OOF, tidak di public |
| E120 | analisis kasus salah | FN berprofil bersih, FP berprofil buruk | — | analisis kualitatif 15 timeline | disimpulkan "keacakan label" |
| E126 | pinjaman lama sebagai data berlabel | fitur PD 0,00000 (1 seed); baris tambahan +0,0004 (model 96 fitur) | 1 | lemah | as-of hanya 96 fitur, riwayat tipis |
| E170 | model zona KS | +0,0010 KS / −0,0006 | 1 | lemah | — |
| E141x | DART di set E122x1 | +0,00020 vs E122x1, 3/5, P 0,85 | 3 | sedang | kandidat |

Pola umum: kenaikan besar hanya datang dari **informasi baru** (agregat semua tabel E03 +0,17; jendela e-commerce E50 +0,004; model order E121 +0,0046; urutan order E122 +0,001). Tuning, seed, blend, seleksi masing-masing ≤ +0,0005. Semua sumber kenaikan ≥ 0,001 sejak E50 berasal dari e-commerce.

### 1.6 Hal janggal / perlu dipertanyakan

1. **`src/features/` tidak ter-version**: pola `features/` di `.gitignore` juga cocok dengan `src/features/`. Builder 528 kolom tidak ada di git (0 file ter-track). Risiko besar untuk notebook final (harus inline semua builder) dan kehilangan kode. Perbaikan: ubah jadi `/features/` lalu commit (minta izin).
2. **Script pembuat submission public terbaik (`E122x1_E141_rank50`) tidak tersimpan** di repo (hanya varian `make_sub_e141x.py`, belum di-commit). Sudah saya rekonstruksi; MD5 cocok.
3. **Early stopping memakai fold validasi luar** (`train.run_lgb_cv`, `tuning.fit_fold`): iterasi terbaik dipilih pada data yang dinilai → OOF sedikit optimis. Bias sama untuk semua eksperimen, tetapi memengaruhi perbandingan dengan DART (tanpa ES) dan NN `_iv`.
4. **Parameter tuned berasal dari set lain**: tuning di B3 (176 kolom, fold 0–2, lr 0,05) lalu dipakai untuk 541 kolom. feature_fraction 0,129 berarti ±23 kolom per pohon di B3 tetapi ±70 di set sekarang. Satu-satunya tuning ulang (E146) hanya di fold 0–2. Tuning 5 fold di set sekarang belum pernah dilakukan.
5. **Penolakan 1 seed di Fase 2**: Δ −0,0007 … +0,0001 dengan std antar seed ±0,0006 → bukti lemah. E71/E130a membuktikan grup-grup itu berisi kolom berguna. Kesimpulan "fitur X ditolak" dari Fase 2 tidak boleh dipakai untuk menutup arah baru.
6. **Fitur nested bagian latih berasal dari model 60% data, bagian validasi/test dari model 80% data** (pergeseran skala; sudah diatasi untuk seq_* dengan persentil per blok, ecm_* tidak, E122rr menunjukkan efeknya nol).
7. **Kolom `diterima` di log tidak bisa dipercaya untuk eksperimen `e121.py`/`e141.py`**: selalu ditulis "kandidat"; keputusan nyata ada di CLAUDE.md/Riset.
8. **Importance model acuan sekarang tidak tersimpan** (file importance terakhir E87, set B3). Kita tidak tahu kolom apa yang diandalkan E122x1/E141x.
9. **Analisis kasus salah E120 hanya 15 timeline** dan menyimpulkan "keacakan label". Kesimpulan ini kuat terhadap FN berprofil bersih, tetapi belum menguji pola sistematik (mis. waktu relatif, jadwal cicilan, urutan kejadian lintas tabel) pada ribuan baris.
10. **E126 (pinjaman lama berlabel)** diuji dengan model 96 fitur sendiri, bukan dengan fitur kaya model utama; sinyal "+0,0004 sebagai baris tambahan" diukur pada model yang lebih lemah. Arah ini belum benar-benar teruji.
11. **Tidak ada fitur kalender keputusan** (tanggal dalam bulan, jarak ke tanggal jatuh tempo, hari kerja). Probe 12 Okt hanya mengecek efek bulan dan region × bulan.
12. Prompt menyebut public terbaik 0,55858 (#39); LOG.md mencatat 0,55951 (16 Okt). Perlu konfirmasi angka leaderboard terbaru.

### 1.7 Jadwal sampai feature freeze 26 Okt (usulan)

| Tanggal | Kegiatan |
|---|---|
| 17 Okt malam | Langkah 1 (selesai) + batch #1; mulai Langkah 2 |
| 18 Okt | Langkah 2: proses pembangkitan data, relasi antartabel, kehidupan customer, kasus salah → batch #2; Langkah 3: ≥ 15 hipotesis → serahkan, berhenti |
| 19 Okt | Gabung ide user + batch #3; diff compare_oof/skill/CLAUDE.md; nilai ulang eksperimen lama dengan aturan baru; mulai uji hipotesis #1–#3 |
| 20–24 Okt | Siklus builder → audit → screen (jalur cepat) → CV 3 seed → bootstrap; 2–4 hipotesis/hari di laptop, komponen berat di komputer lain/Kaggle; putaran hipotesis kedua dari kegagalan |
| 25 Okt | Gabungan semua yang diterima + ablation + seed baru (konfirmasi); tuning 5 fold bila set fitur banyak berubah |
| 26 Okt | **Feature freeze**; daftar kandidat final (best-CV + robust) |
| 27–29 Okt | Notebook final self-contained, fresh run ×2, mesin kedua, MD5 |
| 30 Okt ≤ 12.00 | Upload notebook |

---

## 2. Pertanyaan batch #1 (dikirim 17 Okt malam)

**A. Posisi dan leaderboard**
1. Public terbaik kita sekarang 0,55858 (prompt) atau 0,55951 (`E122x1_E141_rank50`, LOG.md 16 Okt)? Berapa skor #1, #8, #10 terbaru? — *menentukan gap nyata dan apakah 8 besar realistis.*
2. Berapa persen test yang dipakai public LB (tertulis di halaman Leaderboard Kaggle)? — *menentukan noise public dan seberapa besar private bisa berbeda.*
3. Berapa submission final yang boleh dipilih di Kaggle (1 atau 2)? Jika 2, mana yang dicocokkan dengan notebook? — *strategi best-CV vs robust.*

**B. Data**
4. Apakah panitia pernah menyebut asal data (simulasi/sintetis atau data nyata yang dianonimkan)? — *kalau sintetis, struktur pembangkitnya bisa direkonstruksi dari data itu sendiri (sah, data panitia).*
5. Dictionary bertuliskan "v2". Apakah pernah ada v1 atau update data selama lomba, dan apa yang berubah? — *perubahan versi biasanya menutup kebocoran; itu petunjuk "sambungan" pembangkit data.*
6. Sejak 12 Okt, apakah tab Discussion/Code di Kaggle sudah dicek lagi (pengumuman host, notebook publik skor tinggi)? — *hanya untuk tahu arah; tidak disalin.*

**C. Notebook final**
7. Apakah panitia menjalankan ulang notebook, atau hanya mencocokkan output yang tersimpan di .ipynb dengan submisi terpilih? Jika dijalankan ulang: di mesin apa dan ada batas waktu? — *menentukan apakah komponen GPU/berat boleh masuk final.*
8. Komposisi notebook final sudah diputuskan (dengan atau tanpa fitur GRU dari GPU)? — *acuan yang saya tingkatkan harus pipeline yang benar-benar masuk final; kalau tanpa GRU, acuannya E130a/E141, bukan E122x1.*

**D. Komputasi dan cara kerja**
9. Spesifikasi "komputer lain yang lebih kuat" (CPU/core, RAM, GPU, OS) dan kapan tersedia? — *untuk tag komputasi; LightGBM bisa beda hasil antar OS/jumlah thread sehingga MD5 berubah.*
10. Sisa kuota GPU Kaggle minggu ini? — *untuk komponen urutan/NN.*
11. Untuk Langkah 4 (eksperimen yang menulis `oof/`, `preds/`, `experiments/log.csv`, Riset, CLAUDE.md), boleh saya bekerja langsung di checkout utama? Saat ini saya terisolasi di worktree; checkout utama punya perubahan belum di-commit (log.csv, Riset, CLAUDE.md, LOG.md). — *supaya tidak menimpa pekerjaan Anda.*
12. Boleh saya perbaiki `.gitignore` (`features/` → `/features/`) dan meng-commit `src/features/`? Apakah repo GitHub `mikirui` privat (boleh push branch)? — *kode fitur saat ini tidak ter-version; push ke repo publik bisa melanggar larangan kerja sama.*
13. Apakah ada anggota tim yang sedang/akan menjalankan job di laptop, dan jam berapa laptop bebas untuk job panjang? — *RAM 7,7 GB; dua job bersamaan pernah membuat pipeline mati.*

---

## 3. Langkah 2 — Data sampai level baris

Script: `s01`–`s18` di folder ini (semua hanya membaca checkout utama). Residual = target − OOF E122x1 (bag 3 seed) yang dikalibrasi logistik; z = korelasi Pearson rank(fitur) vs residual × √n (noise |z| ≈ 1; |z| > 3 = sinyal di luar model). Alat: `resid.py`.

### 3.1 Bagaimana data dibangkitkan (bukti per tabel)

| Tabel | Struktur yang ditemukan (bukti) |
|---|---|
| Aplikasi sekarang | Keputusan jam 08.00–20.59, detik selalu 0, jam seragam; lama proses 0,2–120 jam; nominal kelipatan Rp100 rb; cicilan = anuitas persis; 5 tier bunga; target per bulan 12,0–14,7%, per hari/jam keputusan rata. |
| Profil | Umur 23–60, income kontinu (bukan bulat), tanggungan 0–4; korelasi Spearman nominal pinjaman ~ income 0,81. |
| Aplikasi lama | 1–11 per customer (70% customer), Jan 2023–Des 2025; tenor 6/12/18 sama rata; 72% disetujui penuh, sisanya dipotong ke 73–100%; anuitas persis; lama proses sama sebarannya dengan aplikasi sekarang. |
| Cicilan | Due = tanggal-jam pencairan + k bulan; `amount_due` = cicilan terjadwal (yang terakhir menyesuaikan pembulatan). **Pembayaran hanya terjadi tepat di ulang bulan due**, dialokasikan FIFO ke cicilan tertua, tidak pernah dibayar di muka. Per event customer membayar 0–3 cicilan. DPD karenanya terkuantisasi (0/28–31/59–62/88–92/…). Event berlanjut setelah tenor selama masih menunggak. Default = tunggakan mencapai 3 cicilan pada event 1–9 (setara DPD ≥ 90 dalam 12 bulan). |
| Biro | 1–3 provider per aplikasi (sekarang dan lama, 100%). Laporan dibuat ≤ 120 hari sebelum aplikasi, tersedia di antara waktu aplikasi dan keputusan. Skor = laten customer **stabil** + noise per laporan independen: korelasi antar provider 0,81, antar aplikasi 0,90–0,91 untuk jarak 0–60 hari maupun > 1 tahun, deviasi provider tidak menetap antar aplikasi (r = 0,005). |
| Rekening | Akuntansi tertutup sempurna (opening = closing bulan lalu; closing − opening = inflow − outflow; low days > 0 ⇔ min < 250 rb). Panjang riwayat seragam 8–23 bulan, selalu berakhir bulan sebelum keputusan. Ada untuk persis 70% customer yang sama dengan pemilik aplikasi lama. Cicilan dibayar dari rekening: di bulan lewat-bayar outflow turun dan saldo naik. |
| E-commerce | Riwayat 2 tahun, **dipotong tepat 24 jam sebelum keputusan**. Jam order seragam 0–23 (sintetis). Jumlah order **tidak** terkait target (AUC 0,49); **sebaran waktunya** terkait: rasio-kemungkinan per order +1,03 di 1 hari sebelum keputusan → +0,4 di 25 hari → 0 di ±140 hari → −0,39 di 650 hari. Defaulter: nominal order ±22% lebih besar dan COD +3–6 poin di semua jendela. |

### 3.2 Kehidupan customer dan relasi antartabel

- Customer dengan riwayat (70%): aplikasi lama sejak 2023 → tiap aplikasi mendapat laporan biro → yang disetujui menjadi pinjaman dengan event bayar bulanan → rekening 8–23 bulan terakhir → e-commerce 2 tahun → aplikasi sekarang (laporan biro, tier, keputusan). Customer tanpa riwayat (30%): hanya profil, aplikasi, biro sekarang, e-commerce.
- **Satu risiko laten per customer yang stabil** menggerakkan semua pinjamannya:
  - Laju lewat-bayar per event **datar sepanjang 23 bulan** sebelum keputusan: 0,33–0,37 untuk customer yang nantinya default vs 0,15 untuk yang tidak (s06). Tidak ada tren menjelang keputusan.
  - Korelasi lewat-bayar antarpinjaman milik customer yang sama: rasio vs independen 1,60 di bulan kalender sama, turun ke 1,40 di lag 6–12 bulan → sebagian besar heterogenitas customer tetap, plus guncangan bulanan kecil.
  - **Label default pinjaman lama sangat berulang**: ICC 0,40 antar pinjaman customer yang sama (81.745 pinjaman berjendela lengkap); P(default | pinjaman lain default) 0,44 vs 0,09 (s14).
  - **Target sekarang = "satu pinjaman lagi"** (s14):

    | n pinjaman lama lengkap, k default | P(target) | P(pinjaman lama terbaru default \| k dari n lainnya) |
    |---|---|---|
    | n=1, k=0 / k=1 | 0,109 / 0,463 | 0,085 / 0,435 |
    | n=2, k=0 / 1 / 2 | 0,077 / 0,309 / 0,501 | 0,077 / 0,323 / 0,643 |
    | n=3, k=0 / 1 / 2 / 3 | 0,058 / 0,177 / 0,384 / 0,585 | 0,076 / 0,272 / 0,411 / 0,600 |

  - Di dalam customer yang sama, ciri pinjaman hampir tidak menentukan pinjaman mana yang default (pasangan beda label: tier lebih tinggi 0,28 vs lebih rendah 0,30; PTI 0,52/0,48; LTI 0,55/0,44) (s16). Tier adalah penilaian lender atas **customer**, bukan atas pinjaman.
- **Sinyal "ramp" khusus keputusan sekarang**: dalam 1–3 bulan terakhir, defaulter punya outflow/income 0,95 vs 0,81, arus bersih 0,16 vs 0,31, hari saldo rendah 1,64 vs 1,24 (s13); order e-commerce 2–2,8× lebih rapat dalam 7 hari terakhir. Ramp ini **tidak ada sebelum keputusan pinjaman lama** (rasio laju 7 hari 1,19 vs 1,12; P(order 7 hari) 0,28 vs 0,34) dan **bergantung langsung pada target**, tidak hanya pada risiko customer: di kuintil risiko (posterior pinjaman lama) terendah, order 7 hari defaulter 1,04 vs 0,53 (s17).

### 3.3 Batas informasi

- Dengan risiko default per customer π ~ Beta(0,19; 1,33) (rata-rata 0,123, ICC 0,40), **AUC maksimum jika π diketahui sempurna = 0,927** (simulasi 400 rb, s14). Kita di 0,845. Batasnya bukan keacakan label, melainkan ketepatan menaksir π (ditambah ramp yang bergantung pada target).
- Namun hasil pinjaman lama **sudah dipakai penuh** oleh model: posterior beta-binomial dengan definisi target persis + sensor untuk pinjaman belum lengkap: AUC 0,708, z residual −0,06; model sudah terkalibrasi persis mengikutinya per desil (s15).

### 3.4 Kasus salah model terbaik (E122x1, s17)

| Kelompok | n | punya riwayat | π posterior | q0 | jam order terakhir (median) | order 7 hari | ramp outflow | skor biro | tier |
|---|---|---|---|---|---|---|---|---|---|
| FN: y=1, skor < median | 890 | 0,66 | 0,079 | 0,139 | 412 | 0,36 | 1,02 | 676 | 0,205 |
| TN pembanding: y=0, skor < median | 34.109 | 0,66 | 0,073 | 0,114 | 556 | 0,30 | 0,94 | 688 | 0,200 |
| FP: y=0, skor > p95 | 1.197 | 0,79 | 0,389 | 0,304 | 47 | 2,43 | 1,36 | 619 | 0,245 |
| TP pembanding: y=1, skor > p95 | 2.303 | 0,82 | 0,424 | 0,314 | 44 | 2,64 | 1,38 | 614 | 0,249 |

- FN hampir identik dengan TN berskor sama; FP identik dengan TP berskor sama. Kesalahan individual berasal dari undian Bernoulli pada π yang sama, bukan pola yang terlewat (menguatkan E120 dengan ribuan baris).
- **Baterai sinyal baru** (s07, s08, s09, s11, s13, s15): hanya **proses bayar level event** yang membawa sinyal di luar model — laju mengejar tunggakan `c` z 6,3; laju lewat-bayar saat lancar `q0` z 4,0; bulan sejak lewat-bayar terakhir z −3,0; event setelah tenor z 2,2. Screen tim (11 kolom, `cand_ev_v1.parquet`): **+0,00069 mean-fold AUC, 5/5 fold, |z| maks 6,3 → lolos**. Semua lainnya |z| < 2: skor biro gabungan semua laporan, tier gabungan, keputusan lender, LLR waktu e-commerce (jumlah/rata/30 hari), atribut order, rekening relatif income, waktu biro/aplikasi, order setelah aplikasi, posterior pinjaman lama.
- **Cek kebocoran** (s11): posisi baris customer di 6 tabel, keterurutan order/aplikasi lama, pola hex `loan_id`/`application_id` → semua AUC 0,500–0,503, |z| ≤ 1,4. Tidak ada kebocoran dari data panitia yang terdeteksi.

### 3.5 Data berlabel tambahan dan kurva belajar

- **Pinjaman lama = label dari mekanisme yang sama** (s20). Regresi logistik 7 fitur dengan bukti *leave-one-loan-out* (hasil pinjaman lengkap lain, laju lewat-bayar event pinjaman lain, skor biro, tier): AUC **0,825** untuk default pinjaman lama tenor 12 (27.868 pinjaman) dan **0,817** untuk target sekarang (customer punya riwayat). Dilatih di pinjaman lama → dinilai di target sekarang: **0,808**; arah sebaliknya 0,817. Koefisien laju lewat-bayar hampir sama (9,3 vs 9,2); tier jauh lebih kuat untuk pinjaman sekarang (0,77 vs 0,15).
- **Mengapa E126 gagal**: desain *as-of* membuang bukti yang datang setelah pinjaman lama itu (pinjaman berikutnya, pembayaran berikutnya); karena risiko customer stabil, bukti itu sah dan justru yang terkuat (AUC model PD E126 0,704 vs 0,825 di sini).
- **Kurva belajar** (s18, set E130a, LGB tuned lr 0,05, 1 seed, 5 fold): 14k baris 0,8351 → 28k 0,8392 → 42k 0,8414 → 56k 0,8421. Kurva pangkat: AUC = 0,8479 − 3,56·n^−0,589 → 63k (10-fold) 0,8426, 70k (refit) 0,8429, 112k (2× data) 0,8441, 140k 0,8446. **Model masih lapar data (+0,002 untuk 2× data), tetapi asimtot representasi sekarang ≈ 0,848** (≈ 0,851 untuk versi lr 0,02 bag 3 seed). Data saja tidak cukup untuk 0,858.

### 3.6 Uji tambahan yang negatif (supaya tidak diulang)

| Uji | Hasil |
|---|---|
| Interaksi ramp × risiko (z residual ramp per kuintil skor model) | semua dalam noise (maks 2,4 dari 20 uji) |
| Ramp biro (laporan sekarang − laporan lama: skor, utang, inquiry, delinquent, limit, util) | \|z\| ≤ 2,1 |
| Aplikasi lama menjelang keputusan (14/30/60/90 hari) | tidak pernah ada dalam 14 hari; \|z\| ≤ 0,4 |
| Guncangan kalender pada lewat-bayar | P(miss) per bulan 0,173–0,184; sd antar bulan 0,0025 vs sampling 0,0017 |
| Ciri pinjaman sekarang relatif pinjaman lama (LTI, PTI, tier, nominal) | \|z\| ≤ 1,0 |
| Segmen tanpa riwayat: sinyal e-commerce | \|z\| ≤ 1,7 |

### 3.7 Implikasi untuk hipotesis

1. Gap ke 0,858 tidak bisa ditutup dengan "pola baru" pada customer yang salah prediksi; harus dari **taksiran risiko customer (π) yang lebih tajam** untuk semua customer, dan dari **ekstraksi informasi baru** (asimtot representasi sekarang ≈ 0,848–0,851).
2. Sumber yang terbukti masih punya sisa: **rantai bayar level event** (screen +0,0007) dan **data berlabel tambahan dari pinjaman lama** dengan bukti LOO (kurva belajar: 2× data ≈ +0,002; transfer lintas tipe pinjaman 0,808).
3. Sumber yang terbukti jenuh: hasil pinjaman lama sebagai hitungan, biro, tier, keputusan lender, waktu e-commerce, rekening, waktu aplikasi/biro, interaksi ramp.
4. Perkiraan jujur: semua tuas yang punya bukti dijumlahkan ≈ +0,003 s.d. +0,006. Sisa gap butuh putaran penemuan berikutnya (Langkah 4), terutama pada **mekanisme ramp** (sinyal yang bergantung langsung pada target sekarang) dan **cara lender menetapkan tier** sekarang (koefisien tier 5× lebih kuat untuk pinjaman sekarang daripada pinjaman lama).

---

## 4. Pertanyaan batch #2 (dikirim 18 Okt dini hari)

1. **Pinjaman lama milik customer test sebagai baris latih model bantu** (18.415 dari 81.745 pinjaman lengkap; label dihitung dari `repayment_history`, bukan dari target test). Saya menilainya sah (data panitia, target test tidak disentuh), tetapi ini pemakaian transduktif data test. Boleh? — *menentukan desain H02/H06; tanpa itu label tambahan turun ke ±63 rb.*
2. **Sinyal ramp** (order e-commerce rapat dan outflow rekening naik dalam 1–3 bulan terakhir) hanya muncul sebelum keputusan sekarang dan bergantung langsung pada target. Menurut saya ini bukan kebocoran: informasinya tersedia sebelum keputusan (e-commerce dipotong 24 jam sebelum keputusan), lebih mirip "jejak digital" yang memang disuntikkan pembangkit, dan sudah dipakai model sejak E03. Setuju tetap dipakai? — *aturan: pola mirip kebocoran wajib dilaporkan.*
3. **Definisi label pinjaman lama**: saya memakai "tunggakan ≥ 3 cicilan pada event 1–9" (setara DPD ≥ 90 dalam 12 bulan sejak `decision_datetime`). Apakah tim tahu titik awal "12 calendar months" versi panitia? Jika "tidak tahu", saya uji beberapa varian definisi dan memilih yang paling konsisten dengan pola target sekarang. — *label H02.*
4. **Prioritas**: tuas yang punya bukti ≈ +0,003–0,006, asimtot representasi sekarang ≈ 0,848–0,851. Urutan yang Anda inginkan: (a) amankan kenaikan pasti dulu (H01, H02, H05) lalu penemuan, atau (b) langsung paralel dengan eksperimen penemuan berisiko (NN multi-tugas, mekanisme ramp/tier)? — *urutan Langkah 4.*
5. **Batas waktu notebook final**: model bantu PD pinjaman (H02) menambah ±30–60 menit CPU. Apakah ada batas waktu run (terkait batch #1 no. 7)? — *desain final.*

---

## 5. Langkah 3 — Hipotesis (diserahkan 18 Okt; menunggu daftar ide dari user)

Skor = Bukti (1–5) × Dampak (1–5) × Kebaruan (1–3). Bukti: 5 = sinyal residual |z| > 4 atau screen lolos di data kita; 4 = bukti struktural kuat; 3 = bukti tidak langsung yang konsisten; 2 = prinsip umum + petunjuk lemah; 1 = tanpa bukti. Dampak: 5 ≥ +0,003; 4 = +0,0015–0,003; 3 = +0,0007–0,0015; 2 = +0,0003–0,0007; 1 < +0,0003. Kebaruan: 3 = belum pernah; 2 = variasi eksperimen lama dengan alasan berbeda; 1 = pengulangan. Biaya tidak ikut menentukan urutan. Tag reproduksibilitas: **CPU-det** = CPU-deterministik, **GPU**, **cek** = butuh cek.

| # | Hipotesis | B | D | K | Skor | Perkiraan Δ AUC | Komputasi | Tag |
|---|---|---|---|---|---|---|---|---|
| H01 | Rantai bayar bulanan level event | 5 | 3 | 3 | **45** | +0,0005–0,001 | CPU < 5 mnt build, CV 3 seed ±20 mnt, RAM 2 GB | CPU-det |
| H02 | Model bantu PD pinjaman dengan bukti leave-one-loan-out | 4 | 4 | 2 | **32** | +0,001–0,003 | CPU 1–1,5 jam total, RAM 3 GB | CPU-det |
| H03 | Model bantu "sifat statis" untuk segmen tanpa riwayat | 3 | 3 | 3 | **27** | +0,0005–0,0015 | CPU ±1 jam, RAM 3 GB | CPU-det |
| H04 | NN multi-tugas: encoder customer atas semua pinjaman | 3 | 3 | 3 | **27** | +0,0007–0,002 | GPU 3–6 jam (Kaggle T4/komputer lain) | GPU |
| H05 | 10-fold bersarang di 5 fold + refit 100% untuk test | 4 | 2 | 3 | **24** | +0,0003–0,0007 CV | CPU 2× CV (±40 mnt 3 seed) | CPU-det |
| H06 | Pinjaman lama sebagai baris latih model utama | 3 | 4 | 2 | **24** | +0,001–0,003 (tumpang tindih H02) | CPU builder LOO 528 kolom: dev 1 hari, run 1–2 jam, RAM 4–6 GB (komputer lain) | CPU-det |
| H07 | Sub-model per sumber bukti (nested) + stacking | 2 | 3 | 3 | **18** | +0,0005–0,002 | CPU 1–2 jam, RAM 3 GB | CPU-det |
| H08 | Posterior "tipe default" untuk pinjaman yang masih berjalan | 3 | 2 | 3 | **18** | +0,0002–0,0006 | CPU < 30 mnt | CPU-det |
| H09 | Status bulan kalender terakhir lintas semua pinjaman | 3 | 2 | 3 | **18** | +0,0002–0,0005 | CPU < 5 mnt | CPU-det |
| H10 | GRU pada rantai event bulanan (kalender) | 3 | 2 | 2 | **12** | +0,0003–0,001 | GPU ±2–3 jam | GPU |
| H11 | Ramp e-commerce bersyarat risiko (LLR per strata π) | 2 | 2 | 3 | **12** | +0,0002–0,0007 | CPU < 1 jam | CPU-det |
| H12 | Tuning ulang Optuna 5 fold di set sekarang | 2 | 2 | 2 | **8** | +0,0003–0,001 | CPU 5–8 jam (komputer lain) | CPU-det |
| H13 | Pangkas kolom yang bertentangan dengan pembangkit + nilai ulang E130b | 2 | 2 | 2 | **8** | +0,0002–0,0005 | CPU 20–40 mnt/varian | CPU-det |
| H14 | Rekonstruksi skor kontinu lender di balik tier | 2 | 2 | 2 | **8** | +0,0002–0,0007 | CPU < 1 jam | CPU-det |
| H15 | Pseudo-label test (protokol per fold) | 1 | 2 | 3 | **6** | 0 – +0,0005 | CPU ±1 jam | CPU-det |
| H16 | Seed bag 5–10 + DART untuk final | 5 | 1 | 1 | **5** | +0,0001–0,0003 | CPU 1–2 jam | CPU-det |

### Rincian per hipotesis

**H01 — Rantai bayar bulanan level event.**
- *Bukti*: pembayaran hanya terjadi di ulang bulan due, FIFO, 0–3 cicilan per event (s02, s05). Laju lewat-bayar stabil 23 bulan dan memisahkan target (0,33–0,37 vs 0,15, s06). Residual E122x1: `c` (laju mengejar) z 6,3, `catch_rate` 5,1, `q0` 4,0, `miss_post` 3,4, bulan sejak lewat-bayar terakhir −3,0, event pasca-tenor 2,2; fitur DPD lama z ≈ 0 (s07, s10, s11). Screen tim 11 kolom: **+0,00069, 5/5 fold** (s12). `miss_post` AUC univariat 0,808 (> `rp_share_dpd1` 0,784).
- *Definisi (as-of cutoff = `decision_datetime` aplikasi sekarang)*: per pinjaman lama, event j = 1..J, J = jumlah ulang bulan pencairan yang < cutoff; event setelah tenor hanya dihitung selama masih menunggak. X_j = jumlah cicilan yang lunas penuh dengan `last_payment_datetime` di event j; A_j = min(j, tenor) − ΣX. Per customer: n0/m0 (event dan lewat-bayar saat A_prev = 0), n1/m1 (saat A_prev ≥ 1), p1/c1 (event bayar saat menunggak; yang membayar ≥ 2), `q0`, `q1`, `c` = posterior Beta (prior method of moments dari semua customer, tanpa target), `pd_struct` = P(A mencapai 3 dalam 9 event | q0, q1, c) lewat DP, bulan kalender sejak lewat-bayar terakhir, jumlah event pasca-tenor, laju lewat-bayar 3/6/12 bulan kalender terakhir. Prefix `rp_ev_`.
- *Dampak*: screen +0,0007; CV penuh biasanya 60–100% dari screen → +0,0005–0,001.
- *Kebocoran*: tanpa target; hanya cicilan due < cutoff dan alokasi pembayaran < cutoff (dictionary). AUC univariat `miss_post` 0,81 → audit wajib: cek cutoff kode, KS train-test, adversarial. Uji label acak tidak perlu (tanpa target).
- *Kebaruan*: belum pernah (E10–E13 memakai DPD per cicilan; GRU seqx_rp memakai baris cicilan, bukan event).

**H02 — Model bantu PD pinjaman dengan bukti leave-one-loan-out (LOO).**
- *Bukti*: ICC label default antarpinjaman customer 0,40; target sekarang = "satu pinjaman lagi" (tabel 3.2); regresi logistik LOO 7 fitur AUC 0,825 (pinjaman lama) vs 0,817 (target sekarang), transfer silang 0,808 (s20); kurva belajar: 2× data ≈ +0,002 (s18). E126 gagal karena desain as-of membuang bukti setelah pinjaman (AUC 0,704).
- *Definisi*: baris = pinjaman lama dengan jendela 12 bulan lengkap sebelum cutoff (81.745 dari semua 100 rb customer; varian: hanya tenor 12, 27.868, + tenor sebagai fitur). Label = A ≥ 3 pada event 1–9. Fitur = bukti customer **tanpa** pinjaman itu: hasil pinjaman lengkap lain (k, n, tersensor), statistik H01 dihitung tanpa event pinjaman itu, semua laporan biro, profil, ciri pinjaman itu (tier, tenor, LTI, PTI, potongan approved/requested), sifat e-commerce stabil (N order, porsi COD, nominal) **tanpa** ramp sekarang. LightGBM, CV GroupKFold per customer untuk memilih iterasi. Prediksi aplikasi sekarang dengan bukti penuh + ciri pinjaman sekarang → `x_auxpd` (probabilitas dan logit) sebagai fitur model utama.
- *Dampak*: +0,001–0,003 (setara sebagian "2× data" untuk komponen π, ditambah pemetaan bukti yang lebih bersih).
- *Kebocoran*: target sekarang tidak pernah dibaca → tidak perlu nested. Risiko: fitur pinjaman itu memuat dirinya sendiri (harus dikecualikan event & hasilnya); guncangan bulan bersama membuat baris pinjaman lama sedikit optimis (tidak menyentuh target). Pinjaman lama milik customer train dan test diperlakukan simetris. Audit: (1) acak label pinjaman lama → `x_auxpd` harus kehilangan nilai di CV; (2) adversarial train vs test; (3) grep kode: kolom `target` tidak dibaca.
- *Kebaruan*: variasi E126 dengan alasan berbeda (bukti LOO, bukan as-of; termasuk pinjaman customer test).

**H03 — Model bantu "sifat statis" untuk segmen tanpa riwayat.**
- *Bukti*: segmen tanpa riwayat (30%) hanya punya biro, profil, aplikasi, e-commerce; AUC model di segmen ini 0,81–0,83 vs 0,85 punya riwayat. Defaulter punya nominal order +22% dan COD +3–6 poin di **semua** jendela (sifat stabil, s09). Kurva belajar: model masih lapar data.
- *Definisi*: model PD dilatih pada pinjaman lama (seperti H02) **hanya** dengan fitur yang juga dimiliki customer tanpa riwayat: profil (umur disesuaikan ke waktu pinjaman), biro laporan pinjaman itu, ciri pinjaman, sifat e-commerce stabil (tanpa 90 hari terakhir sebelum keputusan sekarang). Diterapkan ke semua aplikasi sekarang → `x_staticpd`.
- *Dampak*: +0,0005–0,0015 (terutama segmen tanpa riwayat).
- *Kebocoran*: sama dengan H02; tambahan: jangan memakai e-commerce di jendela ramp sekarang.
- *Kebaruan*: belum pernah.

**H04 — NN multi-tugas: encoder customer atas semua pinjaman.**
- *Bukti*: sama dengan H02 (mekanisme bersama); NN setara GBDT di data ini (TabM 0,8448).
- *Definisi*: sampel = pinjaman (lama lengkap + aplikasi sekarang). Input = urutan pinjaman lain milik customer (ciri, rantai event, hasil) + profil/biro + ciri pinjaman target + (khusus aplikasi sekarang) ramp. Head biner default 12 bulan, bobot tugas per tipe baris. Baris aplikasi sekarang nested per fold (validasi fold k tidak pernah dilatih).
- *Dampak*: +0,0007–0,002 sebagai fitur/anggota blend (diversitas + data lebih banyak).
- *Kebocoran*: baris target sekarang wajib nested; uji label acak (acak target dan label pinjaman lama).
- *Kebaruan*: belum pernah.

**H05 — 10-fold bersarang di 5 fold + refit 100% untuk test.**
- *Bukti*: kurva belajar: 63k baris → +0,0005, 70k → +0,0008 (s18).
- *Definisi*: tiap fold k dibelah 2 (StratifiedKFold(2, seed 42) di dalam fold) → 10 sub-fold; model dilatih di 9/10 data; mean-fold AUC tetap dihitung pada 5 fold asli. Fitur nested memakai `tr{k}`/`va{k}` yang ada (model luar fold k tidak pernah melihat label fold k → aman untuk kedua sub-fold). Test = rata 10 model; versi refit 100% (iterasi rata-rata × 1,1) hanya untuk submission.
- *Kebocoran*: rendah; audit penyelarasan fold.
- *Kebaruan*: belum pernah (E145 refit tanpa ukuran CV).

**H06 — Pinjaman lama sebagai baris latih model utama.**
- *Bukti*: H02 + kurva belajar.
- *Definisi*: latih model utama pada aplikasi sekarang fold latih ∪ pinjaman lama lengkap (fitur 528 kolom versi LOO + indikator tipe baris); pinjaman lama milik customer validasi fold k (dan grup test yang sesuai) dikeluarkan dari latih fold k agar tidak "menghafal" customer.
- *Kebocoran*: tinggi jika pengelompokan salah (ICC 0,4) → GroupKFold konsisten + uji label acak.
- *Kebaruan*: variasi E126(c) (baris tambahan +0,0004 pada model 96 fitur) dengan fitur LOO kaya.

**H07 — Sub-model per sumber bukti (nested) + stacking.**
- *Bukti*: E121 (sub-model e-commerce) memberi kenaikan terbesar sejak Fase 2 (+0,0046); pembangkit tampak aditif per sumber (π + ramp); model utama depth 4/ff 0,13 jarang menggabungkan banyak kolom satu sumber dalam satu pohon.
- *Definisi*: 4 model nested (format npz): riwayat (rp/pa/bp + H01), rekening (ab), biro+aplikasi+profil, e-commerce agregat; logit OOF → fitur model utama.
- *Kebocoran*: nested + uji label acak.
- *Kebaruan*: baru di level fitur customer (E121d di level baris, +0,0001).

**H08 — Posterior "tipe default" untuk pinjaman yang masih berjalan.**
- *Bukti*: 56.171 pinjaman tersensor; default terjadi di event 3–9 (S1 = 1; 1; 1; 0,83; 0,65; 0,46; 0,29; 0,19; 0,09; 0); posterior beta-binomial dari pinjaman lengkap sudah terpakai (z ≈ 0), tetapi progres tunggakan pinjaman berjalan (A = 1/2 di event terakhir) belum diperlakukan sebagai bukti tipe default.
- *Definisi*: per pinjaman berjalan, P(tipe default | jalur) = π·L1/(π·L1 + (1−π)·L0); L1, L0 = likelihood jalur di bawah model Markov yang diestimasi pada pinjaman lengkap default vs tidak; agregat maks dan jumlah per customer.
- *Kebaruan*: belum pernah.

**H09 — Status bulan kalender terakhir lintas semua pinjaman.**
- *Bukti*: lewat-bayar antarpinjaman di bulan kalender sama rasio 1,60 vs 1,40 di lag 6–12 (guncangan bulanan bersama); bulan sejak lewat-bayar terakhir z −3,0.
- *Definisi*: per customer, untuk 1/2/3 bulan kalender terakhir sebelum cutoff: jumlah pinjaman jatuh tempo, lewat-bayar, tunggakan total, indikator semua pinjaman lewat-bayar.
- *Kebaruan*: belum pernah (rp_window memakai jendela due_date, bukan bulan kalender event lintas pinjaman).

**H10 — GRU pada rantai event bulanan (kalender).** Urutan 24 bulan kalender sebelum cutoff; per bulan: pinjaman aktif, jatuh tempo, dibayar, lewat-bayar, tunggakan total/maks, event pasca-tenor, pencairan baru; nested 5×(1+4)×3 seed seperti E122x. Bukti: H01; GRU baris cicilan (seqx_rp) +0,0004. Variasi E122x dengan representasi berbeda. Tag GPU, determinisme butuh cek.

**H11 — Ramp e-commerce bersyarat risiko.** Bukti: ramp bergantung langsung pada target **di dalam** kuintil π (order 7 hari 1,04 vs 0,53 di kuintil terendah); LLR marginal sudah jenuh (z −1,2) dan z per kuintil dalam noise → bukti lemah bahwa model salah. Definisi: LLR per order dari densitas waktu yang diestimasi per strata π (cross-fit per fold), dijumlahkan per customer. Nested (memakai target) + uji label acak.

**H12 — Tuning ulang Optuna 5 fold di set sekarang.** Bukti: parameter dipilih di B3 (176 kolom, fold 0–2, lr 0,05) lalu dipakai di 541 kolom; E146 (fold 0–2) overfit ke fold tuning; kurva belajar: iterasi naik 629 → 1.361 dengan data. 60–100 trial × 5 fold (lr 0,05), lalu final lr 0,02 3 seed; konfirmasi seed baru. Variasi E146.

**H13 — Pangkas kolom yang bertentangan dengan pembangkit + nilai ulang E130b.** Bukti: laju lewat-bayar datar 23 bulan → kolom jendela/tren rp_ dan ab_ redundan; E130b (333 kolom) +0,00036 (3/5) — status bisa berubah dengan aturan baru. Variasi E130b.

**H14 — Rekonstruksi skor kontinu lender di balik tier.** Bukti: koefisien tier untuk target sekarang 0,77 vs 0,15 untuk pinjaman lama (s20) → tier sekarang memuat informasi spesifik keputusan sekarang; tier diskrit 5 level membuang variasi di dalam tier. Definisi: model ordinal tier ~ semua bukti (tanpa target), cross-fit; fitur = ekspektasi tier, posisi di dalam tier (residual ordinal). Variasi E61 (residual regresi sederhana, netral) dengan model ordinal + bukti lengkap.

**H15 — Pseudo-label test.** Bukti umum (lapar data), tanpa bukti spesifik; label semu dari model fold lain (protokol E160). Dampak kecil.

**H16 — Seed bag 5–10 + DART untuk final.** Bukti terukur (E144 +0,0001; E141x +0,0002, P 0,85). Pengulangan; tetap dipakai di final.

### Ide yang dibuang karena terbantah data (bukti di bagian 3)

| Ide | Bukti yang membantah |
|---|---|
| Hitungan hasil pinjaman lama dengan definisi target persis + sensor | posterior beta-binomial z −0,06 (sudah terserap) |
| Skor biro gabungan semua laporan / berbobot presisi | z −0,9; laten skor stabil tetapi sudah terpakai |
| Agregat rasio-kemungkinan waktu e-commerce (jumlah/rata/30 hari) | z −1,2 |
| Agregat rekening relatif income, lonjakan outflow bulan terakhir | \|z\| ≤ 1,5 |
| Ramp biro sekarang vs laporan lama | \|z\| ≤ 2,1 |
| Fitur waktu aplikasi/biro (lama proses, jam, hari, lag available) | \|z\| ≤ 0,75 |
| Ciri pinjaman sekarang relatif pinjaman lama | \|z\| ≤ 1,0 |
| Bobot peluruhan waktu / jendela recency untuk perilaku bayar (E131) | laju lewat-bayar datar 23 bulan |
| Efek kalender/musiman pada pembayaran | sd antar bulan ≈ sampling |
| Fitur posisi baris / ID (cek kebocoran) | AUC 0,500–0,503 |

---

## 6. Jawaban user (batch #1–#2) dan gabungan dengan ide eksternal

### 6.1 Jawaban yang mengubah rencana (`Jawaban_Pertanyaan.txt`, 17 Okt 21.51)

- Leaderboard: #1 cupuu 0,57768 (21 entri), #2 0,57530, #3 0,57174, #8 0,56832, #10 0,56756. Skor kita tidak disebut ulang → dipakai angka LOG.md (0,55951).
- Bobot nilai: public 45% + private 45% (porsi test untuk public **belum diketahui**). Final: **2 submission** boleh dipilih.
- Data hanya dari `Dataset/`; tidak ada v1; Discussion/Code sudah dicek (tanpa catatan).
- Notebook final **dijalankan ulang panitia**, tanpa batas waktu. Komposisi final belum diputuskan: "cari model yang lebih baik, jangan terpaku model lama".
- Komputer lain dianggap kuat (atau Kaggle/Colab); kuota GPU Kaggle 40 jam; laptop bebas 24 jam.
- Boleh kerja langsung di checkout utama, memperbaiki `.gitignore`, commit `src/features/`, push branch. Dikerjakan: branch `explore-17okt` (commit `382c256`, sudah di-push). Pengaman sesi latar dimatikan **hanya lokal** lewat `.claude/settings.local.json` (`worktree.bgIsolation = none`, di-gitignore).
- Pinjaman lama milik customer test boleh dipakai (tidak melanggar guidebook: data dari panitia, target test tidak disentuh). Sinyal ramp boleh dipakai. Definisi "12 calendar months" tidak diketahui → diuji dengan data. Prioritas: **amankan kenaikan pasti dulu**, lalu eksperimen penemuan.

### 6.2 Ide eksternal (`LAMPIRAN_ide_eksternal.md`) — status setelah dicek dengan data

| # | Ide eksternal | Status | Bukti |
|---|---|---|---|
| 1 | Learning curve | **Sudah dikerjakan** (s18) | 14k 0,8351 → 56k 0,8421; asimtot ≈ 0,848; 2× data ≈ +0,002 |
| 2 | Audit ID & urutan baris tabel pendukung | **Sudah dikerjakan, bersih** (s11, s22) | hex `installment_id`, `loan_id`, `order_id`, `order_item_id`, `bureau_observation_id`, posisi baris 6 tabel, urutan cicilan/pinjaman/provider: AUC 0,500–0,505, \|z\| ≤ 1,4 |
| 3 | Model hazard level cicilan (label milik cicilan, kovariat as-of) | **Masuk daftar (H17)** | lewat-bayar per event bergantung pada guncangan bulanan (bulan dengan income rekening di bawah rata-rata customer: P(miss) 0,329 vs 0,205); q0 punya sinyal residual (z 4,0) |
| 4 | Pinjaman lama sebagai data berlabel (as-of) | **Digabung ke H02/H06 dengan desain LOO** | as-of (E126) AUC 0,704 vs bukti LOO 0,825; varian as-of disimpan sebagai sub-varian |
| 5 | Kelas laten (GMM) / KNN target mean | GMM **terbantah**; KNN → H20 (bukti lemah) | GMM 1–6 komponen: posterior \|z\| ≤ 1,6 (BIC turun hanya karena tier diskrit) |
| 6 | Artefak angka / heaping | **Terbantah** | income kontinu (kelipatan 1.000 hanya 0,1%), digit terakhir seragam; requested/approved selalu kelipatan Rp100 rb; potongan lender tepat 80%/90%; \|z\| ≤ 1,2 |
| 7 | Tujuan pinjaman ↔ belanja | **Terbantah** | selisih univariat besar (household + groceries 90 hari 16,0% vs 7,6%) tetapi hanya cerminan ramp; \|z\| ≤ 1,1 |
| 8 | Pinjam untuk menutup utang (sisa pokok, hari ke due) | **Terbantah (lemah)** | sisa pokok z 1,84; rasio pinjaman/sisa pokok −1,77; hari ke due berikutnya −0,76; tunggakan sekarang −0,63 |
| 9 | Kalender & proxy gajian | **Terbantah** | hari aplikasi, sisa hari bulan, jarak ke tanggal 25: \|z\| ≤ 0,53; tanggal bayar selalu ulang bulan pencairan → tidak bisa jadi proxy gajian |
| 10 | Analisis kasus salah | **Sudah dikerjakan** (s17) | FN/FP tidak bisa dibedakan dari kasus benar berskor sama |
| 11 | OpenFE / Featuretools | **Masuk daftar (H19)**, bukti lemah | 25+ agregat baru semua \|z\| < 2 (jenuh) |
| 12 | Objektif ranking LightGBM | **Masuk daftar (H18)**, bukti lemah | tanpa bukti data; murah diuji |
| 13 | Pseudo-label + refit 100% | **Digabung ke H15 dan H05** | — |

### 6.3 Daftar gabungan, diurutkan ulang (Bukti × Dampak × Kebaruan; seri diurut bukti lalu dampak)

| Urut | ID | Hipotesis (asal) | B | D | K | Skor | Δ AUC perkiraan | Tag |
|---|---|---|---|---|---|---|---|---|
| 1 | H01 | Rantai bayar bulanan level event (sendiri) | 5 | 3 | 3 | **45** | +0,0005–0,001 | CPU-det |
| 2 | H02 | Model bantu PD pinjaman, bukti LOO, termasuk pinjaman customer test (sendiri + ide 4) | 4 | 4 | 2 | **32** | +0,001–0,003 | CPU-det |
| 3 | H03 | Model "sifat statis" untuk segmen tanpa riwayat (sendiri) | 3 | 3 | 3 | **27** | +0,0005–0,0015 | CPU-det |
| 4 | H04 | NN multi-tugas atas semua pinjaman (sendiri) | 3 | 3 | 3 | **27** | +0,0007–0,002 | GPU |
| 5 | H05 | 10-fold bersarang + refit 100% (sendiri + ide 13) | 4 | 2 | 3 | **24** | +0,0003–0,0007 | CPU-det |
| 6 | H06 | Pinjaman lama sebagai baris latih model utama (sendiri + ide 4) | 3 | 4 | 2 | **24** | +0,001–0,003 (tumpang tindih H02) | CPU-det |
| 7 | H17 | Model hazard event bulanan, kovariat as-of, 1,4 jt label (ide 3) | 3 | 2 | 3 | **18** | +0,0003–0,001 | CPU-det |
| 8 | H08 | Posterior tipe default pinjaman berjalan (sendiri) | 3 | 2 | 3 | **18** | +0,0002–0,0006 | CPU-det |
| 9 | H09 | Status bulan kalender terakhir lintas pinjaman (sendiri) | 3 | 2 | 3 | **18** | +0,0002–0,0005 | CPU-det |
| 10 | H07 | Sub-model per sumber bukti + stacking (sendiri) | 2 | 3 | 3 | **18** | +0,0005–0,002 | CPU-det |
| 11 | H10 | GRU rantai event kalender (sendiri) | 3 | 2 | 2 | **12** | +0,0003–0,001 | GPU |
| 12 | H11 | Ramp e-commerce per strata π (sendiri) | 2 | 2 | 3 | **12** | +0,0002–0,0007 | CPU-det |
| 13 | H12 | Tuning Optuna 5 fold di set sekarang (sendiri) | 2 | 2 | 2 | **8** | +0,0003–0,001 | CPU-det |
| 14 | H13 | Pangkas kolom + nilai ulang E130b (sendiri) | 2 | 2 | 2 | **8** | +0,0002–0,0005 | CPU-det |
| 15 | H14 | Skor kontinu lender di balik tier (sendiri) | 2 | 2 | 2 | **8** | +0,0002–0,0007 | CPU-det |
| 16 | H15 | Pseudo-label test nested (sendiri + ide 13) | 1 | 2 | 3 | **6** | 0 – +0,0005 | CPU-det |
| 17 | H16 | Seed bag 5–10 + DART (sendiri) | 5 | 1 | 1 | **5** | +0,0001–0,0003 | CPU-det |
| 18 | H19 | Fitur otomatis OpenFE / Featuretools (ide 11) | 1 | 2 | 2 | **4** | 0 – +0,0005 | CPU-det (berat: 4–10 jam, RAM 16+ GB → komputer lain) |
| 19 | H18 | Objektif ranking LightGBM `rank_xendcg` / `lambdarank` (ide 12) | 1 | 1 | 3 | **3** | 0 – +0,0002 | CPU-det |
| 20 | H20 | KNN target mean nested (ide 5, E127) | 1 | 1 | 3 | **3** | 0 – +0,0003 | CPU-det |

**H17 — Model hazard event bulanan dengan kovariat as-of (ide eksternal 3).**
- *Bukti*: lewat-bayar per event = kecenderungan customer (rasio korelasi 1,42 tetap sampai lag 12) + guncangan bulanan (rasio 1,60 di bulan sama; P(miss) 0,329 di bulan income rekening terendah relatif rata-rata customer vs 0,205 tertinggi); `q0` punya sinyal residual z 4,0; kurva belajar: model lapar data (1,4 jt event vs 70 rb target).
- *Definisi*: baris = event bayar pinjaman lama (j = 1..J, < cutoff); label = lewat-bayar (X = 0 saat ada kewajiban). Kovariat **as-of tanggal event**: riwayat event customer sebelum tanggal itu (laju, tunggakan, jumlah pinjaman aktif), laporan biro dengan `available_datetime` < tanggal event, rekening bulan sebelumnya, e-commerce sebelum tanggal event, profil, ciri pinjaman. LightGBM, GroupKFold per customer. Diterapkan pada kondisi customer di cutoff (12 event hipotetis pinjaman baru) → `x_hazard_q`, `x_hazard_pd12`.
- *Komputasi*: CPU; builder as-of 1–2 hari dev, run ±1–2 jam, RAM 6–10 GB → komputer lain.
- *Kebocoran*: label milik event, bukan target sekarang → tidak perlu nested; kovariat wajib as-of event; uji label acak (acak label event) + adversarial.
- *Kebaruan*: belum pernah (E126 level pinjaman as-of, bukan level event).

**H18 — Objektif ranking LightGBM.** `rank_xendcg` (satu grup per fold) dan `lambdarank` (grup acak 1.000 baris). Bukti a priori lemah: logloss sudah hampir optimal untuk AUC; manfaat paling mungkin diversitas blend. Murah (±20 menit 3 seed).

**H19 — Fitur otomatis (OpenFE / Featuretools, E123/E124).** Bukti lemah: 25+ agregat baru semua |z| < 2 dan kurva belajar menunjukkan batas representasi; OpenFE (interaksi aritmetika) paling mungkin menambah sedikit. Featuretools DFS dengan cutoff per aplikasi 4–10 jam, RAM 16+ GB → komputer lain.

**H20 — KNN target mean (nested, E127).** Tanpa bukti spesifik; GBDT sudah menangkap kemiripan lokal. Murah.

---

## 7. Pertanyaan batch #3

1. **Dua submission final**: apakah notebook final wajib menghasilkan **kedua** file (dua MD5 yang cocok dengan dua submisi terpilih), atau cukup satu? — *menentukan apakah final = 2 pipeline (best-CV + robust) dalam satu notebook.*
2. **Mesin panitia untuk menjalankan ulang** (Kaggle CPU/GPU? laptop panitia? OS?). — *komponen GPU (GRU `seq_*`/`seqx_*` di E122x1, NN) tidak dijamin bit-identik antar mesin; kalau panitia menjalankan di mesin lain, MD5 bisa beda → risiko posisi diganti. Jawabannya menentukan apakah fitur GPU boleh masuk final.*
3. **Porsi test untuk public LB** (tulisan "calculated with approximately X% of the test data" di halaman Leaderboard). Jawaban batch #1 berisi bobot nilai 45/45, bukan porsi test. — *hanya untuk membaca noise submission; kalau tidak tahu, saya pakai 30–50%.*
4. **Persetujuan diff** perubahan aturan (`experiments/explore_17okt/diff_aturan_baru.patch`): `compare_oof.py`, skill run-experiment, skill audit-leakage + `audit_features.py`, CLAUDE.md. Boleh dipakai? — *wajib sebelum eksperimen baru pertama dan sebelum menilai ulang eksperimen lama.*
5. **Usulan perbaikan skill lain yang menghambat temuan sah**: audit-leakage Lapis 3 butir 3 ("fitur model baris harus intrinsik milik baris") melarang fitur konteks customer pada **semua** model level baris, padahal H02/H04/H17 sah memakai bukti dari pinjaman/event lain milik customer karena labelnya **bukan target sekarang**. Usul: batasi butir itu ke model baris yang labelnya target customer (gaya E121); untuk model berlabel sendiri, konteks customer boleh asalkan baris itu sendiri (hasil dan jalurnya) dikecualikan (LOO) dan kovariat as-of untuk model hazard. Setuju?

---

## 8. Persiapan Langkah 4 (menunggu persetujuan diff)

- **Diff siap ditinjau**: `experiments/explore_17okt/diff_aturan_baru.patch` (421 baris). Validasi kode: AUC berbobot cepat = `roc_auc_score(sample_weight)` persis (selisih 0); bootstrap 1000× pada 70 rb baris 4 detik; cek penyelarasan E122x1 vs E122x1 = 0,84495 (sama dengan log). Belum dipakai untuk keputusan apa pun.
- **Builder H01 siap** (tidak memakai aturan baru): `src/features/events.py` → `features/rp_events_v1.parquet` (18 kolom `rp_ev_*`, 66,2% baris terisi, 0,7 menit). Terdaftar di `EXTRA_GROUPS` (bukan `GROUPS`, supaya set "all" 528 kolom tidak berubah); dibangun dengan `python build_extra.py rp_events_v1`; dipakai lewat `e121.py --extra rp_events_v1.parquet`.
- **Urutan setelah disetujui**: (1) nilai ulang eksperimen lama berstatus kandidat/ditolak dengan Δ positif yang punya OOF ≥ 3 seed (E130b, E141, E122r, E122c, E122x1/2/3, E141x, E146, E150_xgb, …) → laporkan perubahan status, acuan tidak diganti sebelum lapor; (2) E200 = H01: audit → screen → CV 3 seed (±20 menit) → `compare_oof`; (3) E201 = H02; (4) E202 = H05.

---

## 9. Langkah 4 — log kerja (18 Okt)

**Jawaban batch #3** (`Jawaban_Pertanyaan.txt`): notebook final wajib menghasilkan **2 file** (keduanya untuk private LB); mesin panitia **tidak diketahui** → kandidat CPU-deterministik diutamakan (PROMPT 3 Langkah 7: kandidat A CPU, kandidat B dengan GPU hanya bila terbukti identik); public LB = **30% test (±9.000 baris)**, private 70%; diff **disetujui**; perbaikan audit-leakage Lapis 3 butir 3 **disetujui** dan diterapkan. Semua di-commit (`6bf2580`).

**Urutan PROMPT 3 (`PROMPT_3_gabungan_kenaikan_kecil.md`)** — saran: Langkah 1 (inventaris) sekarang, karena sama dengan PROMPT UTAMA Langkah 4.4; Langkah 2–7 **setelah** eksperimen PROMPT UTAMA menghasilkan komponen E200+ (target 22/23–25 Okt). Alasan: inti PROMPT 3 menggabungkan semua komponen diterima/kandidat termasuk E200+; menjalankannya sekarang berarti mengulang kerja mahal (DART ±1,6 jam/run, CatBoost ≤ 3 jam, seed baru) setelah H01/H02/H05 masuk; ablation dan koreksi "kutukan pemenang" paling jujur dilakukan sekali pada set lengkap; header PROMPT 3 sendiri menjadwalkannya setelah PROMPT UTAMA.

**Langkah 4.4 / PROMPT 3 Langkah 1 — penilaian ulang** (detail di file Riset bagian 11.1, `reevaluasi_aturan_baru.csv`): 7 berubah status → **Diterima**: E130b, E141, E122c, E122x1, E122x2, E122x3, E140a-sel_b2; **Kandidat**: E122r, E122m, E141x, E140a-B3. **Acuan tetap E122x1** (0,84495).

**E200 = H01 (rantai bayar level event)**:
- Builder `src/features/events.py` → `features/rp_events_v1.parquet` (18 kolom). Tag: **CPU-deterministik**.
- Audit: Lapis 1 kode 9/9 lolos (event & pembayaran < cutoff; tanpa target; train/test satu fungsi; kolom terurut); Lapis 2 `audit_features.py`: ERROR 0, AUDIT 1 (`rp_ev_miss_post` AUC 0,807 → audit wajib: cutoff dicek di kode, KS train-test 0,007, adversarial 0,4993, label acak tidak berlaku karena tanpa target), KS train-test maks 0,010, NaN train = test. **AMAN**.
- Screen (jalur cepat): +0,00059, naik 5/5 fold, |z| maks 6,3 → lolos.
- CV 3 seed: `python e121.py --cv --use <7 npz E122x1> --basefs all --extra rp_events_v1.parquet --name E200` — sedang berjalan.

**E200 hasil CV**: seed 42/43/44 = 0,84619 / 0,84610 / 0,84649; **bag 0,84659** vs E122x1 0,84495 → Δ **+0,00163**, CI 95% [+0,00117; +0,00212], P 1,000, naik 5/5 fold; KS mean-fold 0,5266 → 0,5296 → **DITERIMA**, diusulkan acuan baru (dilaporkan ke user 18 Okt). Kenaikan CV ±3× hasil screen: model tuned memanfaatkan representasi rantai bayar jauh lebih baik dari pohon 7 daun.

**E201 = H02 (model bantu PD pinjaman lama)** — builder `src/features/auxpd.py` (`x_auxpd_v1` di `EXTRA_GROUPS`): model bantu dilatih pada **81.702 pinjaman lama** (default 14,4%), AUC GroupKFold per customer **0,797** (vs desain as-of E126 0,704); fitur terpenting: `ev_miss_post`, `ev_q0_post`, `ev_q1_post`, `ec_log_amount`, skor biro, `ec_orders_per_30d`, LTI. Fitur `x_auxpd` (logit & pd) untuk aplikasi sekarang: **AUC univariat 0,799** terhadap target sekarang (tanpa memakai satu pun target sekarang). Audit: Lapis 1 lolos; Lapis 2 ERROR 0, KS train-test 0,008; adversarial 0,498; **uji label acak** (label pinjaman lama diacak, `x_auxpd_v1_shuffle`): AUC model bantu 0,506, fitur hampir konstan (sd logit 0,15 vs 1,25), screen di atas E200 +0,00005 (2/5), |z| 0,15 → lolos. Screen fitur asli di atas E200: **+0,00022, naik 5/5, |z| maks 1,8** (tidak lolos ambang screen, tapi bukti struktural kuat → CV penuh sesuai aturan baru). CV 3 seed berjalan (`e121.py ... --extra rp_events_v1.parquet+x_auxpd_v1.parquet --name E201`).

**Screen lain di atas E200 (18 Okt)**:
- H08 (peluang "default sedang berlangsung" pinjaman berjalan dari tabel hazard empiris pinjaman lengkap: tunggakan 2 cicilan di event 2–9 → default 45–65%, tunggakan 1 → 11–36%, lancar 2–9%) dan H09 (status bulan kalender 1–3 terakhir lintas pinjaman): Δ −0,00001, |z| ≤ 1,4 → **ditolak di screen** (sudah tertangkap E200).
- Rekening × pinjaman (s26): pencairan pinjaman internal **tidak** masuk rekening (inflow non-gaji bulan pencairan 0,488× income vs 0,495× bulan lain; median 16% nominal pencairan) → rekening hanya terhubung lewat cicilan. Lonjakan inflow tak terjelaskan (indikasi utang lembaga lain) z −1,7; ramp outflow/saldo/belanja relatif baseline sendiri |z| ≤ 1,5 → tidak bersinyal.
- **H21 (baru): rasio inflow non-gaji terhadap income deklarasi** (`features/cand_h21_v1.parquet`, 5 kolom): z residual maks **4,55** (lolos screen), tetapi boosting hanya +0,00006 (3/5) → CV penuh antre prioritas rendah (perkiraan < +0,0003).

**Persiapan berikutnya**: `experiments/run_e202.cmd` (H05, 10-fold bersarang, ±33 mnt; basis disesuaikan dengan keputusan E201) dan `experiments/run_e203.cmd` (H12 pendahuluan: 8 konfigurasi parameter, lr 0,05, 1 seed, ±30–50 mnt) untuk menilai apakah Optuna 5 fold (>3 jam, perlu izin) layak.

**E201 hasil CV**: seed 42/43/44 = 0,84628 / 0,84571 / 0,84593, bag 0,84627 vs E200 0,84659 → Δ −0,00032, P 0,086, naik 0/5 → **DITOLAK**. Pelajaran: model utama sudah menaksir risiko customer dari riwayat hampir sebaik model yang dilatih dengan 82 rb label tambahan; "lapar data" di kurva belajar bukan berasal dari pemetaan riwayat → risiko. Konsekuensi: H06 (pinjaman lama sebagai baris latih), H03 (sifat statis), H04 (NN multi-tugas), H17 (hazard event) turun prioritas — mekanismenya sama (menaksir π dari bukti riwayat/sifat stabil).

**Rantai job 18 Okt 11.14** (`experiments/run_e202_e204.cmd`, Start-Process): E202 (H05 10-fold) → E203 (sensitivitas parameter) → E204 (H21). Acuan tetap **E200**.

**Putaran penemuan 2 (18 Okt, ringan)**:
- Pola tanggal kalender e-commerce (hari dalam bulan/minggu, tanggal lonjakan, ramp × kalender): porsi sama untuk kedua target, tidak ada tanggal z > 3 → terbantah.
- Ramp nominal order relatif rata-rata jangka panjang customer (3/7/30 hari, maks/rata-rata, jumlah item): |z| ≤ 2,05 → sudah tertangkap (model order ecm + GRU).
- Rekening × pinjaman (s26): lihat di atas; H21 (inflow non-gaji/income) satu-satunya sisa lemah.
- Disiapkan `s27_residual_capacity.py`: residual boosting di atas OOF E200 dengan semua kolom dan pohon lebih dalam (depth 3/6/8) untuk menjawab "masih adakah interaksi/kapasitas yang belum dipakai?" — dijalankan setelah rantai job (RAM).

**Catatan desain E201/H02 (`auxpd.py`)** — anti-bocor-lintas-pinjaman: event pinjaman lain di jendela 12 bulan pinjaman P dikecualikan; hasil pinjaman lain hanya yang jendelanya tidak tumpang tindih; pinjaman yang berjalan saat keputusan P hanya dilihat tunggakannya sebelum t; laporan biro as-of keputusan P; sifat e-commerce/rekening di luar zona ramp (90 hari / 3 bulan terakhir). Label = DPD ≥ 90 dalam 12 bulan kalender sejak pencairan. Target sekarang tidak dibaca. Tag: **CPU-deterministik**.

**E202 = H05 (10-fold bersarang) hasil CV** (rantai job dimulai ulang 11.44 setelah terputus `^C`; seed yang sudah ada dilewati): `e121.py --cv --use <7 npz> --basefs all --extra rp_events_v1.parquet --subfolds 2 --name E202` — tiap fold asli dibelah 2 (StratifiedKFold seed 42), model dilatih pada 4 fold + separuh fold validasi (≈90% data), baris fold k memakai fitur nested va{k}, test = rata-rata 10 model × 3 seed. **Bag 0,84710**, KS 0,5306 vs E200 0,84659 / 0,5296 → Δ **+0,00051**, CI 95% [+0,00022; +0,00079], P 0,999, naik 4/5 (per fold +0,00073 / +0,00099 / +0,00041 / −0,00006 / +0,00050) → **DITERIMA** (`cmp_E202.json`). Sesuai prediksi kurva belajar (+0,0005). Pemakaian: skema latih model akhir; eksperimen fitur tetap dibandingkan 5-fold terhadap E200 (2× lebih cepat). Biaya: ±2× waktu latih. Tag: **CPU-deterministik**.

**E203 = H12 pendahuluan (sensitivitas parameter, set E200, lr 0,05, 1 seed, 5 fold; `experiments/e203_sens.csv`)**:

| konfigurasi | mean-fold AUC | iterasi |
|---|---|---|
| tuned (Fase 5) | **0,84594** | 995 |
| feature_fraction 0,30 | 0,84578 | 602 |
| depth 5 | 0,84557 | 815 |
| feature_fraction 0,20 | 0,84552 | 1010 |
| feature_fraction 0,08 | 0,84543 | 1117 |
| depth 6, 63 daun | 0,84537 | 674 |
| min_child 100 | 0,84526 | 821 |
| tanpa extra_trees | 0,84457 | 403 |

Parameter tuned tetap optimal walau dipilih di set 176 kolom; semua variasi turun (kecuali tanpa extra_trees, selisihnya di dalam noise 1 seed). Kesimpulan: **H12 (Optuna 5 fold ulang) tidak diprioritaskan** — ruang perbaikannya kecil dibanding biaya >3 jam. Kapasitas pohon (depth 5–6) juga tidak membantu → menguatkan dugaan jenuh representasi (dikonfirmasi `s27`).

**Probe murah putaran 2 (`s30_misc_probes.py`, residual E202)**: fase aplikasi dalam siklus jatuh tempo pinjaman aktif (hari ke due berikutnya, due dalam 3/7 hari, due di antara aplikasi dan keputusan) |z| ≤ 0,7 — target memang 16,9% vs 14,0% antar kuintil hari ke due, tetapi residual datar (sudah tertangkap); inkonsistensi profil (masa kerja > umur kerja: 0 kasus; rasio masa kerja/umur kerja, income per tanggungan) |z| ≤ 1,5 → **terbantah semua**. Ramp e-commerce berjangkar waktu aplikasi (bukan keputusan) sudah diuji di s11 (`ec_last_vs_app_h`, `ec_n_after_app`, |z| < 2).

**Rantai berikutnya** (`experiments/run_s27_s29.cmd`, otomatis setelah E204): E205 = set E200 **versi CPU saja** (528 + ecm + rp_ev, tanpa fitur GRU dari GPU; acuan kandidat A PROMPT 3 dan uji tumpang tindih rp_ev ↔ seqx_rp) → `s27` uji kapasitas residual → `s29` kurva belajar set E200 (syarat refit PROMPT 3 Langkah 8). `e141.py` diberi opsi `--extra` untuk DART di set E200.

**E204 = H21 (inflow non-gaji / income)**: bag 0,84677 vs E200 0,84659 → Δ +0,00019, CI 95% [−0,00011; +0,00050], P 0,884, naik 4/5; KS 0,5296 → 0,5293 → **KANDIDAT** (`cmp_E204.json`). Audit Lapis 2 bersih (ERROR 0, AUDIT 0, KS train-test ≤ 0,009, AUC univariat ≤ 0,52); Lapis 1: bulan rekening selalu berakhir sebelum bulan keputusan, tanpa target. Builder masih skrip eksplorasi → dipindah ke `src/features/` bila dipakai PROMPT 3. Tag: **CPU-deterministik**.

**E205 = set E200 versi CPU saja** (528 + ecm + rp_ev; `e121.py --cv --use e121_ordermodel_v1.npz --basefs all --extra rp_events_v1.parquet --name E205`): seed 0,84494 / 0,84470 / 0,84503, **bag 0,84517**, KS OOF 0,5230 (mean-fold 0,5263). Bootstrap: vs E130a **+0,00203** (CI [+0,00150; +0,00261], P 1,000 → diterima); vs E122x1 +0,00022 (P 0,75; Spearman 0,9891); E205 → E200 (+ fitur GPU) +0,00141 (P 1,000). Artinya: (1) **kandidat A (CPU-deterministik) kini setara model GPU terbaik lama** dan bisa direproduksi identik di notebook; (2) fitur GPU masih bernilai +0,0014 di atasnya; (3) rp_ev dan seqx_rp sebagian tumpang tindih.

**s27 (kapasitas residual)**: residual boosting di atas logit OOF E200 dengan semua kolom (528 + rp_ev + fitur nested versi OOF), lr 0,03, depth 3/7 daun, depth 6/63 daun, depth 8/255 daun → **berhenti di iterasi 1 di semua konfigurasi**, Δ −0,00002 / −0,00005 / −0,00006. Tidak ada struktur interaksi yang bisa dipelajari di atas model sekarang → model jenuh terhadap representasi; kenaikan berikutnya hanya dari **informasi baru**, **bias induktif yang lebih hemat data**, atau **lebih banyak data/variansi lebih kecil**.

**s29 (kurva belajar set E200, lr 0,05, 1 seed)**: 14k 0,84058 → 28k 0,84293 → 42k 0,84520 → 56k 0,84594 (s18 set E130a: 0,83507 / 0,83924 / 0,84137 / 0,84211). Δ 50% → 100% = +0,0030 (sd antar fold 0,0013) → **syarat refit PROMPT 3 Langkah 8 terpenuhi**. Asimtot dengan eksponen tetap 0,3 / 0,589 / 1,0: 0,8565 / 0,8500 / 0,8473 (E130a: 0,8564 / 0,8479 / 0,8443); proyeksi 70k (refit) 0,8459–0,8466, 140k 0,8466–0,8484 (versi bag lr kecil ±+0,0007). Selisih set E130a → E200 (fitur urutan GPU + rantai bayar) +0,0055 di 14k dan +0,0037–0,0038 di 28k–56k: set E200 dengan 28k baris (0,84293) sudah melampaui set E130a dengan 56k (0,84211) → representasi yang lebih baik setara > 2× data.

**Implikasi (18 Okt sore)**: target 0,857–0,860 hanya tercapai pada asimtot optimistis (data tak hingga). Sumber kenaikan yang masih terbuka: (a) fitur yang "menghitung duluan" apa yang dipelajari model secara lambat dari data (seperti rp_ev = setara ±2× data); (b) refit 100% (+0,0003), seed/ensemble GBDT+DART (+0,0003–0,001), H21 (+0,0002); (c) informasi baru — belum ada bukti tersisa setelah ±40 uji sinyal. Uji berikutnya: `s31` (H07 stacking per sumber = bias induktif aditif), `s32` (H14 mekanisme tier), lalu E207 (DART di set CPU E205).
