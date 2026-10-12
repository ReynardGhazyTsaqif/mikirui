# Planning Model DSC LOGIKA UI 2026 — Prediksi Default Kredit

Oct 10, 2026 · @Rey

## Ringkasan eksekutif

Skor terbaik akan datang dari feature engineering relasional yang teliti, divalidasi dengan 5-fold CV, digabung lewat ensemble ringan, lalu dikunci dalam satu notebook deterministik.

- **Sumber kenaikan utama:** fitur agregat dari 6 tabel pendukung, semuanya dihitung relatif terhadap `decision_datetime`. Alokasikan sekitar separuh waktu di sini.
- **Model:** LightGBM sebagai model utama, CatBoost dan XGBoost sebagai pendamping. Logistic Regression berbasis binning dan MLP hanya untuk diversitas ensemble.
- **Validasi:** StratifiedKFold 5-fold dengan fold tetap untuk semua model. Keputusan diambil dari OOF AUC, bukan public LB yang noise-nya sekitar ±0,006–0,008.
- **Fine-tuning** di sini berarti tuning hyperparameter dengan Optuna, retrain dengan learning rate kecil, dan rata-rata 3–5 seed. Model pretrained tidak dipakai karena aturan melarang data luar.
- **Ensemble:** rata-rata rank berbobot jika metriknya AUC, atau rata-rata logit plus kalibrasi jika LogLoss. Bobot dicari pada OOF.
- **Reproduksibilitas adalah syarat lolos**, bukan hanya 10% nilai. Notebook final wajib menghasilkan file yang sama persis dengan submission terpilih, dibuktikan dengan hash MD5.
- **Tenggat internal:** feature freeze Oct 26, 2026, notebook final selesai Oct 29, 2026, upload notebook Oct 30, 2026 paling lambat 12.00 WIB.

Konfigurasi final yang paling mungkin menang: blend LightGBM (dua set fitur, 3–5 seed) + CatBoost + XGBoost, dengan bobot kecil untuk LR/MLP. Ini hipotesis awal; OOF yang memutuskan.

&#91;embedded content: alur solusi · data, fitur, model, blend, submission\]

Fitur dihitung per grup dengan cutoff yang sama, semua model menyimpan OOF pada fold yang sama, dan hanya file dengan MD5 yang cocok yang dipilih di Kaggle.

## Batasan lomba yang menentukan strategi

Tiga aturan paling mengubah cara kerja: skor 90% dari Kaggle dengan public dan private sama berat, output notebook wajib sama dengan submission terpilih, dan larangan data luar.

| Aturan (guidebook / CLAUDE.md) | Implikasi ke strategi |
| --- | --- |
| Public 45% + private 45% + notebook 10% | Kejar generalisasi. Jangan memilih submission hanya karena public-nya tinggi. |
| Lolos 8 besar hanya jika output notebook = submission terpilih | Pipeline deterministik, satu sumber kebenaran, cek MD5. Gagal berarti posisi diganti tim berikutnya. |
| Notebook telat: −1 poin per menit | Upload jauh sebelum 30 Okt 23.59 WIB. |
| Wajib Python; hanya data panitia | Tanpa data eksternal dan tanpa model pretrained (LLM, TabPFN, embedding siap pakai). |
| Notebook dinilai reproduksibilitas dan kelancaran kode | Run All dari nol tanpa error, runtime wajar, markdown penjelasan di setiap bagian. |
| Split train/test acak; 1 customer = 1 aplikasi | StratifiedKFold sah. GroupKFold dan split waktu tidak perlu. |
| Urutan `sample_submission` berbeda dari test | Selalu merge berdasarkan `application_id`. |
| Metrik tidak tertulis di guidebook | Asumsikan ROC-AUC, siapkan cabang LogLoss (bagian "Cabang jika metrik bukan AUC"). |

## Fase 0: cek dan siapkan sebelum eksperimen pertama

Pastikan metrik dan aturan pemilihan submission jelas dulu; dua jawaban ini mengubah cara menggabungkan dan memilih model.

### Tanyakan ke halaman Kaggle atau grup WhatsApp panitia

- [ ] Metrik evaluasi persisnya apa: ROC-AUC, Gini, LogLoss, atau Brier?
- [ ] Berapa persen data test yang masuk public LB?
- [ ] Berapa batas submission per hari?
- [ ] Berapa submission final yang boleh dipilih? Jika dua, mana yang dicocokkan dengan notebook, dan public score mana yang dipakai?
- [ ] Jam tutup kompetisi di Kaggle (Kaggle umumnya memakai UTC) dibanding tenggat notebook 23.59 WIB.
- [ ] Apakah panitia menjalankan ulang notebook? Di mana, dan adakah batas waktu run?
- [ ] Apakah model pretrained seperti TabPFN dianggap melanggar? Default kita: tidak dipakai.

Rencana di CLAUDE.md untuk memilih 2 submission akhir perlu dikonfirmasi dulu. Guidebook mensyaratkan output notebook sama dengan "submisi yang dipilih", sehingga dua pilihan bisa menimbulkan ambiguitas.

### Setup teknis

- [ ] Nama tim di Kaggle sama persis dengan nama registrasi; semua anggota sudah bergabung ke tim.
- [ ] Repo git privat bersama dengan struktur folder seperti di Lampiran.
- [ ] `requirements.txt` dengan versi terkunci: pandas, scikit-learn, lightgbm, xgboost, catboost, optuna, pyarrow, shap.
- [ ] `src/config.py` berisi `SEED = 42`, `N_FOLDS = 5`, dan `DATA_DIR` yang mudah diganti.
- [ ] `folds/folds.csv` berisi `application_id → fold` dari `StratifiedKFold(5, shuffle=True, random_state=42)`, dibuat sekali dan dipakai semua anggota.
- [ ] Loader hemat memori untuk `ecommerce_transactions.csv` dengan cache parquet di `features/`.

### Trik deteksi metrik (E00)

Submit prediksi konstan 0,1358 untuk semua baris. Skor 0,5 berarti AUC (0 berarti Gini), sekitar 0,397 berarti LogLoss, dan sekitar 0,117 berarti Brier. Angka LogLoss dan Brier sedikit bergeser jika rasio target di test bukan tepat 13,58%.

## Protokol eksperimen

Setiap eksperimen mengubah satu hal, dibandingkan dengan model terbaik saat ini pada fold yang sama, dan tercatat otomatis.

1. **Fold tetap.** Semua model dan semua anggota memakai `folds.csv` yang sama. Ini syarat agar OOF bisa di-stack dan perbandingan bersifat berpasangan.
2. **Dua konfigurasi LightGBM.** "Dev" (learning rate 0,05) untuk menguji fitur dengan cepat; "final" (0,01–0,02) hanya untuk kandidat akhir. Parameter dev ada di Lampiran.
3. **Simpan setiap run:** `oof/<exp_id>.npy`, `preds/<exp_id>.npy` (rata-rata prediksi test dari 5 fold), dan satu baris di `experiments/log.csv`. Tanpa ini, ensemble di akhir tidak bisa dibangun.
4. **Aturan terima:** grup fitur diterima jika OOF AUC naik minimal 0,0005 **dan** membaik di minimal 4 dari 5 fold. Jika ragu, ulangi dengan 3 seed dan bandingkan rata-ratanya.
5. **Tambah per grup, bukan per fitur.** Di akhir fase FE, jalankan ablation (hapus satu grup tiap run) untuk memastikan setiap grup masih berguna.
6. **Penamaan:** prefix per sumber, yaitu `app_`, `prof_`, `pa_`, `rp_`, `bu_`, `bp_`, `ab_`, `ec_`, `x_`. Daftar kolom selalu diurutkan dengan `sorted()`.
7. **Train dan test diproses bersama** oleh fungsi fitur yang sama. Ini mencegah bug fitur test yang dihitung berbeda.
8. **Cutoff waktu:** semua fitur tabel pendukung memakai `cutoff = decision_datetime` aplikasi saat ini, lewat peta `customer_id → cutoff`.

## Strategi validasi

StratifiedKFold 5-fold adalah hakim utama: dengan 70.000 baris, OOF AUC jauh lebih stabil daripada public LB.

- **Skema:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Split acak dan tidak ada overlap customer, jadi skema ini sah.
- **Noise OOF:** dengan AUC sekitar 0,80 dan ±9.500 positif, galat baku OOF AUC sekitar 0,003. Perbandingan dua model pada fold yang sama jauh lebih presisi karena galatnya berkorelasi.
- **Noise public LB:** jika public berisi 30–50% test (9.000–15.000 baris), galat bakunya sekitar 0,006–0,008. Selisih public di bawah ±0,005 praktis tidak bermakna.
- **Korelasi CV–LB:** catat pasangan (OOF, public) setiap submission. Jika OOF naik tetapi public anjlok jauh, curigai bug pipeline test lebih dulu.
- **Adversarial validation (E04):** latih model untuk membedakan train dan test. AUC sekitar 0,5 berarti aman; di atas 0,55 berarti ada fitur yang dihitung berbeda.
- **Konfirmasi final:** kandidat akhir dinilai ulang dengan 3 × 5-fold (seed fold berbeda) sebelum dipilih.
- **Analisis segmen OOF:** AUC untuk customer dengan dan tanpa riwayat, per `employment_type`, dan per jumlah provider biro. Segmen terlemah menjadi target FE berikutnya.

## Rencana feature engineering per tabel

Urutan pengerjaan mengikuti kekuatan sinyal dari EDA: riwayat pembayaran dan biro dulu, lalu aplikasi sebelumnya, rasio aplikasi, perilaku rekening, e-commerce, dan fitur lintas tabel. Label prioritas: P1 wajib, P2 penting, P3 jika sempat.

### Riwayat pembayaran: `repayment_history` (`rp_`, P1)

Ini sinyal terkuat (max DPD historis AUC 0,744). Mulai dengan memastikan semantik kolom: apakah `days_past_due` adalah DPD saat cutoff dan `max_days_past_due` adalah DPD terburuk cicilan itu.

- **Turunan per cicilan:** `paid_ratio = amount_paid / amount_due`, tunggakan `amount_due − amount_paid`, `pay_delay_days = last_payment_datetime − due_date` (negatif berarti bayar lebih awal), flag DPD ≥ 1/30/60/90, umur cicilan `cutoff − due_date`.
- **Agregat per customer,** all-time dan jendela 90/180/365 hari (berdasarkan `due_date`): max/mean/std DPD, jumlah dan proporsi cicilan DPD ≥ 1/30/60/90, Σpaid/Σdue, total tunggakan, mean/max/std keterlambatan, proporsi bayar lebih awal.
- **Kondisi saat cutoff:** DPD terbesar saat ini, total tunggakan berjalan, jumlah cicilan yang masih menunggak.
- **Recency:** hari sejak DPD ≥ 30 terakhir dan sejak pembayaran terakhir.
- **Dua tingkat** (cicilan → pinjaman → customer): per pinjaman hitung max DPD, paid ratio, dan flag `ever90`. Lalu agregasikan: jumlah pinjaman `ever90` (analog langsung target), statistik per pinjaman, dan fitur pinjaman paling baru.
- **Tren (P2):** max DPD 6 bulan terakhir dikurangi periode sebelumnya, DPD berbobot waktu `Σ dpd · exp(−umur/180)`, streak terlambat terpanjang (P3).

### Biro kredit: `bureau_observations` (`bu_` saat ini, `bp_` sebelumnya, P1)

Pisahkan laporan untuk aplikasi saat ini dan untuk previous applications sebelum agregasi.

- **Saat ini, pivot per provider (A/B/C):** `score`, `active_accounts`, `outstanding_balance`, `utilization`, `inquiries_last_6m`, `delinquent_accounts_last_12m`, dan umur laporan `cutoff − report_datetime`.
- **Lintas provider:** mean/min/max/std skor, jumlah provider, jumlah laporan unscored, max inquiries, max delinquent.
- **Utang:** simpan **max dan sum** `outstanding_balance`. Akun bisa tercakup di lebih dari satu biro, jadi sum bisa menghitung ganda; biarkan model memilih.
- **Normalisasi skor per provider:** z-score atau persentil per provider (dihitung di train+test, tanpa target) sebelum dirata-rata.
- **Utilisasi ulang:** Σ revolving balance / Σ revolving limit, plus flag tanpa limit revolving (utilisasi kosong di 59% baris).
- **Sebelumnya (P2):** skor terakhir per provider, delta skor saat ini dikurangi sebelumnya (provider sama), kemiringan skor terhadap waktu, pertumbuhan utang dan inquiries, panjang riwayat biro.

### Aplikasi sebelumnya: `previous_applications` (`pa_`, P1)

- **Hitungan:** jumlah aplikasi, approved, rejected, cancelled, beserta rasionya; juga dalam jendela 90/180/365 hari.
- **Recency dan kecepatan:** hari sejak aplikasi, approval, dan penolakan terakhir; panjang riwayat; jumlah aplikasi 30/90 hari terakhir.
- **Nominal:** statistik `requested_amount`, rasio `approved_amount / requested_amount`, `loan_amount` saat ini dibanding approved terbesar dan terakhir.
- **Bunga:** mean/max bunga sebelumnya, serta bunga saat ini dikurangi bunga pinjaman terakhir (re-pricing oleh lender).
- **Beban cicilan aktif:** pinjaman dengan `decision_datetime + term_months > cutoff` dianggap masih berjalan; jumlahkan `scheduled_monthly_installment`-nya. Verifikasi asumsi ini dengan `due_date` pertama di repayment.

### Aplikasi dan profil (`app_`, `prof_`, P1)

- **Mentah:** semua kolom kecuali `*_id`. `term_months` konstan 12, buang setelah dicek.
- **Keterjangkauan:** `installment / income` (PTI), `loan_amount / income`, `income / (dependents + 1)`.
- **Stabilitas:** `employment_tenure_months / (age × 12)` dan usia mulai bekerja.
- **Cek formula anuitas:** cicilan teoretis `P · r / (1 − (1 + r)^−12)` dengan `r = annual_interest_rate / 12`; hitung selisih dan rasionya terhadap `proposed_monthly_installment`. Penyimpangan bisa jadi sinyal tersembunyi di data sintetis.
- **Waktu:** lama proses `decision_datetime − application_datetime`, jam, hari, dan bulan aplikasi.
- **Residual bunga (P2):** regresi tanpa target yang memprediksi `annual_interest_rate` dari skor biro dan fitur aplikasi. Residualnya mewakili penilaian risiko "privat" lender.
- **Kategorikal:** native categorical di LightGBM/CatBoost. Target encoding tidak perlu karena kardinalitasnya kecil.
- **Pendapatan relatif (P3):** income dibagi median income per `employment_type` atau `region`.

### Perilaku rekening: `account_behavior` (`ab_`, P2)

Agregat mentahnya lemah, jadi fokus pada tren, volatilitas, dan rasio.

- **Cakupan:** jumlah bulan, jarak bulan terakhir ke cutoff, flag punya data.
- **Jendela 3/6/12 bulan:** mean/min/std saldo, Σ inflows dan outflows, mean `income_like_credits`, Σ dan proporsi `low_balance_days`, jumlah bulan dengan arus bersih negatif.
- **Tren dan volatilitas:** kemiringan saldo dan income (dinormalisasi rata-rata), rasio 3 bulan terakhir terhadap sebelumnya, koefisien variasi income.
- **Verifikasi pendapatan (P1):** `mean income_like_credits / declared_monthly_income` dan `installment / mean income_like_credits`.
- **Bantalan likuiditas:** `average_balance / installment` dan `outflows / inflows`.

### E-commerce: `ecommerce_transactions` (`ec_`, P2)

Baca dengan `usecols` dan dtype hemat, dedup per `order_id` untuk metrik order, dan keluarkan order `cancelled` dari spending.

- **RFM order completed:** jumlah order, total belanja, nilai order rata-rata/median/max, belanja 30/90/180/365 hari, hari sejak order terakhir, jarak antarorder.
- **Pembatalan:** jumlah dan rasio cancel, keseluruhan dan 90 hari terakhir.
- **Metode bayar:** proporsi COD, e\_wallet, bank\_transfer, debit\_card, credit\_card (jumlah dan nilai), serta pergeseran terbaru ke COD atau kartu kredit.
- **Kategori:** proporsi nilai per kategori, plus lonjakan electronics dan health menjelang aplikasi.
- **Pola waktu (P3):** proporsi order malam (00–05) dan akhir pekan.
- **Rasio:** belanja bulanan dibagi income deklarasi dan dibagi `income_like_credits`.

### Lintas tabel (`x_`, P1–P2)

- **Total PTI:** (cicilan saat ini + Σ cicilan aktif) / income.
- **Utang biro / pendapatan tahunan**, versi max dan sum.
- **Tiga sumber pendapatan:** deklarasi vs mutasi rekening vs belanja e-commerce.
- **Skor stres gabungan:** jumlah sumber yang menunjukkan tanda bahaya (DPD ≥ 30 baru-baru ini, low balance tinggi, cancel tinggi, biro delinquent > 0).
- **Interaksi:** bunga × PTI.

### Missingness

- 30% customer tanpa riwayat punya target rate 11,7%, jadi missing itu informatif. Untuk GBDT, biarkan NaN dan tambahkan flag `has_prev`, `has_repay`, `has_account`, `has_bureau_prev`.
- Untuk LR/MLP: imputasi median yang di-fit di dalam fold, plus kolom indikator.

## Seleksi fitur

Seleksi dikerjakan setelah FE selesai (21 Okt) dan diperlakukan sebagai eksperimen biasa: dipakai hanya jika OOF tidak turun. Untuk GBDT, fitur noise biasanya hanya sedikit merugikan; manfaat utamanya adalah kecepatan dan stabilitas.

1. Buang kolom konstan, duplikat identik, NaN di atas 99%, dan salah satu dari pasangan berkorelasi |ρ| > 0,995.
2. **Null importance:** bandingkan gain LightGBM pada target asli dengan 30–50 run bertarget acak. Buang fitur yang importance aslinya tidak melampaui persentil 75 distribusi null.
3. Verifikasi dengan CV. Jika OOF turun lebih dari 0,0003, kembalikan fitur yang dibuang secara bertahap.
4. Simpan dua set fitur: "full" dan "selected". Keduanya kandidat, dan perbedaannya menambah diversitas ensemble.
5. Opsional: eliminasi bertahap 10% fitur terbawah menurut SHAP. Boruta dan RFE penuh terlalu lambat untuk ratusan fitur.

## Model yang dicoba dan yang sengaja tidak

Tiga GBDT menjadi tulang punggung. LR dan MLP hanya masuk jika menaikkan OOF blend, bukan karena skor tunggalnya.

| Model | Peran | Prioritas | Catatan eksekusi |
| --- | --- | --- | --- |
| LightGBM | Model utama dan penguji fitur | P1 | Native categorical, `deterministic=True`; lr dev 0,05, final 0,01–0,02 |
| CatBoost | GBDT kedua, kuat untuk kategorikal | P1 | CPU, depth 6–8, early stopping 200; lambat, jalankan malam hari |
| XGBoost (`hist`) | Diversitas GBDT | P2 | max\_depth 4–8, eta 0,02; cek determinisme dengan dua kali run |
| Logistic Regression + binning | Diversitas linear, bahan interpretasi presentasi | P2 | Pipeline per fold: imputasi, KBinsDiscretizer 10–20 kuantil, one-hot, L2 |
| MLP | Diversitas non-pohon | P3 | QuantileTransformer per fold, 2–3 layer + dropout, CPU agar deterministik, 3–5 seed |
| Varian LightGBM (DART, extra\_trees, set fitur lain) | Diversitas murah | P3 | DART tidak mendukung early stopping, jadi lambat |
| Random Forest / ExtraTrees | Diversitas murah | P3 | NaN diisi nilai sentinel, `min_samples_leaf` 20–50 |
| AutoGluon / FLAML | Pembanding saja | P3 | Jangan masuk notebook final: `time_limit` membuat hasil tidak deterministik |

**Sengaja tidak dicoba:**

- **SVM, KNN, Naive Bayes:** lambat atau lemah untuk 70.000 baris dengan ratusan fitur campuran.
- **HistGradientBoosting:** algoritmanya mirip LightGBM, jadi diversitasnya kecil.
- **TabPFN dan model pretrained lain:** berisiko melanggar larangan data luar.
- **LSTM/GRU/Transformer pada urutan mentah:** biaya tinggi, kenaikan kecil dibanding fitur tren buatan tangan. Hanya jika semua fase selesai lebih cepat (E111).
- **Model survival (Cox, DeepHit):** tidak ada label waktu-ke-default, hanya target biner.
- **GNN:** tidak ada relasi antar customer.
- **SMOTE, ADASYN, oversampling:** tidak menaikkan AUC pada GBDT dan merusak kalibrasi probabilitas.

## Daftar eksperimen berurutan

Eksperimen berjalan per fase dan tiap baris punya hipotesis serta aturan keputusan, sehingga baseline terbaik terbentuk bertahap dari B0 ke B2. Ubah kolom Status saat eksperimen berjalan; "aturan terima" mengacu ke Protokol eksperimen.

### Fase 1: fondasi dan baseline (11–12 Okt)

| ID | Eksperimen | Hipotesis / tujuan | Keputusan | Status |
| --- | --- | --- | --- | --- |
| E00 | Submit konstanta 0,1358 | Validasi format dan deteksi metrik | Sekali saja | Selesai: LB 0 → metrik KS |
| E01 | LightGBM dev, kolom mentah aplikasi + profil | Baseline B0; uji pipeline CV dan log | Menjadi acuan awal | Diterima 0,6517 |
| E02 | + rasio aplikasi/profil (PTI, anuitas, lama proses, waktu) | Keterjangkauan menambah sinyal | Aturan terima | Ditolak |
| E03 | + agregat naif semua tabel (count/mean/max/min/std; biro saat ini di-pivot) | Baseline B1 "semua tabel" | Submit serius pertama | Diterima 0,8256 (B1), LB 0,524 |
| E04 | Adversarial validation train vs test | Deteksi bug pipeline test | AUC ≈ 0,5 berarti lanjut | Selesai: AUC 0,503 (aman) |

### Fase 2: feature engineering (13–20 Okt)

| ID | Eksperimen | Hipotesis / tujuan | Keputusan | Status |
| --- | --- | --- | --- | --- |
| E10 | `rp_` DPD dan pembayaran all-time | Merapikan sinyal terkuat | Aturan terima | Diterima +0,0010 |
| E11 | `rp_` jendela 90/180/365 hari + recency | Perilaku terbaru lebih prediktif | Aturan terima | Ditolak |
| E12 | `rp_` dua tingkat, `ever90`, pinjaman terakhir | Default sebelumnya adalah analog target | Aturan terima | Ditolak |
| E13 | `rp_` tren, DPD berbobot waktu, streak | Arah perubahan menambah informasi | Aturan terima | Ditolak (+0,0001) |
| E20 | `bu_` pivot provider, agregat, normalisasi skor | Skor antar biro tidak sebanding | Aturan terima | Ditolak |
| E21 | `bp_` delta dan tren skor, pertumbuhan utang | Skor yang memburuk menandai risiko | Aturan terima | Ditolak |
| E30 | `pa_` hitungan, rasio, recency, kecepatan | Penolakan dan lapar kredit | Aturan terima | Ditolak |
| E31 | `pa_` cicilan aktif, total PTI, nominal dan bunga | Beban utang total | Aturan terima | Diterima +0,0014 (B2) |
| E40 | `ab_` jendela, tren, volatilitas | Tren lebih kuat dari agregat mentah | Aturan terima | Ditolak |
| E41 | `ab_` verifikasi pendapatan, bantalan likuiditas | Income deklarasi bisa dilebihkan | Aturan terima | Ditolak |
| E50 | `ec_` RFM, jendela, pembatalan | Belanja mencerminkan kondisi keuangan | Aturan terima | Diterima +0,0039 |
| E51 | `ec_` kategori, metode bayar, pola waktu | Pergeseran ke COD/kartu kredit dan lonjakan belanja | Aturan terima | Ditolak (−0,0021) |
| E60 | `x_` total PTI, utang/pendapatan, tiga sumber income, skor stres | Sinyal lintas tabel saling menguatkan | Aturan terima | Ditolak |
| E61 | Residual bunga | Lender punya informasi risiko privat | Aturan terima | Ditolak (netral) |
| E62 | Ablation per grup (hapus satu grup tiap run) | Setiap grup masih berguna | Grup tak berguna dibuang; hasilnya B2 | Selesai: semua grup berguna, B2 = E31 |

### Fase 3: seleksi dan variasi pelatihan (21–22 Okt)

| ID | Eksperimen | Hipotesis / tujuan | Keputusan | Status |
| --- | --- | --- | --- | --- |
| E70 | Buang konstan, duplikat, korelasi tinggi | Lebih cepat tanpa rugi | Terima jika OOF tidak turun | Selesai: 312 kol, +0,0005 (3 seed) |
| E71 | Null importance, set "selected" | Kurangi noise | Simpan set full dan selected | Selesai: B3 = sel_full 176 kol (E71b), KS 3 seed 0,5066; set kedua E71c 117 kol |
| E72 | `scale_pos_weight` / `is_unbalance` | Biasanya tidak membantu AUC | Lewati jika metrik LogLoss | Ditolak (is_unbalance & spw 2,5 netral/negatif) |
| E73 | Monotone constraints pada skor biro dan DPD | Lebih robust | Terima jika OOF turun < 0,0003 | Ditolak setelah uji 3 seed (AUC sama, KS turun) |
| E74 | Native categorical vs one-hot vs target encoding dalam fold | Cara terbaik untuk 4 kolom kategori | Pilih yang terbaik | Native tetap (hanya 1 kolom kategori tersisa; semua varian netral) |
| E75 | Analisis segmen dan error OOF | Temukan segmen lemah | Ide FE tambahan, paling lambat 25 Okt | Selesai: segmen terlemah = tanpa riwayat (KS 0,466) & belanja <=2 hari (KS 0,428) |

### Fase 4: diversitas model (21–23 Okt, paralel dengan Fase 3)

| ID | Eksperimen | Hipotesis / tujuan | Keputusan | Status |
| --- | --- | --- | --- | --- |
| E80 | CatBoost pada B2 | GBDT kedua terkuat | Kandidat jika menaikkan blend | Kandidat: AUC 0,8346, KS 0,5092 (model tunggal terbaik); blend +0,0031 |
| E81 | XGBoost pada B2 | Diversitas GBDT | Kandidat jika menaikkan blend | Kandidat: AUC 0,8331; blend +0,0017 |
| E82 | LR + binning | Diversitas linear | Kandidat jika menaikkan blend | Kandidat: AUC 0,8231; blend bobot 0,25 +0,0015 |
| E83 | MLP | Diversitas non-pohon | Kandidat jika menaikkan blend | Kandidat (sklearn MLP): AUC 0,8218; blend bobot 0,25 +0,0013 |
| E84 | RF / ExtraTrees (opsional) | Diversitas murah | Kandidat jika menaikkan blend | Kandidat lemah: AUC 0,8190; blend bobot 0,15 +0,0004 |
| E85 | AutoGluon (opsional) | Batas atas kasar | Pembanding saja | Ditunda: autogluon/torch tidak terpasang, instalasi berat di RAM 7,7 GB |

### Fase 5: tuning (23–25 Okt)

| ID | Eksperimen | Hipotesis / tujuan | Keputusan | Status |
| --- | --- | --- | --- | --- |
| E90 | Optuna LightGBM | Kenaikan kecil tapi nyata | Retrain dengan lr kecil | Selesai 12 Okt: 100 trial, bag 3 seed 0,83788 / KS 0,5138, public 0,54685 (terbaik) |
| E91 | Optuna CatBoost | Kenaikan kecil tapi nyata | Retrain dengan lr kecil | Selesai 12 Okt: 40 trial, bag 3 seed 0,83666 (iterasi mentok 5000) |
| E92 | Optuna XGBoost | Kenaikan kecil tapi nyata | Retrain dengan lr kecil | Selesai 12 Okt: 41 trial, bag 3 seed 0,83700 |
| E93 | Rata-rata 3–5 seed | Kurangi varians | Selalu dipakai di final | Dipakai (3 seed) di semua model Fase 5 |

### Fase 6: ensemble (26–27 Okt)

| ID | Eksperimen | Hipotesis / tujuan | Keputusan | Status |
| --- | --- | --- | --- | --- |
| E100 | Rata-rata rank sederhana | Baseline ensemble | Acuan | Belum |
| E101 | Rank berbobot, bobot dioptimasi di OOF | Bobot optimal | Pakai jika ≥ E100 + 0,0003 di nested CV | Belum |
| E102 | Stacking LR pada logit OOF | Penggabungan lebih fleksibel | Pakai jika ≥ E101 + 0,0005 | Belum |
| E103 | Hill climbing (Caruana) | Seleksi anggota otomatis | Bandingkan dengan E101 | Belum |
| E104 | Kalibrasi Platt / isotonic | Probabilitas terkalibrasi | Wajib jika metrik LogLoss/Brier | Belum |

### Fase 7: opsional jika waktu tersisa

| ID | Eksperimen | Hipotesis / tujuan | Keputusan | Status |
| --- | --- | --- | --- | --- |
| E110 | Pseudo-labeling test, protokol per fold | Data tambahan | Hanya jika OOF jujur naik | Belum |
| E111 | GRU pada urutan cicilan dan rekening | Pola urutan | Anggota ensemble tambahan | Belum |
| E112 | Model per segmen (punya/tanpa riwayat) | Spesialisasi | Uji untuk diversitas; biasanya kalah dari model global | Belum |

Untuk E110, label semu yang dipakai saat melatih fold k harus berasal dari model yang tidak melihat fold k. Tanpa itu, OOF menjadi terlalu optimis.

## Hyperparameter tuning ("fine-tuning")

Tuning dijalankan 23–25 Okt setelah set fitur hampir final, karena kenaikannya biasanya hanya beberapa per seribu AUC, jauh di bawah kenaikan dari FE. Inilah arti fine-tuning dalam rencana ini.

1. Optuna `TPESampler(seed=42)` dalam satu proses. Objektifnya mean AUC dengan fold tetap. Untuk menghemat waktu, cari dengan 3-fold lalu konfirmasi 5 trial terbaik dengan 5-fold.
2. Selama pencarian, learning rate dikunci 0,05 dengan early stopping 200. Parameter terbaik kemudian di-retrain dengan lr 0,01–0,02.
3. Jalankan semalaman: LightGBM 100–200 trial, XGBoost 50–100, CatBoost 30–60.
4. Model final = rata-rata prediksi 5 fold × 3–5 seed.
5. Parameter terbaik di-hardcode di notebook final. Kode tuning tetap disertakan dengan `RUN_TUNING = False`.

| Model | Ruang pencarian |
| --- | --- |
| LightGBM | `num_leaves` 15–255 (log), `max_depth` −1 atau 4–12, `min_child_samples` 20–500 (log), `feature_fraction` 0,1–0,8, `bagging_fraction` 0,5–1,0 (freq 1), `lambda_l1` 1e-3–10 (log), `lambda_l2` 1e-3–100 (log), `min_gain_to_split` 0–1, `max_bin` {63, 127, 255} |
| CatBoost | `depth` 4–9, `l2_leaf_reg` 1–50 (log), `random_strength` 0–5, `bagging_temperature` 0–3, `border_count` 64–254, `grow_policy` {SymmetricTree, Depthwise} |
| XGBoost | `max_depth` 3–10, `min_child_weight` 1–100 (log), `subsample` 0,5–1,0, `colsample_bytree` 0,1–0,8, `gamma` 0–5, `reg_alpha` 1e-3–10 (log), `reg_lambda` 1e-3–100 (log) |

## Ensemble

Ensemble dibangun dari OOF yang disimpan sejak hari pertama, dan setiap metode dinilai dengan nested CV agar bobotnya tidak overfit.

- **Prasyarat:** semua model memakai fold yang sama dan menyimpan OOF serta prediksi test.
- **Cek diversitas:** korelasi Spearman antar OOF. Model dengan korelasi di bawah sekitar 0,98 terhadap model terbaik paling berpotensi menambah skor.
- **Urutan metode:** (1) rata-rata rank; (2) rank berbobot, bobot non-negatif berjumlah 1 dicari dengan `scipy.optimize` atau coordinate search; (3) stacking Logistic Regression pada logit OOF; (4) hill climbing Caruana dengan pengembalian.
- **Penjaga overfit:** cari bobot di 4 fold OOF, uji di fold ke-5, ulangi untuk semua fold. Jika selisih dua metode di bawah 0,0005, pilih yang lebih sederhana.
- **Cara menggabung:** jika AUC, gabungkan rank yang dinormalisasi ke 0–1. Jika LogLoss, gabungkan logit lalu kalibrasi.
- **Kandidat final:** A = model tunggal terbaik (rata-rata seed), B = blend terbaik.

## Cabang jika metrik bukan AUC

Pipeline fitur dan model tetap sama; yang berubah hanya metrik early stopping, cara menggabung, dan pemrosesan probabilitas akhir.

| Metrik | Yang berubah |
| --- | --- |
| ROC-AUC / Gini | Hanya urutan yang penting. Rank blend aman; kalibrasi dan class weight tidak perlu. |
| LogLoss | Jangan rank blend, class weight, atau resampling. Early stopping dengan logloss, gabungkan logit, kalibrasi Platt atau isotonic di dalam CV, clip ke \[1e-4, 1 − 1e-4\], dan pastikan rata-rata prediksi mendekati 13,58%. |
| Brier | Sama seperti LogLoss; isotonic biasanya cukup. |
| PR-AUC | Pipeline sama; perhatikan performa di desil risiko tertinggi. |
| F1 / akurasi | Cari threshold optimal di OOF, jangan memakai 0,5. Kecil kemungkinannya karena kolom submission berupa probabilitas. |

## Reproduksibilitas dan notebook final

Notebook final harus bisa Run All dari nol di mesin bersih dan menghasilkan file dengan MD5 yang sama dengan submission terpilih; ini syarat lolos 8 besar.

### Checklist determinisme

- [ ] Sel pertama mencetak versi Python dan library, menyetel `SEED = 42` untuk `random`, `numpy`, dan setiap model, serta mendeteksi `DATA_DIR` lokal atau Kaggle.
- [ ] Daftar kolom selalu `sorted()`. Jangan iterasi `set` berisi string: urutannya berubah antar proses karena hash randomization Python, dan hasil model ikut berubah.
- [ ] LightGBM: `seed`, `deterministic=True`, `force_row_wise=True`, `num_threads` tetap.
- [ ] XGBoost: `seed`, `tree_method="hist"`, `nthread` tetap, CPU.
- [ ] CatBoost: `random_seed`, `thread_count` tetap, CPU.
- [ ] MLP (jika dipakai): `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, CPU. Jika tetap tidak stabil, keluarkan dari final.
- [ ] Fold dibuat ulang di notebook dengan seed yang sama, lalu di-assert identik dengan `folds.csv`.
- [ ] Parameter hasil tuning dan daftar fitur terpilih di-hardcode beserta penjelasan asalnya; `RUN_TUNING = False`.
- [ ] Tidak ada batas waktu berbasis jam seperti `time_limit`, karena membuat hasil tidak deterministik.
- [ ] Probabilitas akhir dibulatkan ke 6 desimal sebelum disimpan, untuk meredam perbedaan floating point antar mesin.
- [ ] Submission di-merge ke `sample_submission` berdasarkan `application_id`; assert 30.000 baris, ID unik dan lengkap, tanpa NaN, nilai dalam \[0, 1\].
- [ ] Sel terakhir mencetak MD5 `submission.csv`. Catat di `submissions/LOG.md` dan unggah file yang persis itu ke Kaggle.
- [ ] Fresh run H-2: restart kernel, clear output, Run All di laptop anggota lain atau Kaggle. MD5 harus sama.
- [ ] Runtime per bagian dicatat; target total di bawah 2 jam.
- [ ] Nama file `NomorPeserta_NamaTim_Notebook.ipynb`.

### Struktur notebook final

1. Ringkasan solusi dan hasil (tabel CV dan LB)
2. Setup dan reproduksibilitas
3. Memuat data
4. EDA ringkas: temuan yang membentuk fitur
5. Feature engineering per tabel beserta alasan bisnisnya
6. Strategi validasi
7. Model dan hasil per model
8. Ablation: kontribusi tiap grup fitur
9. Ensemble
10. Interpretasi: SHAP global dan beberapa contoh individual
11. Pembuatan submission dan verifikasi MD5
12. Kesimpulan dan keterbatasan

## Jadwal 10–30 Oktober

&#91;embedded content: jadwal penyisihan · fase dan milestone, 10–30 Okt 2026\]

Feature freeze 26 Okt menyisakan empat hari untuk ensemble, notebook final, dan dua kali fresh run; tenggat resmi tetap 30 Okt 23.59 WIB. Fase 7 (opsional) hanya dikerjakan jika tuning selesai lebih cepat.

Ritme harian: pagi meninjau log dan memutuskan terima/tolak, siang FE atau eksperimen, malam menjalankan job panjang seperti tuning dan CatBoost.

## Pembagian tugas tim

Bagi pekerjaan per tabel agar FE berjalan paralel, dengan satu orang memegang pipeline, submission, dan notebook final.

| Peran | Tanggung jawab | Grup fitur |
| --- | --- | --- |
| Anggota A: pipeline dan modeling | Repo, folds, `run_cv`, log, LightGBM, tuning, ensemble, satu-satunya pengirim submission, notebook final | `app_`, `prof_`, `x_` |
| Anggota B: riwayat kredit | FE repayment, previous applications, biro; CatBoost dan XGBoost | `rp_`, `pa_`, `bu_`, `bp_` |
| Anggota C: perilaku dan dokumentasi | FE account behavior dan e-commerce, visual EDA untuk notebook, draf laporan final | `ab_`, `ec_` |

Jika tim hanya 2 orang, Anggota A juga memegang `ab_` dan `ec_`.

- **Kontrak fungsi fitur:** `build_<grup>(data_dir, cutoffs) -> DataFrame` ber-index `application_id`, kolom ber-prefix dan terurut, disimpan ke `features/<grup>_v<n>.parquet`.
- **Git:** satu branch per grup fitur; merge setelah eksperimennya diterima.
- **Sinkron harian 15 menit:** hasil kemarin, keputusan terima/tolak, rencana hari ini.
- **Job panjang** (tuning, CatBoost, null importance) dijalankan malam hari di laptop yang tidak dipakai bekerja.

## Strategi submission Kaggle dan pemilihan final

Submission dipakai untuk memeriksa pipeline dan korelasi CV–LB, bukan untuk memilih fitur; pilihan final ditentukan OOF.

- **Kuota:** maksimal 1–2 per hari selama fase FE; sisakan kuota untuk 27–29 Okt.
- **Kapan submit:** saat OOF naik jelas (sekitar 0,002 atau lebih) atau setelah perubahan pipeline besar, agar korelasi CV–LB terbaca di atas noise.
- **Log tiap submission:** ID eksperimen, nama file, MD5, OOF AUC, public LB, catatan.
- **Pemilihan:** OOF dan repeated CV yang menentukan; public LB hanya cek kewajaran. Jangan memilih submission dengan public tinggi tetapi OOF lebih rendah.
- **Pilih manual di Kaggle.** Jika tidak memilih, Kaggle bisa otomatis memilih berdasarkan public LB, dan file itu mungkin bukan output notebook. Simpan screenshot halaman submission terpilih.
- **Satu atau dua pilihan:**
  - Hanya satu, atau aturan pencocokan belum jelas: pilih blend robust (kandidat B).
  - Panitia mengonfirmasi dua boleh: notebook menulis `submission_A.csv` dan `submission_B.csv`, keduanya dengan MD5 yang cocok.

## Risiko dan mitigasi

Risiko terbesar bukan skor yang kurang tinggi, melainkan gugur karena output notebook tidak cocok atau pelanggaran aturan.

| Risiko | Dampak | Mitigasi |
| --- | --- | --- |
| Output notebook ≠ submission terpilih | Posisi 8 besar diganti tim lain | Checklist determinisme, MD5, fresh run H-2 di mesin lain |
| Kaggle memilih submission otomatis | Submission terpilih bukan output notebook | Pilih manual dan simpan screenshot |
| Memakai data luar atau model pretrained | Diskualifikasi | Hanya folder `Dataset/`; tanya panitia sebelum memakai apa pun yang pretrained |
| Notebook telat | −1 poin per menit | Upload 30 Okt paling lambat 12.00 WIB |
| Overfit public LB | Private (45%) turun | Keputusan dari OOF; selisih public < ±0,005 dianggap noise |
| Fitur test dihitung beda dari train | Skor LB anjlok | Satu fungsi untuk train+test, adversarial validation, cek distribusi |
| Kebocoran waktu atau target | CV terlalu optimis | Semua fitur pakai cutoff; encoder bertarget di-fit di dalam fold |
| Metrik ternyata bukan AUC | Rank blend merusak LogLoss | Deteksi metrik di E00, ikuti cabang metrik |
| Memori atau runtime e-commerce | Notebook lambat atau crash | dtype hemat, dedup di awal, cache parquet, ukur runtime |
| Nama tim Kaggle berbeda | Masalah administrasi | Cek di Fase 0 |
| Konflik kode antaranggota | Waktu terbuang | Kontrak fungsi fitur, prefix, branch per grup |

## Triase daftar 240 teknik dan 545 model

Hanya sebagian kecil dari daftar itu yang cocok untuk data tabular relasional ini; sisanya untuk gambar, teks, audio, forecasting, atau dilarang aturan. Daftar mana pun tidak akan pernah lengkap, dan memang tidak perlu: skor ditentukan oleh sedikit teknik yang tepat yang dieksekusi dengan disiplin.

| Kelompok (nomor di daftarmu) | Keputusan | Alasan |
| --- | --- | --- |
| Data preparation (1–18) | Pakai integrasi tabel, dedup, pencegahan leakage, stratified split. Lewati data collection, labeling, group/time split, buang outlier. | Data hanya dari panitia; split acak; GBDT tahan outlier. |
| EDA (19–27) | Secukupnya, fokus ke ide fitur. | Temuan utama sudah ada di CLAUDE.md. |
| Feature engineering (28–50) | Inti pekerjaan: agregasi, interaksi, rasio domain, binning untuk LR. Lewati PCA, polynomial, scaling untuk GBDT. | Pohon tidak butuh skala; PCA mengaburkan sinyal. |
| Data augmentation (51–61) | Lewati semua. | SMOTE tidak menaikkan AUC GBDT; MixUp, CutMix, teks, audio tidak relevan. |
| Model selection (62–88) | Lihat bagian Model. | — |
| Hyperparameter optimization (89–101) | Optuna TPE saja. | Grid, random, genetik, PSO kurang efisien. |
| Training optimization (102–123) | Early stopping dan lr kecil untuk GBDT; optimizer dan normalisasi hanya untuk MLP. Lewati transfer learning, pretraining, fine-tuning model pretrained. | Dilarang data luar; tidak ada model pretrained yang relevan. |
| Regularization (124–133) | L1/L2, `min_child_samples`, `feature_fraction`; dropout untuk MLP. | Sudah tercakup di tuning. |
| Validation (134–147) | Stratified K-Fold, repeated, OOF, bootstrap AUC untuk keputusan tipis. Lewati LOOCV dan time-series CV. | Split acak; LOOCV terlalu mahal. |
| Ensemble (148–159) | Blending, stacking, weighted, seed averaging. Lewati TTA dan mixture of experts. | TTA untuk gambar. |
| Error analysis (160–172) | Analisis segmen; kalibrasi jika LogLoss; threshold hanya jika metrik berbasis label. | — |
| Advanced (173–189) | Pseudo-labeling opsional. Lewati sisanya. | Active learning, federated, continual, few-shot tidak relevan. |
| Data-specific (190–203) | Missingness indicator, fitur temporal, lag, rolling statistics; class weighting diuji sekali. | Tabel pendukung adalah deret waktu per customer. |
| Metrik (204–216) | Ikuti metrik Kaggle. | — |
| Reproducibility (217–226) | Pakai semua. | Syarat lolos. |
| Pemilihan pendekatan (227–240) | Leakage audit dan adversarial validation. Drift, retraining, champion–challenger masuk laporan final. | Relevan untuk presentasi, bukan skor. |
| 18 kategori tambahan | Pakai SHAP-based selection, ablation, kalibrasi (jika LogLoss). Lewati MICE/MissForest, NAS/CMA-ES, LoRA, causal, adversarial training, graph, TFT/N-BEATS, NLP, computer vision, label error detection. | GBDT menangani NaN sendiri; ini bukan forecasting; tidak ada teks, gambar, atau graf. |
| Daftar 545 model, bagian 3–18 | Hampir semua dilewati: clustering, computer vision, NLP, reinforcement learning, rekomendasi, foundation model, AutoML (kecuali benchmark). | Bukan klasifikasi tabular atau melanggar larangan data luar. |

## Setelah penyisihan: rencana babak final

Mulai draf laporan dan presentasi segera setelah 30 Okt, karena jendela pengumpulan final hanya 14–19 Nov dan format laporan baru diumumkan 14 Nov.

1. **31 Okt–10 Nov:** bekukan repo dengan tag git pada commit submission final. Susun draf laporan dan slide dari notebook, lalu latihan presentasi internal pertama.
2. **11 Nov:** pengumuman finalis.
3. **Registrasi ulang paling lambat Nov 13, 2026.** Tanpa ini, tim dianggap mengundurkan diri.
4. **14 Nov:** technical meeting final; catat format laporan dan kriteria penilaian.
5. **Kirim paling lambat Nov 19, 2026 23.59 WIB:** `NomorPeserta_NamaTim_laporan.pdf` dan `NomorPeserta_NamaTim_presentasi.pdf`.
6. **Final luring di UI Depok, Nov 21, 2026:** 10 menit presentasi dan 10 menit tanya jawab. Pesan transportasi dan akomodasi segera setelah pengumuman.

### Isi laporan dan presentasi

- Masalah dan konteks bisnis, data, dan temuan EDA kunci.
- Fitur: 10 fitur teratas menurut SHAP beserta arti bisnisnya.
- Validasi, model, ensemble, hasil CV vs LB, dan tabel ablation.
- Penerapan bisnis: risk band dan simulasi cut-off (tingkat persetujuan vs tingkat default pada OOF).
- Etika dan keadilan: pengaruh usia dan region, serta performa per segmen.
- Keterbatasan dan rencana produksi: monitoring drift, retraining, champion–challenger.

### Persiapan tanya jawab

- Mengapa GBDT, bukan deep learning?
- Bagaimana memastikan tidak ada kebocoran waktu?
- Mengapa CV lebih dipercaya daripada public LB?
- Bagaimana menangani 30% customer tanpa riwayat?
- Fitur apa yang paling berpengaruh, dan apakah masuk akal secara bisnis?
- Apakah model adil untuk semua kelompok usia dan region?

## Lampiran

Struktur folder, skema log, dan parameter dev ini menjadi standar bersama semua anggota sejak hari pertama.

### Struktur folder

```
Dataset/                 # data panitia, read-only
Guidebook/
notebooks/               # eksplorasi
src/
  config.py              # SEED, N_FOLDS, DATA_DIR
  io_utils.py            # loader hemat memori
  features/              # app.py, repayment.py, previous.py, bureau.py,
                         # account.py, ecommerce.py, cross.py
  train.py               # run_cv(model, X, y, folds) -> oof, test_pred, log
  blend.py
features/                # cache parquet
folds/folds.csv
oof/                     # <exp_id>.npy
preds/                   # <exp_id>.npy
experiments/log.csv
submissions/             # csv + LOG.md
final/                   # notebook final
requirements.txt
```

### Kolom `experiments/log.csv`

`exp_id`, `tanggal`, `pembuat`, `deskripsi`, `feature_set`, `n_fitur`, `model`, `params_hash`, `auc_f1`–`auc_f5`, `oof_auc`, `std_fold`, `delta_vs_best`, `diterima`, `runtime_menit`, `git_commit`, `submission_file`, `md5`, `public_lb`.

### Parameter LightGBM dev dan aturan terima

```python
LGB_DEV = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=63, min_child_samples=100,
    feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, max_bin=255,
    seed=42, deterministic=True, force_row_wise=True,
    num_threads=4, verbose=-1,
)
# lgb.train(LGB_DEV, dtrain, num_boost_round=5000, valid_sets=[dvalid],
#           callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])

def terima(oof_baru, oof_best, fold_baru, fold_best, min_delta=0.0005):
    naik_fold = sum(b > a for b, a in zip(fold_baru, fold_best))
    return (oof_baru - oof_best) >= min_delta and naik_fold >= 4
```
