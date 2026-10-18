# Riset dan Rencana Peningkatan Skor — DSC LOGIKA UI 2026

Dokumen ini terpisah dari `Planning Model DSC LOGIKA UI 2026.md`. Isinya: riset teknik dari solusi juara kompetisi kredit dunia, hasil uji langsung ke data (12 Okt 2026), analisis leaderboard, dan daftar eksperimen lanjutan (ID E120 ke atas) beserta urutan prioritasnya.

## 0. Ringkasan

- **Posisi kita (12 Okt)**: public **0,54685** (P5 single, LightGBM tuned, bag 3 seed), **peringkat 30**. CV: mean-fold AUC 0,8379, OOF KS 0,5138.
- **Leaderboard 12 Okt**: #1 = **0,58**, di bawahnya sangat rapat. **Target akhir 0,65** dipakai sebagai perkiraan skor #1 di akhir lomba (30 Okt), dengan target antara 0,56 → 0,58 → 0,60 (bagian 8).
- **Gap ke #1 sekarang (0,58)**: +0,033 KS public ≈ +0,018 AUC. Sebagian gap ini noise subset public: simulasi (bagian 2) menunjukkan peringkat atas dari puluhan tim yang rapat cenderung "beruntung" sekitar +0,010–0,016. **Gap nyata kira-kira +0,01 AUC**: berat, tapi bisa dikejar dengan menumpuk beberapa kenaikan.
- **Gap ke 0,65**: +0,10 KS ≈ +0,05 AUC, kira-kira 4× seluruh kemajuan kita sejak baseline E03. Tuning/ensemble hanya +0,001–0,005 AUC, jadi 0,65 hanya mungkin jika ada sinyal besar baru (grup A).
- **Uji data hari ini** (bagian 3): tidak ada kebocoran lewat ID, urutan baris, sambungan antartabel, atau waktu. Tidak ada efek musiman/region yang belum tertangkap. Fitur agregat e-commerce baru dan fitur "utang di lembaga lain" **tidak menambah sinyal** (sudah terserap).
- **Satu sinyal baru ditemukan**: model level order e-commerce (bagian 3 butir J) menambah **+0,0020 AUC di 5/5 fold, KS +0,0046** di screen. Model level baris untuk tabel lain praktis nol.
- **Artinya**: tidak ada satu "jurus" besar yang terlihat. Mengejar 0,58 butuh menumpuk beberapa kenaikan kecil yang nyata (masing-masing +0,001–0,002 AUC) ditambah satu-dua temuan baru dari grup A.
- **Urutan kerja yang disepakati** (detail di bagian 8, dimulai saat user bilang "lanjut"):
  1. **E140 + E144 – batas iterasi 15.000 + seed lebih banyak** (Fase 5): kenaikan kecil tapi hampir pasti.
  2. **E130 – seleksi fitur ulang dengan parameter tuned** (Fase 3).
  3. **E121 – model level order e-commerce** jadi fitur, untuk segmen belanja dekat cutoff. Ini satu-satunya sumber sinyal baru yang terbukti.
  4. **E120, E125, E128 – segmen tanpa riwayat**: analisis kasus salah, fitur hasil prediksi, model khusus segmen.
  5. Sisa waktu untuk grup A/C (GRU, DART, NN, fitur otomatis), lalu tuning ulang, feature freeze 26 Okt, ensemble, dan notebook final.

## 1. Posisi saat ini

| Submission | Mean-fold AUC | OOF KS | Public | Catatan |
|---|---|---|---|---|
| E03 (baseline B1) | 0,8255 | 0,491 | 0,52436 | agregat naif semua tabel |
| E31 (B2) | 0,8313 | 0,500 | 0,53801 | + fitur Fase 2 |
| E71b (B3, 176 fitur) | 0,8325 | 0,505 | 0,54082 | seleksi null importance |
| P4 blend | 0,8364 | 0,512 | 0,53867 | blend 1 seed (terlalu optimis) |
| E80 CatBoost | 0,8346 | 0,509 | 0,54487 | |
| P5 Caruana | 0,8379 | 0,515 | 0,54435 | blend 5 model tuned |
| **P5 single LGB tuned** | **0,8379** | **0,514** | **0,54685** | **terbaik** |

Pola: public konsisten sekitar +0,033 di atas OOF KS. Blend sudah dua kali kalah dari model tunggal di public.

## 2. Seberapa besar gap-nya, dan berapa yang noise?

### 2a. Konversi skor public ke AUC

Pendekatan binormal: KS = 2Φ(d/2) − 1 dan AUC = Φ(d/√2). Pendekatan ini cocok dengan model kita (AUC 0,838 ↔ KS 0,514, persis OOF kita). Dengan offset public ≈ OOF + 0,033:

| Skor public | OOF KS setara | AUC setara | Selisih AUC dari kita |
|---|---|---|---|
| 0,547 (kita, #30) | 0,514 | 0,838 | — |
| **0,58 (#1 sekarang)** | **0,547** | **0,856** | **+0,018** |
| 0,60 | 0,567 | 0,866 | +0,028 |
| **0,65 (target akhir)** | **0,617** | **0,891** | **+0,053** |

Sebagai pembanding, seluruh kemajuan kita dari E03 ke P5 adalah +0,012 AUC → +0,022 KS public (rasio ≈ 1,8 KS per AUC).

### 2b. Noise subset public (simulasi `experiments/riset_12okt/probe9_lb_noise.py`)

Subset acak OOF seukuran public (9.000 = 30% atau 15.000 = 50% dari test), 1.000 kali:

| Ukuran public | sd KS satu model | sd selisih 2 model, korelasi 0,988 (LGB vs CatBoost tuned) | korelasi 0,976 (LGB tuned vs E71b) | korelasi 0,941 (LGB vs LogReg) |
|---|---|---|---|---|
| 9.000 | 0,0118 | 0,0042 | 0,0055 | 0,0077 |
| 15.000 | 0,0085 | 0,0034 | 0,0041 | 0,0057 |

Tafsiran:

- Model tim lain kemungkinan berkorelasi 0,90–0,97 dengan model kita, jadi selisih skor public antar-tim punya noise sd ±0,005–0,01 **meski kekuatan aslinya sama**.
- Dari 30+ tim yang rapat, yang tampil di puncak cenderung diuntungkan noise: perkiraan +2 sd ≈ **+0,010–0,016**. Makin banyak submission per tim (sampai 3/hari × 20 hari), makin besar efek "memilih yang beruntung" ini.
- Jadi dari gap 0,033 ke #1, kira-kira separuhnya bisa noise, dan **gap nyata sekitar +0,01 AUC**.
- Ini juga berarti skor #1 akan terus naik sebagian karena seleksi noise. Kenaikan itu tidak terbawa ke private. Karena nilai akhir = public 45% + private 45%, kekuatan asli (CV) tetap jadi pegangan utama.
- Selisih public di bawah ±0,005 antara dua submission kita sendiri tidak berarti apa-apa.

## 3. Hasil investigasi data (12 Okt 2026)

Script uji: `experiments/riset_12okt/` (probe1–3: kebocoran, waktu, segmen; probe4: utang eksternal; probe5: screen e-commerce agregat; probe6–8: model level baris). Semua memakai OOF `oof/T_lgb_b3_4f3c7f.npy` sebagai pembanding.

| # | Yang dicek | Hasil | Implikasi |
|---|---|---|---|
| A | Urutan baris train, angka hex `application_id`/`customer_id` vs target | AUC 0,501 / 0,500 / 0,504 | Tidak ada kebocoran ID/urutan |
| B | ID silang antartabel (previous app = aplikasi train/test? loan_id tanpa induk? customer beda?) | Semua 0 | Relasi tabel bersih |
| C | Waktu: previous app setelah aplikasi saat ini; laporan biro setelah aplikasi; bulan rekening terakhir; order terakhir | 0 previous app; 1% laporan biro dibuat setelah aplikasi (tetap sebelum keputusan); bulan rekening terakhir **selalu** bulan sebelum keputusan; order e-commerce terakhir **selalu ≥ 12 jam** (praktis ≥ 24 jam) sebelum keputusan | Data dipotong rapi (dictionary "v2"), tidak ada info masa depan |
| D | Lama proses aplikasi → keputusan | AUC 0,502 | Tidak informatif |
| E | Efek musiman / shock region × waktu | Target per bulan 12,0–14,7%; residual model per region × bulan sd 0,0092 = level noise | Tidak ada pola makro tersembunyi |
| F | Profil customer duplikat | 0 | Tidak ada "orang yang sama" lintas train/test |
| G | Tier bunga | KS bunga saja 0,230; AUC model di dalam tiap tier 0,81–0,83 | Model bekerja di semua tier |
| H | Kandidat e-commerce baru: waktu sejak order ke-k (k=1..10), jumlah order 3–90 hari, akselerasi vs baseline, porsi COD/kartu kredit/batal | Univariat sampai AUC 0,700, tapi **screen di atas OOF terbaik: +0,00002 AUC** | Sinyal e-commerce sudah terserap model |
| I | Utang/akun/inquiry di lembaga lain (biro − pinjaman internal) | Korelasi dengan residual terkalibrasi ≤ 0,006 (noise 0,0038), residual per desil ≤ ±0,008 | Tidak ada sinyal baru |
| J | Model level baris (row-level) per tabel: LightGBM di level baris dengan label target customer, OOF per fold, diagregasi per customer, lalu screen di atas OOF terbaik | **E-commerce (per order): +0,0020 AUC, naik 5/5 fold, KS 0,5138 → 0,5184**. Rekening +0,0001 (4/5), cicilan +0,0001 (3/5), biro +0,0000, aplikasi lama −0,0000 | **Satu-satunya sinyal baru yang ditemukan hari ini ada di pola level order e-commerce** |

Detail butir J:

- Rata-rata prediksi level order per customer punya AUC 0,741, lebih tinggi dari fitur e-commerce tunggal mana pun (terbaik 0,700).
- Penggerak model order (gain): waktu order relatif ke keputusan 35%, nominal order 25%, kuantitas 8,5%, lalu porsi kategori (groceries, home, electronics) dan metode bayar. Jadi sinyalnya ada di **kombinasi kapan × berapa × kategori apa**, yang tidak tertangkap agregat per jendela waktu.
- Agregat yang paling berguna: rata-rata prediksi order 30 hari terakhir (menaikkan screen dari +0,0011 ke +0,0020).
- **Jangan beri model baris fitur konteks customer** (urutan order, jumlah order customer, nominal relatif ke median customer, jumlah order 7/30 hari sebelumnya). Versi "kaya" ini justru turun ke +0,0001, karena prediksinya meniru agregat yang sudah dimiliki model utama. Fitur model baris harus intrinsik milik baris itu.
- Screen ini tidak nested, jadi sedikit optimis. Perkiraan realistis di CV penuh: +0,001–0,002 AUC, atau sekitar 10–20% dari gap nyata ke #1 (bagian 2b).

Lanskap sinyal (AUC univariat 524 kolom cache): 99 kolom ≥ 0,65; terkuat riwayat bayar (`rp_share_dpd1` 0,784), lalu intensitas e-commerce 90 hari 0,698, prediksi tier bunga 0,693, skor biro 0,675. Rekening (`ab_`) paling kuat hanya 0,576 dan profil 0,515.

## 4. Dari mana tim teratas mendapat skor lebih tinggi?

Situasi 12 Okt: #1 = 0,58, di bawahnya rapat, kita #30 (0,547).

| Penjelasan | Seberapa mungkin | Implikasi / cara menguji |
|---|---|---|
| **Noise subset public + memilih submission yang beruntung** | Tinggi, menjelaskan sekitar separuh gap (bagian 2b) | Jangan mengejar public; pegang CV |
| **Perbaikan nyata yang bertahap** (fitur, tuning, ensemble yang lebih baik dari kita) | Tinggi, sisa gap sekitar +0,01 AUC | Grup A–D di bagian 5 |
| **Sinyal di pola level baris / urutan waktu** yang hilang saat diagregasi | Terbukti ada tapi kecil: e-commerce +0,002 AUC; tabel lain ≈ 0 (butir J). Model urutan (GRU) belum dicoba | E121, E122, E120 |
| **Notebook publik** yang dipakai banyak tim (kluster rapat sering berasal dari sini) | Perlu dicek | Tab Code |
| Anomali (akun panitia, bocoran data lama) | Kecil setelah info 12 Okt (#1 tidak jauh dari kluster) | — |
| Data dari luar | Tidak bisa diikuti (melanggar aturan) | — |

### Langkah 0 — intel leaderboard (sisa)

1. Jumlah tim, dan skor #5, #10, #20 (untuk melihat kerapatan).
2. Tab **Code**: adakah notebook publik dengan skor tinggi? Hanya dibaca untuk ide. **Jangan menyalin**, karena guidebook menganggapnya plagiarisme.
3. Tab **Discussion**: pengumuman host (update data, klarifikasi aturan), atau ide yang dibagikan peserta.
4. Ulangi pengecekan ini seminggu sekali, karena dinamika leaderboard berubah.

## 5. Katalog teknik

Perkiraan dampak adalah perkiraan kasar dari kompetisi serupa, bukan janji. Semua keputusan tetap memakai aturan terima di CLAUDE.md (mean-fold AUC, ≥ 4/5 fold, konfirmasi 3 seed untuk keputusan tipis).

### A. Penemuan sinyal (potensi terbesar)

| ID | Teknik | Dasar / sumber | Dampak | Biaya | Prioritas |
|---|---|---|---|---|---|
| E120 | **Analisis kasus salah**: 30 FN (defaulter dengan prediksi terendah) dan 30 FP, baca timeline semua tabel per customer, cari pola | Playbook Grandmaster: "smarter EDA"; cara klasik menemukan "magic feature" | Tidak pasti, bisa besar | 2–3 jam | **1** |
| E121 | **Model level baris e-commerce**: LightGBM di level order (fitur intrinsik order saja: waktu relatif ke keputusan, nominal, kuantitas, porsi kategori, metode bayar, status, jam, hari), label target customer, prediksi OOF **nested** (untuk fold luar k, model baris tidak boleh melihat label fold k), diagregasi (mean, max, std, 3/10 order terakhir, 30 hari terakhir) jadi fitur. Variasi: level item, dan tambah pohon/iterasi | Screen 12 Okt: **+0,0020 AUC, 5/5 fold**. Tabel lain ≈ 0, jadi tidak perlu | +0,001–0,002 AUC (nyata) | 1 hari | **Diterima 12 Okt: +0,00455 AUC (3 seed, 5/5 fold, CI 95% [+0,0035; +0,0055]), KS 0,5142 → 0,5207**. Audit: label acak menambah +0,00003 (aman). Varian ditolak: E121b model order lebih besar (255 daun) 0,84253; E121c + model level item 0,84259 (vs 0,84261); E121d + model baris rekening/cicilan/biro/aplikasi lama 0,84224 (−0,00037, 1/5 fold); E121e (13 Okt, `e121.py --build --aggx --tag _e`) 9 agregat waktu tambahan (rata-rata 7/90 hari, peluruhan half-life 7/30 hari, kuantil 90, 3 tertinggi, jumlah order berisiko 30 hari/semua, tren) di atas E130a: screen +0,00009, 2/5 fold, |z| maks 2,0 → ditolak di screen. **Public E121: 0,55567** |
| E122 | **Model urutan (GRU/1D-CNN)** untuk e-commerce dulu (deret order/item 180 hari terakhir), baru tabel lain jika berhasil. OOF prediksi/embedding → fitur GBDT | Juara 2 Home Credit (Conv1D/LSTM pada tabel anak); pemenang AmEx memakai GRU di ensemble; tim top-10 AmEx RNN + XGBoost +0,003. Screen E121 menunjukkan sinyal e-commerce ada di level urutan/baris | Sedang | 2–3 hari, perlu torch (sebaiknya GPU Kaggle) | **2**. **13 Okt notebook siap: `kaggle/e122_gru.ipynb`** (sumber `kaggle/e122_gru.py`): urutan semua order (≤96, kronologis, fitur intrinsik order + jarak antar order) → GRU 64 + attention pooling; nested 5×(1+4) model × 3 seed; early stopping di 10% holdout internal; keluaran `e122_gru_v1.npz` format e121 + OOF GRU tunggal. Uji lokal CPU (fold 0, 15,6k customer latih, 6 epoch): AUC val **0,757** (≥ `ecm_mean` 0,73–0,755). Data: Kaggle Dataset privat `dsc-logika-exp` dari `kaggle/make_dataset.py`. Impor: `kaggle/import_results.py`, lalu `e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1.npz --basefs all --name E122`. Audit: jalankan ulang dengan `SHUFFLE = True` (label acak) bila diterima. Varian berat (13 Okt): `e122_bigru.ipynb` (GRU 2 lapis dua arah 128) dan `e122_tf.ipynb` (Transformer 3 lapis d=128), keluaran `e122_gru_v1_<varian>.npz`. **Hasil 13 Okt malam (GRU 64, Kaggle T4): model GRU tunggal mean-fold AUC 0,7640** (lebih kuat dari fitur `ecm_*` terbaik `ecm_last10` 0,749; Spearman ke `ecm_last10` 0,92). **E122 = E130a + `seq_gru`: bag 3 seed 0,84411, KS 0,5217 (+0,00097 vs E130a, 5/5 fold, bootstrap P 1,00) → DITERIMA, acuan baru.** Audit: kode nested benar; AUC fitur latih ≈ validasi (0,7635 vs 0,7637); distribusi va ≈ test (KS < 0,01), adversarial 0,499; uji label acak menunggu `kaggle/e122_gru_shuffle.ipynb` (1 seed). Temuan: fitur bagian latih berasal dari 4 model dalam yang skala logit-nya bergeser (KS tr~va sampai 0,08) → `kaggle/rank_blocks.py` (persentil per blok model): **E122r 0,84429 (+0,00019 vs E122, 4/5, P 0,97) → kandidat**, E122rr (ecm juga dipersentil) 0,84427 → tidak berpengaruh. Blend rank E122r + E141 (w 0,3) 0,84441 (+0,00012). **14 Okt:** uji label acak (`e122_gru_shuffle`, 1 seed): AUC fitur vs target asli 0,471 (satu permutasi dipakai semua fold, jadi penyimpangan acaknya ikut sama); uji penentu E122s (fitur label acak di CV) 0,84305 vs E130a 0,84314 → tidak membawa info target, **audit lolos**. Varian: biGRU 2×128 0,7642, Transformer 3×128 0,7625, GRU 0,7640 (Spearman 0,98–0,99). E122m (rata-rata 3 varian, 1 kolom) 0,84433; **E122c (3 kolom) 0,84454** (+0,00044 vs E122, 5/5, P 1,00) → kandidat terbaik. Public E122c + NN 40%: 0,55757 (vs 0,55858 E130a+E141, selisih dalam noise). **E122x (14 Okt, notebook siap)**: model urutan untuk tabel lain — `kaggle/make_seq_tables.py` (cicilan 18 fitur/baris maks 113 langkah; rekening 13 fitur maks 23 bulan; aplikasi lama + ringkasan biro miliknya 21 fitur maks 11), `kaggle/e122x_seq.py` (encoder GRU 64 per tabel + attention pooling, penanda tabel kosong; varian rp/ab/pa dan all = 4 encoder termasuk e-commerce; nested sama dengan E122). Uji lokal fold 0 (24k customer, 8 epoch): rp 0,759, ab 0,642, pa 0,634. **Hasil Kaggle 15 Okt**: rp 0,7531, ab 0,6519, pa 0,6542, all (4 encoder) **0,8279** sendirian — hampir setara GBDT 538 fitur, tetapi Spearman ke GBDT 0,93. Sebagai fitur GBDT kenaikannya kecil: E122x1 (+rp/ab/pa) 0,84495 (+0,00041 vs E122c, 4/5, P 0,99), E122x2 (+all) 0,84487 (+0,00033, 5/5), E122x3 0,84495 → informasi tabel non-e-commerce sebagian besar sudah tertangkap fitur agregat. Di blend: E122x1 + NN_seq 50:50 = **0,84576** (+0,0005 vs E122c+NN40); menambah model all langsung 5% hanya +0,00003. **Public 15 Okt**: E122x1+NNseq50 0,55563, E122c+NNseq50 0,55527, NNseq saja 0,55265 (vs E130a+E141 0,55858). Simulasi noise selisih berpasangan (OOF, subset 9k/15k): semua z −0,5 s.d. −1,5 → masih noise, tetapi blend dengan porsi NN besar konsisten lebih rendah. **Temuan: NN E142 memilih epoch terbaik pada fold validasi luar** (`clf.fit(..., B_va, y[va])`) → OOF NN optimis, test tidak. Diperbaiki: `INNER_VAL = True` di `kaggle/e142_tabnn.py` (holdout bertingkat 10% data latih fold); notebook `e142_realmlp_seqx_iv`/`e142_tabm_seqx_iv.ipynb` (TAG `_seqx_iv`). Sampai hasil _iv ada, bobot NN di blend dianggap belum terpercaya; kandidat robust = GBDT (E122c+E141 / E122x1). |
| E123 | **Featuretools DFS** dengan `cutoff_time` per aplikasi: eksplorasi otomatis agregat bertingkat antartabel | Deep Feature Synthesis, dirancang untuk data relasional bertimestamp | Sedang | 1 hari + RAM besar | 3 |
| E124 | **OpenFE** pada matriks level aplikasi: pencarian otomatis interaksi aritmetika | ICML 2023; fitur OpenFE + XGBoost sederhana mengalahkan 99,3% tim IEEE-CIS Fraud | Kecil–sedang | 0,5 hari | 3 |
| E125 | **Fitur hasil prediksi** ("predicted feature"): perluas `x_rate_pred` (AUC 0,693 > bunga asli 0,646). Prediksi skor biro untuk yang unscored, income rekening untuk yang tanpa data rekening, dan approval lender dari pola keputusan lama | Juara 2 Home Credit: fitur terpenting adalah tenor pinjaman yang diprediksi model. **13 Okt DITOLAK (`e125.py` → `features/x_pred_v1.parquet`, 12 kolom)**: model bantu tanpa target (input profil/aplikasi/biro saat ini/e-commerce → label bantu DPD terburuk, rasio bayar, porsi ditolak, income & saldo rendah rekening; cross-fit 5 fold, korelasi prediksi-aktual 0,16–0,77) + residualnya. CV E130a + x_pred (`e121.py --extra x_pred_v1.parquet`) bag 3 seed 0,84295 vs 0,84314 (−0,00019, 1/5). Info ini sudah tersirat di fitur yang ada | Kecil–sedang | ditolak |
| E126 | **Augmentasi dari pinjaman lama**: model PD level pinjaman dilatih pada pinjaman lama yang sudah ada hasilnya (pernah DPD ≥ 90), fitur dihitung *as-of* keputusan pinjaman lama (laporan biro previous app tersedia), lalu skor untuk aplikasi saat ini | Pinjaman lama = contoh tambahan dari proses yang sama | Sedang | 2 hari | 3 |
| E127 | **KNN target mean** (OOF, nested) di ruang fitur terkuat yang distandardisasi: rata-rata target 100–500 tetangga terdekat | Ide "neighbors target mean" dari forum Home Credit (detail perlu diverifikasi) | Kecil–sedang | 0,5 hari | 3 |
| E128 | **Model khusus segmen tanpa riwayat** (30% data, KS 0,466) + fitur khusus segmen | Planning E112. **13 Okt DITOLAK (`e128.py`)**: model segmen (param tuned, set E130a, kolom kosong dibuang) lebih lemah dari model global di segmennya sendiri: tanpa riwayat 0,8244 vs global 0,8258 (lr 0,02, bag 3 seed), punya riwayat 0,8473 vs 0,8488 (screen 1 seed). Gabungan logit terbaik w=0,25: +0,00004 (4/5). Model global sudah menangkap interaksi segmen | Kecil | ditolak |

### B. Penyempurnaan fitur

| ID | Teknik | Catatan | Dampak | Prioritas |
|---|---|---|---|---|
| E130 | Seleksi fitur ulang memakai **parameter tuned**: null importance ulang + uji 528 kolom penuh | Parameter tuned jauh berbeda (depth 4, feature_fraction 0,13) dari saat seleksi Fase 3. **E130a (12 Okt): 528 kolom + ecm_* 0,84314 vs E121 0,84261 (+0,00053, 4/5 fold, 3 seed) → diterima**, KS 0,5231. Blend rank E121+E130a +0,00025 AUC tapi KS turun (simpan untuk Fase 6) E130b: null importance ulang dengan param tuned → 333/528 kolom, 0,84350 (+0,00036 vs E130a, 3/5 fold, KS turun) → ditolak | Kecil–sedang | selesai: pakai 528 kolom (E130a) |
| E131 | Bobot peluruhan waktu (half-life 30/90/180 hari) untuk cicilan, rekening, order | Perilaku terbaru biasanya lebih prediktif | Kecil | 3 |
| E132 | Fitur gaya AmEx untuk deret bulanan: `last`, `last − mean`, `last / mean`, slope, untuk rekening dan cicilan per bulan | Pola umum solusi AmEx | Kecil | 3 |
| E133 | ~~Recency order ke-k, jendela pendek, porsi COD/kartu kredit~~ dan ~~utang/inquiry eksternal~~ | **Sudah diuji 12 Okt: tidak menambah** (bagian 3 butir H, I) | — | ditolak |
| E134 | Panel bulanan semua tabel pada grid waktu yang sama (bulan −1..−12 relatif cutoff) | Fondasi untuk E122 dan E132 | Tidak langsung | bersama E122 |

### C. Model

| ID | Teknik | Dasar | Dampak | Prioritas |
|---|---|---|---|---|
| E140 | **Batas iterasi 15.000** untuk LGB dan CatBoost (banyak fold mentok di 5.000) | Hasil Fase 5. **E140a (LGB, 12 Okt): B3 +0,00009 (3/5 fold), sel_b2 +0,00041 (4/5 fold), 3 seed → ditolak menurut aturan, tapi dipakai sebagai default** karena early stopping tetap memilih iterasi terbaik (tidak merugikan). Iterasi terbaik ternyata 3.400–6.900, bukan jauh di atas 5.000 | Sangat kecil | selesai (LGB) |
| E141 | LightGBM **DART** | Juara 2 Home Credit: DART + regularisasi ≈ +0,001 CV dan private. **13 Okt screen (`e141.py`, set E130a, param pohon tune_lgb, lr 0,05, drop 0,1, skip 0,5, 3000 ronde, 1 seed): 0,84284 vs GBDT seed 42 0,84291; kurva masih naik (2500: 0,84258 → 3000: 0,84284). Spearman ke GBDT 0,995; GBDT+DART 0,84322 vs GBDT 2 seed 0,84303 (+0,0002 diversitas). Full 13 Okt (5000 ronde × 3 seed, ±32 mnt/seed): **0,84352 / KS 0,5223 vs E130a 0,84314 (+0,00038, naik 5/5 fold, bootstrap CI [0,0000; +0,0008], P 0,97)** → di bawah ambang +0,0005, jadi tidak menggantikan E130a, tetapi **kandidat kuat Fase 6**: blend rank E130a + 0,5·E141 0,84353 (+0,00039, 5/5), + 0,7·E141 0,84358. Kurva seed 42/43 masih naik di ronde 5000 (+0,00015–0,00016 per 500 ronde) → opsi 7000 ronde (±2,5 jam). Catatan: bag di log e141.py awalnya ikut file `E141_scr_s42` (glob), sudah diperbaiki** | Kecil | kandidat ensemble **E141b (7000 ronde, 13 Okt malam) dihentikan setelah 2 seed: seed 42 0,84324 vs 0,84342 di 5000 ronde, seed 43 0,84312 vs 0,84321; kurva memuncak di ronde ±5000–5100 lalu turun → 5000 ronde sudah optimal untuk lr 0,05.** |
| E142 | NN tabular modern: **RealMLP / TabM** (pytabkit) | RealMLP NeurIPS 2024; TabArena 2025 (NN mengejar GBDT dengan ensemble). **13 Okt notebook siap: `kaggle/e142_tabnn.ipynb`** (pytabkit 1.7.3, dilatih dari nol): set E130a (528 kolom + ecm nested), indikator per pola NaN unik + imputasi median + QuantileTransformer (fit di fold latih), validasi = fold luar, 3 seed, anggaran 10 jam. Uji lokal CPU (8% data, 2 epoch): RealMLP 0,779, TabM 0,812. Dijalankan sebagai dua notebook 5 seed: `e142_realmlp.ipynb`, `e142_tabm.ipynb` | Kecil sendiri, berguna untuk diversitas | 3 **Hasil 14 Okt (Kaggle T4, 5 seed): RealMLP per seed 0,8413–0,8426, bag 0,84282; TabM per seed 0,8429–0,8431, bag 0,84332** (setara GBDT; Spearman ke E122c 0,980/0,983). Blend rank E122c + NN: RealMLP 30% +0,00055, TabM 30% +0,00058, keduanya 40% **+0,00073 (0,84527, 5/5 fold)**; bobot dipilih nested (0,4 di 4/5 fold) 0,84525. Kandidat utama ensemble Fase 6. Lanjutan: NN dengan fitur seq_*, lebih banyak seed |
| E143 | Fitur DAE swap-noise (gaya Porto Seguro) | Juara 2 Home Credit: public naik, private datar | Kecil, berisiko | 5 |
| E144 | **Seed bagging 5–10 seed** untuk model final | Playbook Grandmaster #7; 3 seed sudah memberi +0,002 AUC. **12 Okt: LGB B3 r15k 5 seed 0,83806 vs 3 seed 0,83798 (+0,00008)** | Sangat kecil | dipakai di final |
| E145 | **Refit 100% data** dengan iterasi tetap (≈ rata-rata best_iter × 1,1) | Playbook Grandmaster #7 | Kecil | 3 |
| E146 | Tuning ulang singkat setelah fitur berubah; XGBoost sampai 60 trial | Parameter optimal bergeser saat fitur berubah. **13 Okt selesai (`tune_e146.py`, study `tune_lgb_e146`, 43 trial): fold 0–2 naik 0,84670 → 0,84750, tetapi CV 5 fold bag 3 seed 0,84313 vs E130a 0,84314 (−0,00001, 2/5 fold; naik di fold 0 & 2, turun di fold 3 & 4) → DITOLAK. Pelajaran: tuning di fold 0–2 overfit ke fold tuning; tuning ulang 25 Okt wajib memakai 5 fold atau dilewati. OOF `E146` disimpan untuk Fase 6 (blend rank E130a+E146 +0,00013, Spearman 0,998)** | Kecil | ditolak |

### D. Ensemble dan pasca-proses (Fase 6, setelah feature freeze)

| ID | Teknik | Catatan |
|---|---|---|
| E150 | Rata-rata rank / hill climbing (Caruana) memakai OOF ≥ 3 seed | Sudah ada pratinjau; blend dua kali kalah di public, jadi wajib cek stabilitas KS. **13 Okt bahan: E150_xgb (XGB tuned + ecm, set E130a, `e121.py --study xgb --basefs all`) bag 3 seed 0,84230 / KS 0,5206, Spearman ke E130a 0,994. Blend rank E130a + 0,2·xgb +0,00011 (4/5); bobot nested +0,00010; 4 model (E130a, xgb, E121, E146) +0,00027 (5/5). Ruang ensemble antar-GBDT kecil (±0,0003). E150_cat (CatBoost + ecm) belum dijalankan (±1,5–3 jam, tunggu laptop menganggur)** |
| E151 | Stacking level 2 (LR pada rank/logit, atau LightGBM dangkal + meta-fitur seperti `has_history`) | Playbook Grandmaster #5; juara Playground Apr 2025 memakai stack 3 level |
| E152 | Residual fitting: model kedua belajar dari residual model pertama | Juara 2 Home Credit (LightGBM pada residual NN) |

### E. Semi-supervised

| ID | Teknik | Catatan |
|---|---|---|
| E160 | Pseudo-labeling test (soft label, protokol per fold: label semu untuk fold k dari model yang tidak melihat fold k) | Playbook Grandmaster #6. Test 30k = +43% data. Risiko bocor jika protokol salah |

### F. Spesifik metrik KS

| ID | Teknik | Catatan |
|---|---|---|
| E170 | Pilih kandidat dengan mean-fold KS rata-rata 3 seed, selain AUC; analisis di mana titik KS maksimum berada (biasanya dekat kuantil target rate) | KS hanya bergantung pada satu ambang; dampaknya kecil, tapi mengurangi salah pilih |

### G. Infrastruktur dan proses

| ID | Teknik | Catatan |
|---|---|---|
| I1 | **Kaggle Notebook** untuk job berat: sumber pihak ketiga menyebut ~29–30 GB RAM, 4 CPU, GPU T4×2/P100 ~30 jam/minggu (cek angka resminya di pengaturan notebook) | Laptop 7,7 GB sudah sekali membuat pipeline dimatikan. Data lomba bisa langsung di-attach. Sekaligus latihan untuk notebook final |
| I2 | Pasang torch versi CPU di laptop untuk uji kecil NN | Diperlukan E122 dan E142 |
| I3 | **Protokol screening cepat** (bagian 6) sebelum CV penuh | Menghemat waktu: ide yang gagal screen tidak perlu CV penuh |

## 6. Protokol screening cepat (dipakai di bagian 3)

1. **Residual boosting screen** (~2 menit): LightGBM dengan `init_score = logit(OOF terbaik)`, hanya memakai fitur kandidat, 5 fold tetap, pohon kecil (7 daun), early stopping. Ukur delta mean-fold AUC dan jumlah fold yang naik.
2. **Residual terkalibrasi**: kalibrasi ulang OOF (logistik pada logit), lalu korelasi Pearson rank(fitur) dengan residual dan rata-rata residual per desil. Noise sd ≈ 1/√70.000 = 0,0038.
3. Lolos screen (delta ≥ +0,0005 atau |korelasi| > 3 sd) → CV penuh dengan 3 seed, lalu aturan terima biasa.

Keterbatasan: screen tidak nested (sedikit optimis) dan hanya menangkap sinyal yang belum dipakai model. Keputusan akhir tetap dari CV penuh.

Perbaikan 13 Okt: `screen.py` sebelumnya memakai Spearman (residual ikut di-rank). Rank residual pada label biner kembali mengikuti y, sehingga fitur yang kuat tapi redundan tampak sangat signifikan (E125: |z| 97 palsu, padahal residual boosting +0,00001 dan CV −0,00019). Sekarang memakai Pearson rank(fitur) vs residual mentah sesuai butir 2 (E125: |z| maks 2,6 → tidak lolos). Screen sebelum 13 Okt yang lolos hanya karena |z| perlu dicek ulang (E121 lolos lewat delta boosting, jadi tidak terpengaruh).

## 7. Yang sengaja tidak dilakukan

- **Data eksternal dan model pretrained** (termasuk TabPFN, yang dilatih pada data sintetis prior). Guidebook melarang "pretrained berbasis data luar"; TabPFN area abu-abu, jadi tanyakan panitia dulu. Default: tidak dipakai.
- **Menyalin notebook publik** (plagiarisme = diskualifikasi). Membaca ide di Discussion boleh, implementasi harus milik sendiri.
- **ID sebagai fitur**, dan **mengandalkan kebocoran** kalau ternyata ditemukan. Kebocoran harus dikonfirmasi/dilaporkan ke panitia, karena final mewajibkan notebook reproducible dan presentasi metodologi.
- **Memilih final berdasarkan public** (private = 45% nilai).

## 8. Antrean eksekusi (disepakati 12 Okt)

Urutan ini menggabungkan langkah lanjutan Fase 5, planning awal (Fase 2, 3, 5, 6, 7), dan katalog di bagian 5. **Eksekusi dimulai hanya saat user bilang "lanjut"**, berurutan dari nomor 1.

| No | Tanggal | Langkah | ID | Fase planning | Estimasi |
|---|---|---|---|---|---|
| 1 | 12–13 Okt | LightGBM tuned (B3 dan sel_b2), batas iterasi 5.000 → 15.000, 3 seed. Submit jika CV naik | E140a | Fase 5 | ±45–60 mnt |
| 2 | 12–13 Okt | Tambah seed LGB terbaik sampai 5 seed | E144 | Fase 5 (E93) | ±25 mnt |
| 3 | saat laptop tidak dipakai | CatBoost tuned dengan batas iterasi lebih tinggi, 3 seed | E140b | Fase 5 | ±2,5–4 jam |
| 4 | 13–14 Okt | Seleksi fitur ulang memakai parameter tuned (null importance pada 528 kolom), bandingkan dengan B3 pakai 3 seed | E130 | Fase 3 | ±2–3 jam |
| 5 | 14–16 Okt | Segmen **belanja dekat cutoff**: model level order e-commerce (nested) jadi fitur. Fitur agregat recency sudah jenuh (bagian 3 butir H); sinyal yang tersisa ada di level order (butir J) | E121 | Fase 2 | ±1 hari |
| 6 | 16–18 Okt | Segmen **tanpa riwayat** (30%, KS 0,466): analisis kasus salah, fitur hasil prediksi (skor biro yang unscored, income rekening), model khusus segmen | E120, E125, E128 (= E112) | Fase 2 + 7 | ±2 hari |
| 7 | 18–24 Okt | Jika waktu ada, urut potensi: model urutan GRU (Kaggle GPU), peluruhan waktu dan fitur gaya AmEx, DART, RealMLP/TabM, fitur otomatis, pseudo-label | E122 (= E111), E131, E132, E141, E142, E123, E124, E160 (= E110) | Fase 7 | sesuai waktu |
| 8 | 25 Okt | Tuning ulang singkat jika fitur banyak berubah | E146 | Fase 5 | ±3–4 jam |
| 9 | 26 Okt | **Feature freeze** | | | |
| 10 | 26–27 Okt | Ensemble pada OOF ≥ 3 seed (rank average, bobot teroptimasi, Caruana, stacking), cek stabilitas KS, pilih 2 submission final (best-CV + robust) | E100–E104 = E150–E152, E170 | Fase 6 | 1–2 hari |
| 11 | 27–30 Okt | Notebook final self-contained, dua kali fresh run, cek MD5; deadline 30 Okt 23.59 WIB | | Notebook | 3–4 hari |

Aturan di setiap langkah:

- Ide fitur baru di-screen dulu (bagian 6, ±2 menit). Hanya yang lolos yang masuk CV penuh.
- Keputusan memakai aturan terima: mean-fold AUC ≥ +0,0005 dan naik di ≥ 4/5 fold; keputusan tipis dikonfirmasi 3 seed.
- Job lebih dari 30 menit dijalankan lewat launcher terpisah (seperti `experiments/p5/run_p5.cmd`) atau di Kaggle Notebook, bukan background task Claude.
- Kuota submission (3/hari, reset 05.00): maksimal 1–2 per hari untuk memvalidasi perubahan besar, bukan untuk memilih bobot blend.

### Target antara (public, dengan asumsi offset +0,033 dan tanpa keberuntungan noise)

| Tanggal | Target public | Setara mean-fold AUC | Dari mana |
|---|---|---|---|
| 14 Okt | ≥ 0,555 | ≈ 0,842 | E140 + E121 + seed lebih banyak |
| 21 Okt | ≥ 0,565 | ≈ 0,848 | + E122, E125, E130, temuan E120 |
| 26 Okt (freeze) | ≥ 0,575 | ≈ 0,853 | + fitur otomatis, model baru |
| 30 Okt | 0,58–0,60, stretch 0,65 | ≥ 0,856 | + ensemble Fase 6 |

Kalau sampai 21 Okt mean-fold AUC belum lewat 0,845, prioritas digeser ke grup A yang lebih berisiko tapi berpotensi besar (E122 untuk tabel lain, E123, E126).

## 10. Evaluasi arah (16 Okt): kenapa CV naik tetapi public diam?

**Data.** 15 submission. Regresi public terhadap mean-fold AUC pada 9 submission pertama (E03 … E130a+E141): public = −0,888 + 1,71 × AUC, korelasi 0,953. Sisa (public − prediksi):

| Kelompok | Submission | Sisa |
|---|---|---|
| GBDT tunggal / blend sesama GBDT | E03, E31, E71b, E80, P5 single, E121, E130a+E141, E122c, E122c+E141 | −0,0016 … +0,0033 (rata-rata ≈ +0,001) |
| Blend lintas keluarga model | P4 blend (GBDT+CatBoost bobot), P5 Caruana, E122c+NN40, E122x1+NNseq50, E122c+NNseq50 | −0,0022 … −0,0060 (semua negatif) |
| NN saja | NNseq | −0,0069 |

**Temuan.**
1. **Kenaikan CV sejak E130a terlalu kecil untuk terlihat di public.** E130a → E122x1 = +0,0014 AUC ≈ +0,0024 KS public yang diharapkan, sedangkan sd selisih berpasangan di public 0,003–0,004. Jadi "stuck" sebagian besar adalah batas resolusi public, bukan berarti tidak ada kemajuan. Kenaikan awal (E03 → E121, +0,017 AUC) terlihat karena 10× lebih besar.
2. **Blend lintas keluarga model konsisten lebih rendah dari prediksi CV** (5 dari 5, termasuk P4 dan Caruana sebelum ada NN). Penjelasan: prediksi test tiap model sudah rata-rata 5 model fold sehingga antar-model lebih mirip di test daripada di OOF (Spearman E122c~TabM: OOF 0,9881 vs test 0,9904; E130a~E141 0,9972 vs 0,9985). Keragaman di OOF sebagian adalah noise model fold, jadi kenaikan blend di OOF terlalu optimis (pelajaran `seed_bag_check` berlaku juga lintas keluarga).
3. **NN memilih epoch di fold luar bukan penyebab utama**: versi jujur `_iv` (holdout internal) TabM 0,84476, RealMLP 0,84424 (≈ versi lama + fitur seqx). NN memang sekuat GBDT di CV, tetapi di public NN saja 0,55265 (−0,0047 vs E122c, z ≈ −1).
4. **Fitur GRU tetap sah**: uji label acak E122s dan E122xs (fitur dari label acak di CV) = acuan tanpa fitur (0,84305 vs 0,84314; 0,84452 vs 0,84454).
5. **Leaderboard lawan naik** sebagian karena efek seleksi (bagian 2b): tim yang mengirim 3×/hari dan memilih public terbaik mendapat bonus +0,01–0,016 yang tidak terbawa ke private.

**Sumber sinyal besar yang belum dipakai (E126).** Pinjaman lama yang disetujui: 159.323; yang jendela 12 bulan + 90 hari-nya sudah lengkap sebelum cutoff: **61.505 pinjaman, default (maxDPD ≥ 90) 17,9%** (43.090 milik customer train, 18.415 milik customer test). Hampir menggandakan jumlah contoh berlabel (70.000). Label pinjaman lama bukan target aplikasi sekarang, jadi model PD pinjaman lama tidak perlu nested. Syarat: fitur dihitung *as-of* `decision_datetime` pinjaman lama (laporan biro milik aplikasi lama, cicilan/aplikasi/rekening/e-commerce sebelum tanggal itu). Hati-hati: `days_past_due`/`max_days_past_due` di `repayment_history` adalah kondisi per cutoff aplikasi sekarang, jadi untuk cutoff lama harus dihitung ulang dari `due_date`, `last_payment_datetime`, `amount_paid`.

**Leaderboard 16 Okt** (dari user): kita #39, 0,55858 (±16 entri). #1 cupuu 0,57768 (20 entri), #2 0,57530 (15), #3 0,57067 (19), #4 0,57059 (12), #5 0,57052 (17), #6 0,57015 (8), #7 0,57005 (9), #8 0,56832 (16), #9 0,56747 (**4 entri**), #10 0,56742 (14). Tim dengan 4–8 entri di 0,567–0,570 menunjukkan efek seleksi tidak cukup menjelaskan semua gap. Peringkat #10–#39 berada dalam pita 0,009 (≈ 1,5 sd selisih berpasangan antartim), jadi urutan di dalam pita itu sebagian besar acak, tetapi gap ke #1–#2 (+0,017–0,019) kemungkinan sebagian nyata.

**Cek pipeline 16 Okt (semua lolos):** urutan baris `sample_submission`/`test_applications` tidak berkaitan dengan prediksi (Spearman 0,0005 / −0,007); target vs posisi baris, urutan `application_id`/`customer_id`, `decision_datetime`: AUC 0,496–0,501; adversarial validation 528 kolom AUC 0,5024, KS train-test per fitur maks 0,044 (`pa_fresh_loan_inst`, jarang terisi); semua kolom data mentah dirujuk di `src/features` kecuali ID. LGB parameter standar pada set E130a (leaves 31, ff 0,5, lr 0,03: 0,84076; leaves 127, ff 0,3: 0,83950, berhenti di ±400 iterasi) < LGB tuned 0,84291 (1 seed) → parameter tuned bukan penghambat. Kesimpulan: tidak ada kesalahan teknis yang terdeteksi; gap ada pada **informasi yang diekstrak**, dan menumpuk model pada informasi yang sama tidak menutupnya.

**E120 analisis kasus salah (16 Okt, `experiments/e120/`)**: OOF E122x1 (bag 3 seed). AUC per segmen merata 0,80–0,85 (tanpa cicilan 0,828, dengan 0,852; per tier bunga 0,814–0,842; per recency e-commerce 0,801–0,834); hanya 9,3% defaulter mendapat skor di bawah median. Timeline mentah 6 FN terparah, 6 FP terparah, 3 TP (`timelines.txt`): FN berprofil bersih (bunga 0,16–0,20, skor biro 690–810, maxDPD semua 0, saldo stabil), FP berprofil sangat buruk (maxDPD 90–257, skor biro 555–640, util 0,5–0,74) tetapi tidak default. Tidak terlihat pola data yang terlewat; kesalahan terbesar tampak sebagai keacakan label (tier terbaik pun default 5,5%). Implikasi: kenaikan harus datang dari estimasi risiko dasar yang lebih tepat (lebih banyak contoh berlabel → E126), bukan fitur pola baru.

**E126 pinjaman lama sebagai data berlabel (16 Okt, `e126.py`) — DITOLAK.** `--build`: 96 fitur *as-of* tanggal keputusan (syarat pinjaman, profil disesuaikan umur, laporan biro milik aplikasi itu, aplikasi lama sebelum t, cicilan dengan DPD dihitung ulang per t, rekening & e-commerce berjendela + `hist_avail_d`) untuk 100.000 aplikasi sekarang + **81.745 pinjaman lama** (jendela 365 hari lengkap; label = cicilan jatuh tempo ≤ 275 hari setelah cair dengan maxDPD ≥ 90; default 14,4%). Cek: tidak ada laporan biro tersedia ≥ t; fitur as-of aplikasi sekarang identik dengan pipeline lama (korelasi 1,0); riwayat pinjaman lama tipis (e-commerce terisi 31%, cicilan 33%). (b) Model PD pinjaman lama: AUC CV (GroupKFold customer) 0,704; skor ke aplikasi sekarang AUC 0,677; sebagai fitur di atas E122x1 (seed 42) **0,84468 = 0,84468** (nol). (c) Baris latih tambahan pada model 96 fitur as-of: 0,82193 → 0,82235 (bobot 0,5; +0,0004, 3/5) → setara noise.

**E170 model zona KS (16 Okt) — DITOLAK.** Titik KS OOF ada di 30,7% teratas (TPR 0,761, FPR 0,236; per fold 27,8–33,3%). Model kedua dilatih hanya pada zona persentil risiko 15–50% (528 kolom + ecm + skor dasar) untuk mengurutkan ulang zona itu: AUC dalam zona 0,62 vs model dasar 0,63; KS +0,0010 (w 0,3, 3/5) / −0,0006 (w 0,5).

**Kesimpulan 16 Okt:** tiga arah (E120, E126, E170) tidak menemukan informasi baru; model kita tampak dekat dengan batas informasi data untuk metode yang sudah dicoba. **E150c (CatBoost tuned, set E122x1, 1 seed, 56 mnt): 0,84222 vs LGB 0,84468; Spearman 0,9795; blend dengan LGB bag 3 seed w 0,2 +0,00016 (4/5) → ditolak.** Sisa opsi: refit 100% data (E145; tidak terukur CV tetapi hampir pasti menambah sedikit di test), pencarian fitur otomatis (E123 DFS / E124 OpenFE) sebagai upaya penemuan terakhir, lalu fokus pada notebook final. **Public 16 Okt**: E145 refit E122x1 0,55535; **E122x1+E141 rank50 0,55951 (terbaik)**; GBDT3 (E122x1+E141+Cat) 0,55593. Pola: blend LGB+DART 50:50 tertinggi di public 3 dari 3 kali (E130a+E141 0,55858, E122c+E141 0,55858, E122x1+E141 0,55951) padahal CV-nya ≈ LGB saja; simulasi selisih berpasangan: E122c→+E141 z +0,3–0,4, E122x1(refit)→+E141 z +1,5–1,9 → bukti sedang, arah konsisten. Kandidat final #1 = E122x1+E141. Lanjutan: DART pada set E122x1 (E141x). **Rencana berikutnya (disepakati 16 Okt malam):** (1) jalankan `experiments/run_e141x.cmd` (DART 5000 ronde × 3 seed pada set E122x1; `e141.py --use <npz+...>` sudah ditambahkan, uji 10 ronde lolos; ±1,5–2 jam) → (2) submit tes E122x1+E141x rank50 dan DART E141x saja → (3) kandidat final hanya keluarga LightGBM (GBDT + DART), tanpa NN/CatBoost → (4) notebook final + uji reproduksi; pencarian fitur otomatis (E123/E124) maksimal 1 hari bila waktu ada. **E141x selesai 17 Okt** (`experiments/run_e141x.cmd`, ±2 jam): DART pada set E122x1, seed 42/43/44 = 0,84501/0,84473/0,84471, **bag 0,84516, KS 0,5244** (model tunggal CV terbaik; +0,00164 vs E141 5/5 P 1,00; +0,00020 vs E122x1 3/5 P 0,85; Spearman ke E122x1 OOF 0,9971, test 0,9983). Blend E122x1+E141x rank50 0,84526. Submission tes: `E122x1_E141x_rank50_auc0.84526.csv` (MD5 c18002e0…), `E141x_single_auc0.84516.csv` (MD5 497e0c72…), menunggu public. **Public 17 Okt**: E122x1+E141x rank50 0,55646, E141x saja 0,55573 (keduanya < E122x1+E141 0,55951). DART dengan fitur urutan tidak menaikkan public; pasangan public terbaik justru yang komponennya paling berbeda di test (Spearman E122x1~E141 0,9938 vs E122x1~E141x 0,9983). Semua varian GBDT kita di public 0,5554–0,5595 (≈ ±1,5 sd selisih berpasangan) → public tidak bisa lagi memilih di antara model yang hampir identik. Karena 45% nilai = public dari submission terpilih, kandidat final #1 = **E122x1+E141 (public 0,55951)**. Keputusan berikutnya: komposisi notebook final (dengan/ tanpa fitur GRU dari GPU).

**Arah baru (usulan).**
1. Hentikan eksperimen mikro (blend, varian NN, seed, bobot): kenaikannya < resolusi public dan terlalu optimis di OOF.
2. **E126 dulu**: (a) builder fitur *as-of* untuk pinjaman lama, (b) model PD pinjaman lama → skor untuk aplikasi sekarang sebagai fitur, (c) bila berhasil, pinjaman lama sebagai baris latih tambahan.
3. E120 analisis kasus salah (FN/FP terbesar) secara paralel untuk mencari pola yang terlewat.
4. Kebijakan pemilihan: submission utama dominan GBDT; blend lintas keluarga hanya jika divalidasi dengan cara yang meniru test (holdout terpisah dengan prediksi rata-rata model fold).
5. Rencanakan notebook final sejak sekarang: komponen GPU (GRU, NN) sulit direproduksi identik; tiap komponen yang dipakai harus deterministik atau dihitung di CPU.

## 9. Sumber

Solusi kompetisi kredit:
- Home Credit Default Risk, juara 2 (slide): https://speakerdeck.com/hoxomaxwell/home-credit-default-risk-2nd-place-solutions
- Home Credit Default Risk, juara 1 (Kaggle Discussion, isinya tidak bisa diambil otomatis; ringkasan sekunder: "feature engineering lebih berguna daripada tuning dan stacking"): https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821
- Home Credit, tim deepsense.ai (peringkat 5): https://deepsense.ai/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/
- AmEx Default Prediction, juara 1 (Kaggle Discussion): https://www.kaggle.com/competitions/amex-default-prediction/discussion/348111
- AmEx, solusi top-10 RNN + XGBoost (NVIDIA): https://developer.nvidia.com/blog/predicting-credit-defaults-using-time-series-models-with-recursive-neural-networks-and-xgboost/

Teknik umum kompetisi tabular:
- Kaggle Grandmasters Playbook, 7 teknik: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data
- Stacking juara 1 Playground (cuML): https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/
- DAE swap noise, juara 1 Porto Seguro (diskusi fast.ai): https://forums.fast.ai/t/porto-seguro-winning-solution-representation-learning/8499?page=2
- OpenFE (ICML 2023): https://proceedings.mlr.press/v202/zhang23ay.html, kode: https://github.com/IIIS-Li-Group/OpenFE
- Featuretools Deep Feature Synthesis: https://featuretools.alteryx.com/en/v1.5.0/getting_started/afe.html
- RealMLP (NeurIPS 2024): https://arxiv.org/abs/2407.04491, pytabkit: https://pytabkit.readthedocs.io/
- TabArena (2025): https://arxiv.org/abs/2506.16791

Riset credit scoring:
- Berg, Burg, Gombović, Puri, "On the Rise of FinTechs: Credit Scoring Using Digital Footprints": https://www.fdic.gov/analysis/cfr/working-papers/2018/cfr-wp2018-04.pdf
- FinRegLab, cash-flow underwriting (saldo, setoran, volatilitas, saldo rendah paling prediktif): https://finreglab.org/wp-content/uploads/2025/06/FinRegLab_06-03-2025_Sharpening-the-Focus.pdf

Infrastruktur (sumber pihak ketiga, verifikasi di Kaggle):
- Ringkasan GPU gratis termasuk Kaggle: https://research.aimultiple.com/free-cloud-gpu

## 11. Eksperimen 17 Okt+

Dasar: `PROMPT_UTAMA.md` (eksplorasi analis, laporan kerja `experiments/explore_17okt/report.md`). ID baru mulai **E200**. Aturan keputusan baru (berlaku 17 Okt, `compare_oof.py` default): OOF rata-rata ≥ 3 seed; bootstrap berpasangan berstrata per fold 1000× (seed 42); **Diterima** P(Δ>0) ≥ 0,95 dan Δ ≥ +0,0003; **Kandidat** P 0,80–0,95 atau Δ < +0,0003; **Ditolak** P < 0,80. Public LB = 30% test (±9.000 baris), private = 70%. Final: 2 submission, keduanya harus dihasilkan notebook (MD5 cocok); mesin panitia untuk menjalankan ulang tidak diketahui → kandidat CPU-deterministik diutamakan.

### 11.1 Penilaian ulang eksperimen lama dengan aturan baru (18 Okt, `experiments/explore_17okt/s24_reevaluate_old.py`)

Syarat: status lama kandidat/ditolak, Δ > 0 terhadap acuan aslinya, OOF ≥ 3 seed (seed 42/43/44) di kedua sisi. Hasil lengkap: `experiments/explore_17okt/reevaluasi_aturan_baru.csv`.

| Kandidat (vs acuan asli) | Status lama | Δ mean-fold AUC | CI 95% | P(Δ>0) | Fold naik | Status baru |
|---|---|---|---|---|---|---|
| E130b (vs E130a) | ditolak | +0,00036 | [+0,00007; +0,00065] | 0,996 | 3/5 | **Diterima** |
| E141 (vs E130a) | kandidat | +0,00038 | [−0,00000; +0,00078] | 0,973 | 5/5 | **Diterima** |
| E122r (vs E122) | kandidat | +0,00019 | [−0,00003; +0,00040] | 0,954 | 4/5 | Kandidat (Δ < 0,0003) |
| E122m (vs E122) | kandidat | +0,00023 | [−0,00002; +0,00048] | 0,962 | 4/5 | Kandidat (Δ < 0,0003) |
| E122c (vs E122) | kandidat | +0,00044 | [+0,00017; +0,00071] | 0,999 | 5/5 | **Diterima** |
| E122x1 (vs E122c) | kandidat | +0,00041 | [+0,00009; +0,00072] | 0,992 | 4/5 | **Diterima** |
| E122x2 (vs E122c) | kandidat | +0,00033 | [+0,00004; +0,00063] | 0,985 | 5/5 | **Diterima** |
| E122x3 (vs E122c) | kandidat | +0,00041 | [+0,00001; +0,00084] | 0,979 | 5/5 | **Diterima** |
| E141x (vs E122x1) | kandidat | +0,00020 | [−0,00015; +0,00056] | 0,857 | 3/5 | Kandidat |
| E140a B3 r15k (vs T_lgb_b3) | ditolak (dipakai default) | +0,00009 | [+0,00005; +0,00014] | 1,000 | 3/5 | Kandidat (Δ < 0,0003) |
| E140a sel_b2 r15k (vs T_lgb_selb2) | ditolak | +0,00041 | [+0,00030; +0,00054] | 1,000 | 4/5 | **Diterima** |

### 11.2 Eksperimen baru (E200+)

| ID | Hipotesis | Perubahan | Hasil (mean-fold AUC, 3 seed) | Status |
|---|---|---|---|---|
| E200 | H01: rantai bayar bulanan level event (pembayaran hanya di ulang bulan due, FIFO; laju lewat-bayar stabil per customer) | set E122x1 + 18 kolom `rp_ev_*` (`src/features/events.py`, `features/rp_events_v1.parquet`, `e121.py --extra`) | E122x1 0,84495 → **0,84659**, Δ **+0,00163**, CI 95% [+0,00117; +0,00212], P 1,000, naik 5/5 fold; KS mean-fold 0,5266 → 0,5296. Audit: kode 9/9, AUDIT `rp_ev_miss_post` (AUC 0,807) dicek, adversarial 0,4993, KS train-test ≤ 0,010. Screen +0,00059 (5/5) | **Diterima** (diusulkan jadi acuan baru, menunggu konfirmasi user) |
| E201 | H02: model bantu PD dilatih pada 81.702 pinjaman lama (label DPD ≥ 90 dalam 12 bulan), bukti leave-one-loan-out (event pinjaman lain di jendela 12 bulan dikecualikan, hasil pinjaman tumpang tindih tidak dipakai, biro as-of, sifat e-commerce/rekening di luar zona ramp); model bantu AUC GroupKFold 0,797; fitur AUC univariat 0,799 | set E200 + `x_auxpd_logit/pd` (`src/features/auxpd.py`, `x_auxpd_v1`) | E200 0,84659 → 0,84627, Δ **−0,00032**, CI [−0,00075; +0,00013], P 0,086, naik 0/5. Audit lolos (adversarial 0,498; label acak: AUC model bantu 0,506, screen +0,00005). Screen +0,00022 (5/5) | **Ditolak** — informasi tumpang tindih dengan rantai bayar E200; label tambahan tidak memperbaiki pemetaan riwayat → risiko |
| E202 | H05: 10-fold bersarang di 5 fold (tiap fold dibelah 2 bertingkat; model dilatih 90% data; kurva belajar memprediksi +0,0005) | set E200, `e121.py --subfolds 2` (10 model/seed); mean-fold AUC dihitung pada 5 fold asli | E200 0,84659 → **0,84710**, Δ **+0,00051**, CI [+0,00022; +0,00079], P 0,999, naik 4/5; KS 0,5296 → 0,5306 | **Diterima** (perubahan skema latih: dipakai di model akhir; eksperimen fitur tetap dibandingkan 5-fold vs E200 supaya cepat) |
| E203 | H12 pendahuluan: sensitivitas parameter LGB di set E200 (`e203_sens.py`, lr 0,05, 1 seed, 5 fold) | 8 konfigurasi: tuned, feature_fraction 0,08/0,20/0,30, depth 5, depth 6 + 63 daun, tanpa extra_trees, min_child 100 | tuned **0,84594** tertinggi; variasi 0,84526–0,84578 (dalam noise 1 seed); tanpa extra_trees 0,84457 (−0,0014) | Parameter tuned (Fase 5, set 176 kolom) tetap optimal di 559 kolom → Optuna 5 fold ulang (H12) tidak diprioritaskan |
| E204 | H21: rasio inflow non-gaji rekening terhadap income deklarasi (5 kolom `ab_nonin_*`, `features/cand_h21_v1.parquet`; audit Lapis 2 bersih, KS train-test ≤ 0,009) | set E200 + H21, 5-fold, 3 seed | E200 0,84659 → 0,84677, Δ +0,00019, CI [−0,00011; +0,00050], P 0,884, naik 4/5; KS 0,5296 → 0,5293 | **Kandidat** (OOF/preds disimpan; builder perlu dipindah ke `src/features/` bila dipakai PROMPT 3) |
| E205 | Set E200 versi **CPU saja** (528 + ecm + rp_ev, tanpa fitur GRU dari GPU) — acuan kandidat A PROMPT 3 | 5-fold, 3 seed | **0,84517**, KS 0,5263 (mean-fold); vs E130a +0,00203 (CI [+0,00150; +0,00261], P 1,000); vs E122x1 +0,00022 (P 0,75, Spearman 0,9891); fitur GPU di atasnya (E205 → E200) +0,00141 (P 1,000) | **Diterima** vs E130a; kandidat A setara model GPU terbaik lama. rp_ev menyumbang lebih besar tanpa GRU (+0,0020 vs +0,0016) → sebagian tumpang tindih dengan seqx_rp |
| — | Uji kapasitas residual (`s27`): boosting di atas logit OOF E200 dengan semua kolom (termasuk fitur nested versi OOF), depth 3 / 6 / 8 (255 daun) | 5 fold | berhenti di iterasi 1 di semua kapasitas; Δ −0,00002 / −0,00005 / −0,00006, naik 0–1/5 | Tidak ada interaksi tersisa di fitur yang ada → H07 versi "lebih dalam" dan tuning kapasitas tidak berguna |
| — | Kurva belajar set E200 (`s29`, lr 0,05, 1 seed) | 14k / 28k / 42k / 56k baris latih | 0,84058 / 0,84293 / 0,84520 / 0,84594; Δ 50% → 100% +0,0030 (sd antar fold 0,0013); asimtot 0,847–0,857 tergantung eksponen (E130a: 0,844–0,856) | Masih lapar data → refit 100% (PROMPT 3 Langkah 8) memenuhi syarat; target 0,857 hanya tercapai di asimtot (data tak hingga) |
| — | H07 pendahuluan (`s31`): sub-model LightGBM per sumber (rp, biro, aplikasi lama, rekening, e-commerce, aplikasi+profil) + stacking logistik lintas fold di atas logit E202 | 5 fold | sumber sendiri 0,62–0,75; \|z\| residual ≤ 0,96; stacking 0,84710 → 0,84697 (Δ −0,00013, 1/5) | **Ditolak** (model utama sudah memadukan bukti antarsumber) |
| E207 | LGB DART 5000 ronde (setting E141) pada set CPU E205 (528 + ecm + rp_ev) | 5-fold, 3 seed | bag **0,84536** vs E205 0,84517: Δ +0,00018, CI [−0,00019; +0,00057], P 0,834; Spearman 0,9974 | **Kandidat**; blend E205 + E207 rank 50:50 = 0,84545 (KS OOF 0,5242) → submission kandidat A `E205_E207_rank50_auc0.84545.csv` |
| — | Putaran masukan user/mentor 18 Okt (laporan eksplorasi bagian 9.x dan 10): kebocoran urutan file test, anomali 5 tabel + e-commerce, fitur otomatis 10.760 kandidat, KNN, waktu, customer ganda, kalibrasi per segmen (KS), pola missing/batas/frekuensi, CAAFE 2 putaran (39 fitur), SHAP | – | semua negatif; efek bulan keputusan nyata (p ≈ 0,0004) tetapi sudah ditangkap `app_month` | Ditolak; tuas tersisa = variansi/ensemble/portofolio public |
| — | H08 (peluang default pinjaman berjalan dari tabel hazard empiris), H09 (status bulan kalender terakhir lintas pinjaman) | `features/cand_h08h09_v1.parquet` | screen di atas E200: Δ −0,00001, \|z\| ≤ 1,4 | Ditolak di screen (bukti lemah, variasi H01) |

Tidak dinilai ulang: Fase 2 (E11–E61, hanya 1 seed), Fase 3–4 (E71b/c, E80–E83, E87: seed 42 tersimpan sebagai file tanpa akhiran; sudah tergantikan), E126a/E150c (1 seed), E141b (2 seed), E128 (OOF gabungan tidak tersimpan), eksperimen dengan Δ ≤ 0 (E146, E125, E121b/c/d, E122rr, E150_xgb sebagai model tunggal). **Acuan tidak berubah: E122x1** (mean-fold AUC 0,84495, diterima ulang). E130b diterima di set lama (tanpa fitur urutan); efek seleksi yang sama di set E122x1 belum diuji (H13).
