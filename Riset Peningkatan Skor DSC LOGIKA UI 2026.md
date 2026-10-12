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
| E121 | **Model level baris e-commerce**: LightGBM di level order (fitur intrinsik order saja: waktu relatif ke keputusan, nominal, kuantitas, porsi kategori, metode bayar, status, jam, hari), label target customer, prediksi OOF **nested** (untuk fold luar k, model baris tidak boleh melihat label fold k), diagregasi (mean, max, std, 3/10 order terakhir, 30 hari terakhir) jadi fitur. Variasi: level item, dan tambah pohon/iterasi | Screen 12 Okt: **+0,0020 AUC, 5/5 fold**. Tabel lain ≈ 0, jadi tidak perlu | +0,001–0,002 AUC (nyata) | 1 hari | **1** |
| E122 | **Model urutan (GRU/1D-CNN)** untuk e-commerce dulu (deret order/item 180 hari terakhir), baru tabel lain jika berhasil. OOF prediksi/embedding → fitur GBDT | Juara 2 Home Credit (Conv1D/LSTM pada tabel anak); pemenang AmEx memakai GRU di ensemble; tim top-10 AmEx RNN + XGBoost +0,003. Screen E121 menunjukkan sinyal e-commerce ada di level urutan/baris | Sedang | 2–3 hari, perlu torch (sebaiknya GPU Kaggle) | **2** |
| E123 | **Featuretools DFS** dengan `cutoff_time` per aplikasi: eksplorasi otomatis agregat bertingkat antartabel | Deep Feature Synthesis, dirancang untuk data relasional bertimestamp | Sedang | 1 hari + RAM besar | 3 |
| E124 | **OpenFE** pada matriks level aplikasi: pencarian otomatis interaksi aritmetika | ICML 2023; fitur OpenFE + XGBoost sederhana mengalahkan 99,3% tim IEEE-CIS Fraud | Kecil–sedang | 0,5 hari | 3 |
| E125 | **Fitur hasil prediksi** ("predicted feature"): perluas `x_rate_pred` (AUC 0,693 > bunga asli 0,646). Prediksi skor biro untuk yang unscored, income rekening untuk yang tanpa data rekening, dan approval lender dari pola keputusan lama | Juara 2 Home Credit: fitur terpenting adalah tenor pinjaman yang diprediksi model | Kecil–sedang | 0,5–1 hari | 2 |
| E126 | **Augmentasi dari pinjaman lama**: model PD level pinjaman dilatih pada pinjaman lama yang sudah ada hasilnya (pernah DPD ≥ 90), fitur dihitung *as-of* keputusan pinjaman lama (laporan biro previous app tersedia), lalu skor untuk aplikasi saat ini | Pinjaman lama = contoh tambahan dari proses yang sama | Sedang | 2 hari | 3 |
| E127 | **KNN target mean** (OOF, nested) di ruang fitur terkuat yang distandardisasi: rata-rata target 100–500 tetangga terdekat | Ide "neighbors target mean" dari forum Home Credit (detail perlu diverifikasi) | Kecil–sedang | 0,5 hari | 3 |
| E128 | **Model khusus segmen tanpa riwayat** (30% data, KS 0,466) + fitur khusus segmen | Planning E112 | Kecil | 0,5 hari | 4 |

### B. Penyempurnaan fitur

| ID | Teknik | Catatan | Dampak | Prioritas |
|---|---|---|---|---|
| E130 | Seleksi fitur ulang memakai **parameter tuned**: null importance ulang + uji 528 kolom penuh | Parameter tuned jauh berbeda (depth 4, feature_fraction 0,13) dari saat seleksi Fase 3 | Kecil–sedang | **2** |
| E131 | Bobot peluruhan waktu (half-life 30/90/180 hari) untuk cicilan, rekening, order | Perilaku terbaru biasanya lebih prediktif | Kecil | 3 |
| E132 | Fitur gaya AmEx untuk deret bulanan: `last`, `last − mean`, `last / mean`, slope, untuk rekening dan cicilan per bulan | Pola umum solusi AmEx | Kecil | 3 |
| E133 | ~~Recency order ke-k, jendela pendek, porsi COD/kartu kredit~~ dan ~~utang/inquiry eksternal~~ | **Sudah diuji 12 Okt: tidak menambah** (bagian 3 butir H, I) | — | ditolak |
| E134 | Panel bulanan semua tabel pada grid waktu yang sama (bulan −1..−12 relatif cutoff) | Fondasi untuk E122 dan E132 | Tidak langsung | bersama E122 |

### C. Model

| ID | Teknik | Dasar | Dampak | Prioritas |
|---|---|---|---|---|
| E140 | **Batas iterasi 15.000** untuk LGB dan CatBoost (banyak fold mentok di 5.000) | Hasil Fase 5 | Kecil, hampir pasti | **1** |
| E141 | LightGBM **DART** | Juara 2 Home Credit: DART + regularisasi ≈ +0,001 CV dan private | Kecil | 3 |
| E142 | NN tabular modern: **RealMLP / TabM** (pytabkit) | RealMLP NeurIPS 2024; TabArena 2025 (NN mengejar GBDT dengan ensemble) | Kecil sendiri, berguna untuk diversitas | 3 |
| E143 | Fitur DAE swap-noise (gaya Porto Seguro) | Juara 2 Home Credit: public naik, private datar | Kecil, berisiko | 5 |
| E144 | **Seed bagging 5–10 seed** untuk model final | Playbook Grandmaster #7; 3 seed sudah memberi +0,002 AUC | Kecil, hampir pasti | **2** |
| E145 | **Refit 100% data** dengan iterasi tetap (≈ rata-rata best_iter × 1,1) | Playbook Grandmaster #7 | Kecil | 3 |
| E146 | Tuning ulang singkat setelah fitur berubah; XGBoost sampai 60 trial | Parameter optimal bergeser saat fitur berubah | Kecil | 25 Okt |

### D. Ensemble dan pasca-proses (Fase 6, setelah feature freeze)

| ID | Teknik | Catatan |
|---|---|---|
| E150 | Rata-rata rank / hill climbing (Caruana) memakai OOF ≥ 3 seed | Sudah ada pratinjau; blend dua kali kalah di public, jadi wajib cek stabilitas KS |
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
