# PROMPT 3 — Menggabungkan kenaikan-kenaikan kecil dengan benar

> **Cara pakai:** buka sesi Claude Code baru, lalu jalankan setelah PROMPT UTAMA menghasilkan temuan berstatus diterima atau kandidat, paling lambat 22–25 Okt, sebelum feature freeze 26 Okt. Langkah 1 (inventaris) sudah bisa dijalankan kapan saja.
> **Model:** Opus medium, dengan review akhir oleh Opus high.

---

Tujuan sesi ini adalah menggabungkan semua komponen yang pernah menaikkan CV, meskipun kecil, menjadi satu model atau ensemble final, lalu **mengukur kenaikan gabungannya secara jujur**. Gunakan Bahasa Indonesia.

## Prinsip (baca dulu, ini alasan prosedurnya)

1. **Kenaikan kecil tidak menjumlah begitu saja.** Komponen yang mengambil informasi yang sama saling tumpang tindih. Contoh: E141 naik +0,00038 di atas E130a, tetapi ±0 di dalam blend E122c+NN.
2. **Kutukan pemenang.** Dari puluhan eksperimen di 5 fold yang sama, sebagian "kenaikan" sekitar ±0,0005 hanyalah noise (sd antar seed ±0,0006). Menjumlahkan kenaikan yang sudah terpilih akan melebih-lebihkan hasil.
3. **Karena itu,** semua komponen digabung dalam SATU model atau ensemble, diukur sekaligus dengan seed baru, lalu dilakukan ablation untuk membuang komponen yang tidak menyumbang di dalam gabungan.

Aturan umum yang berlaku:

- aturan keputusan berbasis bootstrap, sama dengan PROMPT UTAMA: diterima jika P(Δ > 0) ≥ 0,95 dan Δ ≥ +0,0003; kandidat jika P 0,80–0,95; ditolak jika P < 0,80; jumlah fold yang naik hanya dicatat,
- ensemble hanya dinilai dengan OOF rata-rata ≥ 3 seed,
- job > 30 menit dijalankan lewat launcher `.cmd`, dan job > 3 jam tidak dijalankan tanpa izin,
- kuota submission 3 per hari dipakai untuk kandidat kuat (aturan pemilihan di Langkah 7), selalu lewat `verify_submission.py --log`, dengan konfigurasi lengkap dicatat supaya notebook bisa mereproduksinya.

## Langkah 1 — Inventaris

Dari `experiments/log.csv`, CLAUDE.md, dan file Riset (termasuk bagian 11 "Eksperimen 17 Okt+" hasil PROMPT UTAMA), buat tabel semua komponen berstatus diterima atau kandidat dengan kolom:

komponen | jenis (fitur / model / pelatihan / seleksi) | Δ solo | acuan saat diukur | jumlah seed | fold naik | file (parquet/npz/OOF) | tag repro (CPU-deterministik / GPU / butuh cek)

Tabel minimal mencakup:

- set 528 kolom (E130a),
- ecm (E121),
- seq_gru/bigru/tf persentil (E122c),
- seqx_rp/ab/pa (E122x1) dan seqx_all (E122x2),
- DART (E141), XGB tuned (E150_xgb), dan E146,
- batas iterasi 15000 (E140a) dan 5 seed (E144),
- NN RealMLP/TabM versi `_iv`,
- semua komponen diterima atau kandidat dari PROMPT UTAMA (ID E200 ke atas), termasuk eksperimen lama yang berubah status setelah aturan keputusan baru diterapkan.

## Langkah 2 — Set fitur gabungan F*

- F* = set fitur E122x1 ditambah semua grup fitur berstatus diterima atau kandidat dari PROMPT UTAMA.
- Latih LGB GBDT tuned pada F* dengan 3 seed. Laporkan hasilnya dibanding E122x1.

## Langkah 3 — Ablation di dalam gabungan

1. Lakukan leave-one-group-out pada F*: buang setiap grup kandidat (ecm, seq_*, seqx_*, dan setiap grup baru) satu per satu, lalu latih LGB 1 seed sebagai screen.
2. Grup yang hasil screen-nya tidak jelas (skor turun < 0,0005 saat dibuang) dicek ulang dengan 3 seed dan bootstrap.
3. Pertahankan grup hanya jika membuangnya menurunkan skor dengan P(turun) ≥ 0,80 (bootstrap, OOF 3 seed). Grup lain dibuang. Hasilnya disebut **F\*\***.
4. Buat juga versi **F\*\*-CPU**: F\*\* tanpa fitur turunan GPU (seq_*, seqx_*). Versi ini menjadi dasar kandidat yang reproducible.

## Langkah 4 — Model-model GBDT di atas F** (kombinasi yang belum pernah dicoba)

Jalankan setiap model dengan 3 seed, dan simpan OOF serta preds per seed:

- LGB GBDT tuned (lr final, iterasi ≤ 15000),
- LGB DART 5000 ronde dengan setting E141. Perkiraan ±1,6 jam, jadi jalankan lewat launcher `.cmd`,
- XGB tuned dengan setting E150_xgb,
- CatBoost tuned (E150_cat), hanya jika estimasi waktunya ≤ 3 jam. Kalau lebih, lewati dan catat,
- varian objektif ranking, jika pernah diuji dan berstatus kandidat.

Ulangi untuk F\*\*-CPU, minimal LGB GBDT dan DART.

## Langkah 5 — Ensemble sesama GBDT

- Gunakan rank average dengan bobot yang dicari secara **nested**: cari bobot di 4 fold, uji di fold ke-5, dan pakai OOF rata-rata ≥ 3 seed.
- Bandingkan empat pilihan: model tunggal terbaik, rata-rata sederhana, bobot nested, dan Caruana nested. Jika selisihnya < 0,0003, pilih yang paling sederhana.
- NN hanya diuji terpisah, dengan bobot dibatasi ≤ 20%. Alasannya ada di Riset bagian 10: blend lintas keluarga model konsisten berada di bawah prediksi CV di public. Untuk kandidat utama, bobot NN default = 0.

## Langkah 6 — Konfirmasi yang meniru test

- **Seed baru:** jalankan konfigurasi final dan E122x1 sama-sama dengan seed baru (101, 102, 103). Δ harus tetap positif.
- **Bootstrap:** 1000× resample untuk Δ mean-fold AUC, laporkan CI 95%.
- **Fold baru**, khusus untuk keputusan level model:
  - Cek ulang keputusan "DART vs GBDT" dan bobot blend dengan `StratifiedKFold(5, shuffle=True, random_state=2026)`, memakai set fitur tanpa fitur nested (528 kolom + fitur baru non-target).
  - Fitur nested (ecm, seq, seqx, knn, dan prediksi mode 2 E126) TIDAK boleh dipakai dengan fold baru tanpa dibangun ulang, karena akan bocor. Jelaskan ini di laporan.
- **Simulasi public:** dari OOF, ambil 1000 subset acak berukuran 9.000 dan 15.000 baris. Laporkan sd KS kandidat, serta peluang kandidat mengungguli E122x1 dan E130a+E141 di subset tersebut.

## Langkah 7 — Kandidat final dan reproduksibilitas

- **Kandidat A (CPU-deterministik):** F\*\*-CPU dengan model atau ensemble terbaiknya.
- **Kandidat B (terbaik):** F\*\* dengan ensemble terbaiknya. Kandidat ini hanya dipakai di final jika salah satu syarat berikut terpenuhi:
  - semua komponen GPU-nya terbukti bisa dihasilkan identik oleh notebook (dua kali run, MD5 sama), atau
  - panitia mengonfirmasi bahwa notebook tidak akan dijalankan ulang.
- **Aturan pemilihan final.** Nilai penyisihan = 45% public + 45% private, jadi untuk setiap kandidat yang sudah di-submit hitung:

  ```
  nilai_harapan = 0,45 × public_terlihat + 0,45 × perkiraan_private
  ```

  - perkiraan_private dari CV, memakai regresi public–CV di Riset bagian 10 yang diperbarui dengan semua submission terbaru. Masukkan koreksi untuk jenis kandidat yang konsisten di bawah prediksi regresi (misalnya blend lintas keluarga model).
  - Pilih kandidat dengan nilai_harapan tertinggi, asalkan bisa direproduksi notebook.
  - Kalau panitia mengonfirmasi bahwa yang dinilai adalah skor public terbaik dari semua submission, pilih final hanya berdasarkan perkiraan private.
  - Laporkan tabel semua submission: file | MD5 | mean-fold AUC | KS OOF | public | perkiraan private | nilai_harapan | repro.

## Langkah 8 — Opsional minggu terakhir (setelah freeze, sebelum notebook)

### E145 refit 100% data

- Latih ulang pada 100% data dengan iterasi tetap = rata-rata best_iter × 1,15.
- Kerjakan hanya jika learning curve menunjukkan kurva masih naik (Δ 50% → 100% ≥ 0,0015). Kalau learning curve belum pernah diukur, ukur dulu.
- Validasi aturan iterasinya di CV: di setiap fold, bandingkan model dengan iterasi tetap sesuai aturan (tanpa early stopping) dengan early stopping biasa. Selisihnya harus ≤ 0,0002.

### E160 pseudo-label

- Untuk fold luar k, label lunak test HANYA boleh berasal dari model fold k, yaitu model yang tidak melihat label fold k.
- Baris test diduplikasi menjadi label 1 dengan bobot p dan label 0 dengan bobot 1−p. Alternatifnya, pakai hanya baris yang yakin (p < 0,03 atau p > 0,6).
- Fitur nested untuk baris test memakai versi fold k.
- Validasi di fold k, lalu terapkan aturan terima biasa.

## Laporan akhir

1. Tabel per komponen: komponen | Δ solo | Δ di dalam gabungan (ablation) | dipakai? | repro.
2. Perjalanan skor: E122x1 → F* → F\*\* → ensemble final (kandidat A dan B). Untuk setiap tahap, laporkan mean-fold AUC, KS, fold naik, CI bootstrap, dan prediksi public.
3. Rekomendasi 2 kandidat final beserta risikonya.
