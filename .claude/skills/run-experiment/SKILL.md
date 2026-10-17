---
name: run-experiment
description: Menjalankan dan memutuskan satu eksperimen DSC LOGIKA UI 2026 (prediksi default kredit, metrik KS) sesuai protokol tim. Gunakan skill ini setiap kali user minta menjalankan, menguji, atau menilai apa pun yang bisa mengubah skor - fitur baru, grup fitur, model, parameter, iterasi, seed, blend/ensemble, seleksi fitur - atau bilang "lanjut", "coba E2xx", "screen fitur ini", "diterima atau tidak?", "bandingkan dengan acuan", bahkan jika kata "eksperimen" tidak disebut.
---

# run-experiment

Satu eksperimen = satu perubahan, dinilai berpasangan terhadap acuan terbaik pada fold yang sama, lalu dicatat. Aturan keputusan yang berlaku sejak 17 Okt 2026 ada di bagian 3 (menggantikan aturan lama "+0,0005 dan ≥ 4/5 fold"). Kalau dokumen proyek dan skill ini berbeda, beri tahu user perbedaannya.

Script ada di `.claude/skills/run-experiment/scripts/`. Jalankan dari root repo dengan `python` (Windows, Python 3.12).

## 0. Siapkan sebelum menjalankan apa pun

1. Baca `CLAUDE.md` bagian "Cara Menjalankan" dan hasil terakhir. Eksperimen diambil dari daftar hipotesis aktif (`experiments/explore_17okt/report.md` dan file Riset bagian "## 11. Eksperimen 17 Okt+") atau dari ID yang disebut user. Antrean lama di file Riset bagian 8 hanya riwayat.
2. Tulis untuk diri sendiri: ID eksperimen (E200 ke atas), hipotesis satu kalimat, bukti datanya, **satu** perubahan yang diuji, dan acuan pembanding.
3. Tentukan acuan dari data, bukan dari ingatan: acuan terbaru tertulis di `CLAUDE.md` (baris hasil terakhir) dan dicek di `experiments/log.csv`. Kolom `diterima` bertipe teks dan untuk eksperimen `e121.py`/`e141.py` selalu tertulis "kandidat", jadi keputusan sebenarnya dibaca dari CLAUDE.md/Riset.
4. Kalau eksperimen menambah atau mengubah fitur, jalankan skill `audit-leakage` dulu. Fitur yang belum lolos audit tidak boleh masuk CV penuh.

**Sekali saja, cek penyelarasan:** jalankan `compare_oof.py --base <acuan> --cand <acuan>`. Mean-fold AUC yang dicetak harus sama dengan angka di log (±0,00005). Kalau beda, urutan OOF atau fold tidak cocok; perbaiki argumen `--folds`/`--oof-dir` sebelum mempercayai keputusan apa pun.

## 1. Screen cepat (jalur cepat, bukan gerbang)

```
python .claude/skills/run-experiment/scripts/screen.py --base-oof <acuan> --features features/<grup>_v1.parquet --prefix <prefix_baru>
```

- Screen = residual boosting di atas OOF acuan + korelasi residual. Lolos jika Δ mean-fold AUC ≥ +0,0005 **atau** ada kolom dengan |z| > 3.
- Lolos → CV penuh 3 seed.
- **Tidak lolos tetapi hipotesis punya bukti kuat dari data** (analisis pembangkit, residual, struktur data) → tetap CV penuh 3 seed. Screen tidak bisa melihat interaksi dengan fitur yang sudah ada dan bisa meremehkan perubahan representasi.
- Tidak lolos dan bukti lemah → catat "ditolak di screen" beserta angkanya.
- Screen tidak nested dan sedikit optimis; ia hanya menyaring, tidak memutuskan.
- Tanpa LightGBM (mis. lingkungan uji), tambahkan `--engine sklearn`. Di laptop tim selalu pakai default `lgb`.

Model, parameter, seed, dan blend tidak perlu screen; langsung ke langkah 2.

## 2. CV penuh

- Pakai pipeline yang sudah ada: tambahkan konfigurasi ke `EXPERIMENTS` (dan builder ke `GROUPS`/registry jika fitur baru) di `run.py`, atau pakai `e121.py --cv --use ... --extra ...` untuk set acuan sekarang. Ganti versi cache (`_v2`) kalau logika fitur berubah.
- Jalankan 3 seed (42, 43, 44); simpan `oof/<EXP>_s<seed>.npy` dan `preds/<EXP>_s<seed>.npy`.
- Job yang kira-kira > 30 menit **jangan** dijalankan sebagai background task Claude. Buat launcher `.cmd` seperti `experiments/p5/run_p5.cmd` dan jalankan via `Start-Process`, atau pindahkan ke Kaggle Notebook (privat). Job > 3 jam: minta izin dulu.
- Jangan pernah mengubah fold, `SEED`, atau data di `Dataset/`.

## 3. Putuskan

```
python .claude/skills/run-experiment/scripts/compare_oof.py --base <acuan> --cand <EXP>
```

Script membaca `oof/<nama>_s<seed>.npy` (dirata-rata) atau `oof/<nama>.npy`, menghitung AUC dan KS per fold, lalu bootstrap berpasangan berstrata per fold (1000×, seed 42) untuk Δ mean-fold AUC:

| Kondisi | Keputusan |
| --- | --- |
| P(Δ > 0) ≥ 0,95 dan Δ ≥ +0,0003 | **Diterima** |
| 0,80 ≤ P < 0,95, atau P ≥ 0,95 tetapi Δ < +0,0003 | **Kandidat** (simpan OOF & preds untuk digabung nanti) |
| P < 0,80 | **Ditolak** |

- Kedua sisi wajib OOF rata-rata **≥ 3 seed** pada fold yang sama; kurang dari itu keputusan "belum sah".
- Jumlah fold yang naik dicatat sebagai informasi, bukan syarat. P ≥ 0,95 dengan hanya 1–2 fold naik → periksa fold anomali sebelum menerima.
- Δ > +0,005 dari satu perubahan → hentikan, jalankan `audit-leakage` sebelum melapor "diterima".
- KS dicatat (dan dipakai sebagai pemecah seri), keputusan memakai **mean-fold AUC**.
- Ambang adalah pilihan tim, bukan hukum; usulan perubahan ambang ditanyakan ke user dulu. Gabungan akhir dikonfirmasi ulang dengan ablation dan seed baru.

## 4. Catat

1. `experiments/log.csv`, `oof/`, `preds/` ditulis oleh pipeline. Pastikan baris eksperimen ada.
2. Status eksperimen E200+ ditulis di file Riset bagian "## 11. Eksperimen 17 Okt+" (buat jika belum ada); E00–E112 di file Planning, E120–E199 di file Riset bagian lama. **Jangan gabungkan kedua file.** Format: `Diterima Δ +0,0012, CI [+0,0006; +0,0018], P 0,99 (3 seed, naik 5/5 fold)`.
3. Kalau acuan berubah, tambahkan satu baris hasil di `CLAUDE.md` agar sesi berikutnya tahu acuan baru. Acuan tidak diganti sebelum dilaporkan ke user.

## 5. Submit ke Kaggle? (usulan ke user)

- Usulkan ke user kapan dan file mana yang perlu di-submit; user yang mengunggah. Kuota 3/hari, reset 05.00.
- Sebelum upload, selalu `verify_submission.py --log` dari skill `check-repro` (MD5 tercatat di `submissions/LOG.md`).
- Selisih public kecil antara dua submission sendiri adalah noise. Fitur, bobot blend, dan pilihan final diputuskan dari CV.

## Larangan (berlaku di semua langkah)

- `*_id` tidak boleh jadi fitur. Data hanya dari `Dataset/`. Tidak ada model pretrained atau data eksternal.
- Fitur yang memakai target (model level baris, target encoding, KNN target mean, fitur hasil prediksi bertarget) harus **nested** (untuk fold luar k, model penghasil fitur tidak boleh melihat label fold k) dan lolos uji label acak.
- Tidak menyalin notebook publik peserta lain; membaca ide boleh, kode harus milik sendiri.

## Laporan ke user

Tutup setiap eksperimen dengan blok singkat ini:

```
E<ID> - <judul>
Hipotesis : ...
Bukti     : ...
Perubahan : ...
Hasil     : mean-fold AUC <acuan> → <kandidat> (Δ +0,00xx, CI 95% [.., ..], P 0,xx, naik x/5 fold), KS <..> → <..>, seed <n>
Keputusan : Diterima / Kandidat / Ditolak
Dicatat   : log.csv, Status di Riset bagian 11
Berikutnya: <hipotesis berikutnya>
```
