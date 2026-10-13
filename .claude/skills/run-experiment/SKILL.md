---
name: run-experiment
description: Menjalankan dan memutuskan satu eksperimen DSC LOGIKA UI 2026 (prediksi default kredit, metrik KS) sesuai protokol tim. Gunakan skill ini setiap kali user minta menjalankan, menguji, atau menilai apa pun yang bisa mengubah skor - fitur baru, grup fitur, model, parameter, iterasi, seed, blend/ensemble, seleksi fitur - atau bilang "lanjut", "coba E1xx", "screen fitur ini", "diterima atau tidak?", "bandingkan dengan acuan", bahkan jika kata "eksperimen" tidak disebut.
---

# run-experiment

Satu eksperimen = satu perubahan, dinilai berpasangan terhadap acuan terbaik pada fold yang sama, lalu dicatat. Skill ini menegakkan protokol dari `Planning Model DSC LOGIKA UI 2026.md` (bagian Protokol eksperimen) dan `Riset Peningkatan Skor DSC LOGIKA UI 2026.md` (bagian 6 dan 8). Kalau dokumen itu dan skill ini berbeda, dokumen proyek yang menang; beri tahu user perbedaannya.

Script ada di `.claude/skills/run-experiment/scripts/`. Jalankan dari root repo dengan `python` (Windows, Python 3.12).

## 0. Siapkan sebelum menjalankan apa pun

1. Baca `CLAUDE.md` bagian "Cara Menjalankan" dan hasil terakhir, lalu antrean di bagian 8 dokumen riset. Ambil eksperimen berikutnya dari antrean kecuali user menyebut ID lain.
2. Tulis untuk diri sendiri: ID eksperimen, hipotesis satu kalimat, **satu** perubahan yang diuji, dan acuan pembanding.
3. Tentukan acuan dari data, bukan dari ingatan: model diterima terbaik di `experiments/log.csv` lewat `run.accepted_rows()` (kolom `diterima` bertipe teks, jangan pakai `== True`). Saat skill ini ditulis acuannya `T_lgb_b3_4f3c7f` (mean-fold AUC 0,83788, OOF KS 0,5138).
4. Kalau eksperimen menambah atau mengubah fitur, jalankan skill `audit-leakage` dulu. Fitur yang belum lolos audit tidak boleh masuk CV penuh.

**Sekali saja, cek penyelarasan:** jalankan `compare_oof.py --base <acuan> --cand <acuan>`. Mean-fold AUC yang dicetak harus sama dengan angka di log (±0,00005). Kalau beda, urutan OOF atau fold tidak cocok; perbaiki argumen `--folds`/`--oof-dir` sebelum mempercayai keputusan apa pun.

## 1. Screen cepat (khusus ide fitur baru, ±2 menit)

```
python .claude/skills/run-experiment/scripts/screen.py --base-oof T_lgb_b3_4f3c7f --features features/<grup>_v1.parquet --prefix ec_ordermodel_
```

- Lolos jika Δ mean-fold AUC ≥ +0,0005 **atau** ada kolom dengan |z| korelasi residual > 3. Lolos berarti boleh CV penuh, belum berarti diterima.
- Tidak lolos: catat "ditolak di screen" beserta angkanya, lalu berhenti. Jangan masuk CV penuh hanya karena AUC univariatnya tinggi; riset 12 Okt menunjukkan banyak fitur kuat sendirian tetapi sudah terserap model.
- Screen tidak nested dan sedikit optimis. Ia hanya menyaring, tidak memutuskan.
- Tanpa LightGBM (mis. lingkungan uji), tambahkan `--engine sklearn`. Di laptop tim selalu pakai default `lgb`.

Model, parameter, seed, dan blend tidak perlu screen; langsung ke langkah 2.

## 2. CV penuh

- Pakai pipeline yang sudah ada: tambahkan konfigurasi ke `EXPERIMENTS` (dan builder ke `GROUPS`/registry jika fitur baru) di `run.py`, ganti versi cache (`_v2`) kalau logika fitur berubah.
- Jalankan 3 seed: `python run.py <EXP> --seeds 42 43 44`.
- Job yang kira-kira > 30 menit (CatBoost, tuning, null importance, seed banyak) **jangan** dijalankan sebagai background task Claude. Buat launcher `.cmd` seperti `experiments/p5/run_p5.cmd` dan minta user menjalankannya via `Start-Process`, atau pindahkan ke Kaggle Notebook. RAM laptop 7,7 GB pernah membuat pipeline dimatikan.
- Jangan pernah mengubah fold, `SEED`, atau data di `Dataset/`.

## 3. Putuskan

```
python .claude/skills/run-experiment/scripts/compare_oof.py --base T_lgb_b3_4f3c7f --cand <EXP>
```

Script membaca `oof/<nama>_s<seed>.npy` (dirata-rata) atau `oof/<nama>.npy`, menghitung AUC dan KS per fold, lalu menerapkan aturan:

| Kondisi | Keputusan |
| --- | --- |
| Δ mean-fold AUC ≥ +0,0005 dan naik di ≥ 4/5 fold | Diterima |
| Lolos aturan tapi Δ < 0,001 dan salah satu sisi < 3 seed | Tipis: ulangi dengan 3 seed dulu |
| Selain itu | Ditolak |

- Keputusan memakai **mean-fold AUC**. KS hanya dicatat dan dipakai sebagai pemecah seri (E170), karena KS jauh lebih noisy.
- **Blend/ensemble** wajib dinilai dengan OOF rata-rata ≥ 3 seed (`--blend`). OOF 1 seed melebih-lebihkan kenaikan blend sekitar 2×.
- Δ besar yang mencurigakan (> +0,005 dari satu perubahan fitur) → hentikan, jalankan `audit-leakage` sebelum melapor "diterima".
- Untuk keputusan yang sangat tipis, tambahkan `--bootstrap 300` untuk interval kepercayaan Δ.

## 4. Catat

1. `experiments/log.csv`, `oof/`, `preds/` sudah ditulis otomatis oleh `run.py`. Pastikan baris eksperimen ada.
2. Perbarui kolom Status di dokumen yang memiliki ID itu: E00–E112 di `Planning Model DSC LOGIKA UI 2026.md`, E120 ke atas di `Riset Peningkatan Skor DSC LOGIKA UI 2026.md`. **Jangan gabungkan kedua file.** Format status singkat: `Diterima +0,0012 (3 seed, 5/5 fold)` atau `Ditolak (−0,0003, 2/5 fold)`.
3. Kalau acuan berubah, perbarui baris hasil di `CLAUDE.md` agar sesi berikutnya tahu acuan baru.

## 5. Submit ke Kaggle? (opsional)

- Submit hanya jika CV naik jelas (sekitar ≥ +0,002 AUC) atau ada perubahan pipeline besar. Kuota 3/hari, reset 05.00; pakai maksimal 1–2 per hari.
- Sebelum upload, jalankan `verify_submission.py` dari skill `check-repro` dan catat MD5 ke `submissions/LOG.md`.
- Selisih public < ±0,005 antara dua submission sendiri adalah noise. Jangan pernah memilih fitur, bobot blend, atau submission final berdasarkan public.

## Larangan (berlaku di semua langkah)

- `*_id` tidak boleh jadi fitur. Data hanya dari `Dataset/`. Tidak ada model pretrained atau data eksternal.
- Fitur yang memakai target (model level baris, target encoding, KNN target mean, fitur hasil prediksi bertarget) harus **nested**: untuk fold luar k, model penghasil fitur tidak boleh melihat label fold k.
- Tidak menyalin notebook publik peserta lain; membaca ide boleh, kode harus milik sendiri.

## Laporan ke user

Tutup setiap eksperimen dengan blok singkat ini, lalu satu langkah berikutnya dari antrean:

```
E<ID> - <judul>
Hipotesis : ...
Perubahan : ...
Hasil     : mean-fold AUC <base> → <cand> (Δ +0,00xx, naik x/5 fold), KS <base> → <cand>, seed <n>
Keputusan : Diterima / Tipis / Ditolak
Dicatat   : log.csv, Status di <dokumen>
Berikutnya: <ID berikutnya di antrean>
```
