---
name: check-repro
description: Memastikan submission dan notebook final DSC LOGIKA UI 2026 valid, reproducible, dan identik (MD5) dengan submission terpilih di Kaggle - syarat lolos 8 besar. Gunakan skill ini sebelum setiap upload ke Kaggle, saat membuat atau mengubah notebook final di final/, saat user menyebut "cek submission", "MD5", "fresh run", "Run All", "notebook final", "reproducible", atau mendekati tenggat 27-30 Oktober.
---

# check-repro

Guidebook: delapan tim teratas lolos **hanya jika output notebook sama dengan output submission yang dipilih di Kaggle**; kalau tidak, posisi diganti tim berikutnya. Notebook juga bernilai 10% (reproduksibilitas dan kelancaran kode). Tenggat notebook 30 Okt 2026 23.59 WIB, telat −1 poin per menit; target internal upload 30 Okt ≤ 12.00 WIB.

Script ada di `.claude/skills/check-repro/scripts/`.

## A. Sebelum setiap upload ke Kaggle

```
python .claude/skills/check-repro/scripts/verify_submission.py submissions/<file>.csv --data-dir Dataset --log submissions/LOG.md --note "<exp_id>, mean-fold AUC ..., KS ..."
```

Memeriksa kolom `application_id,default_probability`, 30.000 baris, ID unik dan sama persis dengan `sample_submission`/test, tanpa NaN, nilai dalam [0, 1], tidak konstan, jumlah desimal, dan jenis akhir baris (CRLF vs LF). Mencetak MD5 dan menambah satu baris ke `submissions/LOG.md`. Upload file yang persis itu; jangan diedit atau disimpan ulang di Excel.

## B. Membangun notebook final (27–29 Okt)

1. **Self-contained.** Panitia hanya menerima `.ipynb`: inline semua kode `src/`, hitung fitur dari data mentah, jangan membaca `features/`, `oof/`, `preds/`, atau `experiments/`.
2. Sel pertama: cetak versi Python dan library, `SEED = 42`, `DATA_DIR` yang mendeteksi Kaggle (`/kaggle/input/...`) atau lokal.
3. Parameter hasil tuning dan daftar fitur terpilih di-hardcode beserta asal-usulnya; `RUN_TUNING = False`.
4. Determinisme: LightGBM `seed`, `deterministic=True`, `force_row_wise=True`, `num_threads` tetap; XGBoost `seed`, `tree_method="hist"`, `nthread` tetap; CatBoost `random_seed`, `thread_count` tetap; tanpa `time_limit`; kolom selalu `sorted()`.
5. Simpan submission dengan pembulatan dan akhir baris tetap, supaya MD5 sama di Windows dan di Kaggle (Linux):
   ```python
   sub["default_probability"] = sub["default_probability"].round(6)
   sub.to_csv(OUT_PATH, index=False, lineterminator="\n")
   print(hashlib.md5(open(OUT_PATH, "rb").read()).hexdigest())
   ```
   Tanpa `lineterminator="\n"`, pandas di Windows menulis CRLF dan MD5-nya pasti beda dengan hasil run di Kaggle.
6. Merge ke `sample_submission` berdasarkan `application_id`, lalu assert 30.000 baris, ID unik, tanpa NaN, nilai dalam [0, 1].
7. Markdown penjelasan di setiap bagian (struktur 12 bagian di planning: ringkasan, setup, data, EDA, fitur, validasi, model, ablation, ensemble, interpretasi, submission + MD5, kesimpulan).

Cek statis:

```
python .claude/skills/check-repro/scripts/check_notebook.py final/<NomorPeserta>_<NamaTim>_Notebook.ipynb
```

Memeriksa nama file, impor dari `src`, pembacaan cache/OOF, `time_limit`, `RUN_TUNING`, `SEED`, `DATA_DIR`, `to_csv` tanpa `lineterminator`, sel yang error, urutan eksekusi (bukti Run All), dan jumlah markdown.

## C. Fresh run dan pencocokan MD5 (H-2)

1. Restart kernel, clear output, Run All. Dari terminal:
   ```
   python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 final/<nama>.ipynb --output run1.ipynb
   ```
   Simpan `submission.csv` hasilnya sebagai `run1.csv`, ulangi menjadi `run2.csv`.
2. Bandingkan: `python .claude/skills/check-repro/scripts/verify_submission.py run2.csv --data-dir Dataset --compare run1.csv`. MD5 harus sama.
3. Ulangi sekali di mesin lain (laptop anggota lain atau Kaggle Notebook dengan data lomba di-attach). MD5 harus tetap sama.
4. **Upload output notebook itu ke Kaggle, lalu pilih submission itu secara manual.** Submission lama dari `run.py` tidak dihasilkan notebook, jadi jangan dipilih kecuali MD5-nya terbukti sama. Simpan screenshot halaman submission terpilih.
5. Catat nama file, MD5, mean-fold AUC, KS, dan public di `submissions/LOG.md`.

Soal memilih 2 submission: jika panitia belum mengonfirmasi bahwa keduanya boleh dan bagaimana dicocokkan, pilih satu saja (blend robust). Jika dua diizinkan, notebook harus menulis keduanya dengan MD5 yang cocok.

## D. Kalau MD5 tidak sama

Jalankan `verify_submission.py <baru> --compare <lama>` dan baca diagnosisnya:

| Gejala | Penyebab paling mungkin | Perbaikan |
| --- | --- | --- |
| Hanya akhir baris beda (CRLF vs LF) | `to_csv` tanpa `lineterminator` | `lineterminator="\n"` |
| Selisih maksimum < 1e-6, korelasi rank ≈ 1 | Floating point / jumlah thread | Pembulatan 6 desimal, `num_threads` tetap |
| Korelasi rank < 0,9999 | Sumber acak tidak terkunci | Cari seed yang hilang, iterasi `set`, `time_limit`, urutan kolom, fitur yang tergantung urutan baris |
| Jumlah/ID baris beda | Merge atau filter berbeda | Periksa merge ke `sample_submission` |

Laporkan ke user dengan blok:

```
Submission : <file>  MD5 <...>
Validasi   : lolos / gagal (alasan)
Notebook   : <hasil check_notebook>
Fresh run  : run1 = run2 ? ya/tidak; mesin kedua ? ya/tidak
Kaggle     : file yang dipilih = output notebook ? ya/tidak
```
