# Skill proyek Claude Code — DSC LOGIKA UI 2026

Tiga skill yang menegakkan protokol di `Planning Model DSC LOGIKA UI 2026.md` dan `Riset Peningkatan Skor DSC LOGIKA UI 2026.md`.

| Skill | Kapan terpakai | Script |
| --- | --- | --- |
| `/run-experiment` | Menjalankan dan memutuskan eksperimen (fitur, model, seed, blend), "lanjut" ke antrean E1xx | `screen.py`, `compare_oof.py` |
| `/audit-leakage` | Builder fitur baru/berubah, CV melonjak, fitur bertarget (E121, E125, E127) | `audit_features.py` |
| `/check-repro` | Sebelum upload Kaggle, notebook final, fresh run, MD5 | `verify_submission.py`, `check_notebook.py` |

## Pasang

1. Salin folder `.claude/` dari zip ke **root repo** (sejajar `CLAUDE.md`, `run.py`, `Dataset/`). Jika `.claude/` sudah ada, gabungkan isi `skills/`.
2. Buka ulang Claude Code di folder repo. Skill dipanggil otomatis sesuai konteks, atau langsung lewat `/run-experiment`, `/audit-leakage`, `/check-repro`.
3. Library yang dipakai sudah ada di laptopmu: pandas, numpy, scipy, scikit-learn, lightgbm (pyarrow untuk parquet). `check_notebook.py` hanya butuh library standar.

## Tiga hal pertama yang perlu dijalankan

```
# 1. Uji mesin screen di laptop (harus "SELFTEST LULUS")
python .claude/skills/run-experiment/scripts/screen.py --selftest

# 2. Cek penyelarasan OOF: mean-fold AUC yang dicetak harus 0,83788 (sama dengan log)
python .claude/skills/run-experiment/scripts/compare_oof.py --base T_lgb_b3_4f3c7f --cand T_lgb_b3_4f3c7f

# 3. Validasi submission terbaik sekarang + catat MD5
python .claude/skills/check-repro/scripts/verify_submission.py submissions/P5_single_T_lgb_b3_4f3c7f.csv --log submissions/LOG.md --note "P5 single, public 0,54685"
```

Kalau langkah 2 tidak menghasilkan 0,83788, berarti urutan OOF atau fold tidak sama dengan asumsi script (OOF berurutan seperti `train_applications.csv`, fold dari `folds/folds.csv` atau `StratifiedKFold(5, shuffle=True, random_state=42)`). Sesuaikan `--folds`/`--oof-dir`, atau minta Claude Code menyesuaikan `load_target_and_folds()` dengan cara `run.py` membuat fold.

## Tambahan opsional untuk CLAUDE.md

```
## Skill proyek
- Eksperimen apa pun: ikuti skill run-experiment (screen -> CV 3 seed -> compare_oof.py -> catat Status).
- Fitur baru/berubah: skill audit-leakage sebelum CV penuh.
- Sebelum upload Kaggle dan untuk notebook final: skill check-repro (MD5, lineterminator="\n", round(6)).
```

## Yang sudah diuji

Semua script dijalankan pada proyek tiruan (20.000 train / 8.000 test, struktur file sama): aturan terima/tipis/tolak dan aturan blend ≥ 3 seed, selftest screen (sinyal lolos, noise tidak), audit menangkap kolom ID, fitur bocor (AUC 0,99), pergeseran train/test, beda NaN, inf, dan konstan; verify menangkap baris hilang, NaN, nilai di luar [0, 1], serta mendiagnosis beda CRLF/LF, pembulatan, urutan baris, dan pipeline berbeda; check_notebook menangkap 13 pelanggaran pada notebook contoh.

Belum diuji di sini: mesin LightGBM di `screen.py` (LightGBM tidak bisa dipasang di lingkungan uji). Logikanya sama dengan mesin cadangan scikit-learn yang lulus uji; pastikan dengan `--selftest` di laptop.
