---
name: audit-leakage
description: Audit kebocoran (leakage) dan konsistensi train/test untuk fitur DSC LOGIKA UI 2026 sebelum fitur dipakai. Gunakan skill ini setiap kali builder fitur di src/features ditulis atau diubah, sebelum grup fitur baru masuk CV penuh, saat CV melonjak mencurigakan atau CV naik tapi public turun, untuk fitur yang memakai target (model level baris, target encoding, fitur hasil prediksi, KNN), dan sebelum feature freeze 26 Okt. Juga saat user bertanya "ini bocor nggak?", "cek leakage", atau "kenapa skornya aneh".
---

# audit-leakage

Kebocoran membuat CV terlalu optimis lalu gagal di private (45% nilai), dan di final juri akan bertanya bagaimana kita menjamin tidak ada kebocoran waktu. Audit ini punya tiga lapis: baca kode, cek angka, dan uji khusus untuk fitur bertarget. Laporkan temuan apa adanya; jangan "memperbaiki" angka agar terlihat aman.

Konteks data yang sudah diverifikasi (CLAUDE.md dan riset 12 Okt): semua timestamp tabel pendukung < `decision_datetime` aplikasi saat ini, split train/test acak, tidak ada kebocoran lewat ID atau urutan baris. Jadi bahaya terbesar bukan di data mentah, melainkan di **kode fitur kita sendiri**.

## Lapis 1: baca kode builder

Buka builder yang berubah (`src/features/*.py`, entri di `src/features/registry.py`) dan periksa satu per satu. Tulis lolos/gagal per butir.

1. **Cutoff per aplikasi.** Setiap tabel pendukung difilter `timestamp < cutoff` dengan `cutoff = decision_datetime` aplikasi saat ini per `customer_id`. Bukan tanggal global, bukan `max()` seluruh data, bukan "hari ini". Jendela waktu (7/30/90/180/365 hari) dan recency dihitung dari cutoff itu.
2. **Kolom waktu yang tepat:** previous applications → `decision_datetime`; cicilan → `due_date` dan `last_payment_datetime`; biro → `available_datetime` (bukan hanya `report_datetime`); rekening → `month` (bulan lengkap); e-commerce → `transaction_datetime`.
3. **Biro saat ini vs sebelumnya dipisah** lewat `application_id` (laporan untuk aplikasi saat ini vs laporan milik previous application).
4. **E-commerce:** `order_total` di-dedup per `order_id`; order `cancelled` tidak dihitung sebagai belanja.
5. **Tidak ada `*_id` sebagai fitur** dan tidak ada kolom `target` di keluaran builder.
6. **Statistik global tanpa target** (median, z-score, persentil) boleh dihitung di train+test. **Apa pun yang memakai target** harus di-fit di dalam fold, dan nested bila dipakai sebagai fitur model lain (lihat Lapis 3).
7. **Train dan test diproses oleh fungsi yang sama** dalam satu panggilan; tidak ada cabang `if is_train`.
8. Kolom diurutkan `sorted()`; tidak ada iterasi `set` berisi string (urutan berubah antar proses).
9. Tidak membaca file di luar `Dataset/` (kecuali cache `features/` milik sendiri).

## Lapis 2: cek angka

```
python .claude/skills/audit-leakage/scripts/audit_features.py --features features/<grup>_v1.parquet --data-dir Dataset --prefix <prefix_baru>
```

Script memeriksa cakupan ID (70.000 train + 30.000 test, tanpa duplikat), nama kolom terlarang, kolom konstan, nilai `inf`, beda tingkat NaN train vs test, beda distribusi train vs test (statistik KS dua sampel), dan AUC univariat terhadap target. Exit code 1 jika ada ERROR.

Cara membaca hasil:

| Temuan | Arti | Tindakan |
| --- | --- | --- |
| AUC univariat ≥ 0,80 | Lebih kuat dari fitur sah terkuat yang diketahui (`rp_share_dpd1` 0,784) | Berhenti. Cari jalur ke target atau ke masa depan sebelum lanjut |
| KS train vs test > 0,05 atau beda NaN > 2 poin | Fitur dihitung berbeda di train dan test | Cari cabang kode atau join yang beda |
| KS train vs test 0,02–0,05 | Bisa jadi wajar untuk fitur diskrit, bisa jadi bug | Periksa distribusinya |
| ID tidak lengkap / duplikat | Join salah | Perbaiki sebelum CV |

Lalu jalankan adversarial validation pada set fitur baru (pola E04 di `run.py`). Hasil harus AUC ≈ 0,50; di atas 0,55 berarti ada fitur yang membedakan train dan test.

## Lapis 3: fitur yang memakai target

Berlaku untuk model level baris (E121), fitur hasil prediksi dengan target (E125), KNN target mean (E127), stacking, dan pseudo-label (E160).

1. **Nested wajib.** Untuk setiap fold luar k: latih model penghasil fitur hanya pada fold ≠ k (dengan CV dalam untuk baris train fold itu), lalu prediksi fold k dan test. Satu model yang dilatih pada semua label train lalu dipakai sebagai fitur = bocor.
2. **Uji label acak.** Bangun ulang fitur dengan target train diacak (`np.random.default_rng(42).permutation`). Jalankan `audit_features.py` pada keluaran itu: AUC univariat terhadap target **asli** harus ≈ 0,50 (≤ 0,52). Kalau masih tinggi, informasi target asli bocor lewat jalur lain (mis. fitur konteks customer yang meniru agregat).
3. **Fitur model baris harus intrinsik milik baris** (waktu relatif cutoff, nominal, kuantitas, kategori, metode bayar). Fitur konteks customer di model baris menurunkan nilai screen menjadi ~0 (temuan riset 12 Okt butir J).

## Laporan ke user

```
Audit <grup/eksperimen>
Kode    : x/9 lolos (sebutkan yang gagal + baris kodenya)
Angka   : <ringkasan ERROR/WARN dari script>
Adversarial: AUC <...>
Target  : nested ya/tidak, uji label acak AUC <...> (jika relevan)
Status  : AMAN untuk CV penuh / PERLU PERBAIKAN (apa)
```

Kalau ditemukan kebocoran yang berasal dari data panitia sendiri (bukan kode kita), jangan dimanfaatkan; laporkan ke user agar dikonfirmasi ke panitia.
