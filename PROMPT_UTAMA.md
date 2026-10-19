# PROMPT UTAMA — Cari sendiri cara menembus public 0,58

> **Cara pakai:** buka sesi Claude Code baru di root repo, pakai **Opus high**, lalu tempel seluruh isi di bawah garis. Ini prompt pertama dan utama, sehingga sesi ini boleh menjalankan eksperimen penuh dan mencatatnya.
> **Setelah** Claude Code menyerahkan daftar hipotesisnya sendiri (akhir Langkah 3), kirim isi `LAMPIRAN_ide_eksternal.md` sebagai pesan berikutnya. Jangan dikirim lebih awal.

---

Kamu bergabung dengan tim DSC LOGIKA UI 2026 (prediksi default kredit, metrik KS) sebagai analis utama. Tugasmu adalah menemukan sendiri cara menaikkan skor secara besar.

Prompt ini sengaja tidak memberi saran model, fitur, atau teknik. Kamu yang mencari tahu, dari membaca seluruh proyek dan seluruh data. Gunakan Bahasa Indonesia.

## Target

- **Posisi sekarang:** public terbaik kita 0,55858 (peringkat ±39). Peringkat 1 ada di sekitar 0,578.
- **Target:** public **≥ 0,58**.
- **Target dalam ukuran CV:** hubungan public–CV yang tercatat di file Riset bagian 10 memberi kira-kira mean-fold AUC 0,858 untuk public 0,58, atau +0,013 dari model terbaik sekarang. Verifikasi sendiri hubungan itu dari `submissions/LOG.md` dan `experiments/log.csv`.
- **Anggap gap ini nyata dan bisa dikejar.** Jangan jadikan noise alasan untuk berhenti mencari atau menyimpulkan "sudah mentok". Kegunaan noise hanya satu: menilai apakah selisih kecil antara dua submission kita sendiri bisa dipercaya.
- **Keputusan diambil dari CV, bukan dari public**, karena private juga bernilai 45%. Kenaikan yang nyata di CV akan terbawa ke public dan private sekaligus.

## Cara kerja

1. **Pelajari dulu, baru bertindak.** Jangan memulai dengan melatih model, tuning, atau blend.
2. **Jangan percaya kesimpulan lama begitu saja**, termasuk yang tertulis di dokumen kami. Contohnya: "tidak ada kebocoran", "fitur X ditolak", "sinyal sudah terserap", atau "evaluasi arah 16 Okt". Kalau sebuah kesimpulan lama menghalangi arah yang menjanjikan, periksa ulang dengan data.
3. **Verifikasi ulang asumsi dasar** dan jangan percayai hanya karena tertulis: definisi dan implementasi metrik, format submission, skema validasi, dan kesamaan pipeline train dan test.
4. **Setiap ide harus punya bukti dari data** sebelum diuji.
5. **Tulis catatan kerja secara berkala** ke `experiments/explore_17okt/report.md`, supaya pekerjaan bisa dilanjutkan di sesi baru kalau konteks penuh.

## Protokol bertanya (wajib)

- **Jangan memakai asumsimu sendiri.** Setiap kali ada yang tidak jelas, tanyakan ke saya. Contohnya: arti sebuah kolom, keputusan tim sebelumnya, file mana yang benar, aturan lomba, prioritas, batas waktu atau komputasi, dan apa yang sudah atau belum dicoba.
- **Tidak ada batas jumlah pertanyaan.** Kelompokkan pertanyaan, beri nomor, dan jelaskan singkat kenapa setiap pertanyaan penting.
- **Bagian yang bergantung pada jawaban menunggu** sampai saya menjawab. Bagian lain yang tidak bergantung boleh tetap dikerjakan.
- Kalau saya menjawab "tidak tahu", catat itu sebagai ketidakpastian, lalu rancang cara mengujinya dengan data.

## Batasan (ini aturan, bukan saran)

- **Aturan lomba:**
  - hanya data di `Dataset/`,
  - wajib Python,
  - tidak ada data eksternal atau model pretrained berbasis data luar,
  - tidak menyalin notebook publik peserta lain.
- **Kebocoran:**
  - `*_id` tidak boleh menjadi fitur,
  - fitur yang memakai target wajib nested dan lolos uji label acak,
  - pola yang tampak seperti kebocoran dari data panitia dilaporkan ke saya, jangan dipakai.
- **Notebook final:**
  - harus self-contained dan menghasilkan file yang identik (MD5 sama) dengan submission terpilih,
  - setiap komponen baru diberi tag reproduksibilitas: `CPU-deterministik`, `GPU`, atau `butuh cek`.
- **Laptop:** Windows 11, Python 3.12 (pakai `python -m pip`), RAM 7,7 GB.
  - Job > 30 menit dijalankan lewat launcher `.cmd` (Start-Process).
  - Job > 3 jam harus minta izin dulu, atau dipindahkan ke Kaggle Notebook. Dataset dan notebook Kaggle harus privat.
- **Submission:**
  - kuota 3 per hari, reset pukul 05.00,
  - usulkan ke saya kapan dan file mana yang perlu di-submit, selalu lewat `verify_submission.py --log`,
  - saya yang mengunggah.
- **Jadwal:**
  - feature freeze 26 Okt,
  - notebook final 27–29 Okt,
  - upload notebook 30 Okt paling lambat 12.00 WIB.
- **Waktu dan komputasi bukan alasan untuk membatasi ide:**
  - Jangan takut kekurangan waktu untuk menulis kode, training, atau pekerjaan lain. Keluarkan semua ide yang ada.
  - Anggap saya bisa menyelesaikan semua saranmu dalam 1 hari, karena saya bisa memakai komputer lain yang lebih kuat.
  - Jangan membuang, menunda, atau mengecilkan ide karena berat atau lama. Cukup catat kebutuhan komputasinya (perkiraan jam, RAM, dan GPU), supaya saya bisa memilih komputer yang tepat.
  - Tandai komponen yang akan dijalankan di komputer lain, karena notebook final tetap harus menghasilkan MD5 yang sama dengan submission terpilih.

## Langkah 1 — Pelajari seluruh proyek

Baca semuanya:

- **Dokumen:**
  - `CLAUDE.md`,
  - `Planning Model DSC LOGIKA UI 2026.md`,
  - `Riset Peningkatan Skor DSC LOGIKA UI 2026.md`,
  - `Dataset/DATA_DICTIONARY.md`,
  - guidebook di `Guidebook/` (berupa gambar).
- **Kode:**
  - semua isi `src/`,
  - `run.py`, `e1*.py`, `tune*.py`, `refit.py`,
  - folder `kaggle/`,
  - `.claude/skills/` (berisi protokol eksperimen tim).
- **Riwayat:**
  - seluruh `experiments/log.csv`,
  - `submissions/LOG.md`,
  - `experiments/importance_*.csv`,
  - laporan lain di `experiments/`.
- File biner besar (cache parquet/npy/npz) tidak perlu dibaca baris per baris. Cukup ketahui isinya.

Tulis hasilnya di bagian "Pemahaman awal" di `report.md`:

- **Ringkasan** masalah, data, aturan, dan pipeline saat ini, termasuk perintah persis untuk mereproduksi model terbaik.
- **Peta informasi:** untuk setiap tabel dan setiap kolom mentah, tandai apakah kolom itu:
  - sudah dipakai secara mendalam,
  - hanya diagregasi kasar, atau
  - belum disentuh sama sekali.
- **Daftar semua yang pernah dicoba** beserta hasilnya dan kualitas buktinya:
  - berapa seed yang dipakai dan terhadap acuan apa,
  - apakah ditolak karena memang tidak ada sinyal, atau hanya karena tumpang tindih dengan fitur lain.
- **Hal yang janggal atau perlu dipertanyakan** di kode, data, maupun keputusan sebelumnya.
- **Jadwal kerja** sampai feature freeze 26 Okt.

→ **Kirim pertanyaan batch #1.**

## Langkah 2 — Pahami ulang data sampai level baris

Dengan caramu sendiri, pahami setiap tabel sampai ke baris mentahnya:

- bagaimana data ini kemungkinan dibangkitkan,
- bagaimana tabel-tabel saling berhubungan,
- seperti apa "kehidupan" seorang customer di dalam data,
- apa yang membedakan customer yang gagal bayar dari yang tidak, terutama mereka yang salah diprediksi oleh model terbaik kita.

Pilih sendiri metodenya, lalu tulis buktinya di `report.md`.

→ **Kirim pertanyaan batch #2.**

## Langkah 3 — Hipotesis

- Tulis minimal 15 hipotesis. Setiap hipotesis berisi:
  - nama,
  - bukti: angka, gambar, atau contoh customer,
  - fitur atau perubahan yang diusulkan, dengan definisi persis (as-of cutoff),
  - perkiraan dampak dalam AUC beserta alasannya,
  - kebutuhan komputasi: perkiraan jam, RAM, dan GPU (hanya dicatat, bukan alasan membuang ide),
  - risiko kebocoran dan cara auditnya,
  - status kebaruan: belum pernah dicoba, atau variasi dari ID xx dengan alasan yang berbeda.
- Urutkan dengan skor bukti × perkiraan dampak × kebaruan. Biaya tidak ikut menentukan urutan.
- **Serahkan daftar ini ke saya, lalu berhenti.**
  - Saya mungkin akan mengirim daftar ide dari sumber lain.
  - Gabungkan dengan daftarmu, buang yang duplikat atau yang terbantah oleh buktimu, lalu urutkan ulang semuanya dengan kriteria yang sama.

→ **Kirim pertanyaan batch #3.**

## Langkah 4 — Uji, lalu ulangi

- **Pakai protokol tim** di `.claude/skills/` (run-experiment, audit-leakage, check-repro):
  1. builder parquet,
  2. audit,
  3. screen (jalur cepat, bukan gerbang; lihat penyesuaian skill di bawah),
  4. CV 3 seed,
  5. `compare_oof.py` dengan keputusan bootstrap (aturan di bawah),
  6. catat hasilnya.
- **Aturan keputusan (baru).** Aturan ini menggantikan aturan lama "+0,0005 dan ≥ 4/5 fold" yang tertulis di CLAUDE.md, file Planning, dan skill run-experiment.
  - Bandingkan kandidat dan acuan memakai OOF rata-rata ≥ 3 seed, pada fold yang sama.
  - Hitung Δ mean-fold AUC, lalu bootstrap berstrata per fold (1000×, seed 42) untuk mendapatkan P(Δ > 0) dan CI 95%.
  - **Diterima:** P(Δ > 0) ≥ 0,95 dan Δ ≥ +0,0003.
  - **Kandidat:** P antara 0,80 dan 0,95, atau P ≥ 0,95 tetapi Δ < +0,0003. Simpan OOF dan preds-nya untuk digabung nanti.
  - **Ditolak:** P < 0,80.
  - **Jumlah fold yang naik** dicatat sebagai informasi, bukan syarat. Kalau P ≥ 0,95 tetapi hanya 1–2 fold yang naik, periksa dulu apakah ada fold yang anomali sebelum menerima.
  - **Δ > +0,005 dari satu perubahan:** jalankan audit kebocoran dulu sebelum diterima.
  - KS tetap dicatat, tetapi keputusan memakai mean-fold AUC.
  - Ambang ini pilihan, bukan hukum. Karena eksperimennya banyak, sebagian kecil perubahan yang lolos 0,95 tetap bisa kebetulan, jadi gabungan akhirnya akan dikonfirmasi ulang dengan ablation dan seed baru. Kamu boleh mengusulkan perubahan ambang beserta alasannya, tapi tanyakan dulu ke saya.
- **Sebelum eksperimen baru pertama:**
  1. Perbarui `compare_oof.py` supaya keputusan default-nya memakai aturan di atas, lalu perbarui skill run-experiment dan bagian aturan di CLAUDE.md agar konsisten.
  2. Sesuaikan skill supaya tidak membatasi pencarian:
     - **run-experiment:** hapus perintah mengambil eksperimen dari antrean lama di file Riset bagian 8, karena antrean itu sekarang hanya riwayat. Jadikan screen jalur cepat, bukan gerbang: hipotesis yang punya bukti kuat dari data tetap diuji dengan CV penuh walaupun gagal screen, karena screen tidak bisa melihat interaksi dengan fitur yang sudah ada. Hapus juga nama acuan lama yang tertulis di skill: acuan selalu diambil dari log terbaru.
     - **audit-leakage:** AUC univariat ≥ 0,80 bukan lagi ERROR yang menghentikan pekerjaan, melainkan pemicu audit wajib: cek cutoff waktu, distribusi train vs test, adversarial validation, dan uji label acak bila fitur memakai target. Fitur yang berupa skor dari model lain memang wajar punya AUC tinggi.
     - **check-repro:** tidak diubah.
     - **Bagian skill lain** yang ternyata menghambat temuan yang sah: laporkan ke saya beserta usulan perbaikannya.
  3. Tunjukkan diff semua perubahan di atas ke saya sebelum dipakai.
  4. Terapkan aturan baru ke eksperimen lama yang berstatus kandidat atau ditolak dengan Δ positif, selama OOF ≥ 3 seed-nya tersedia. Laporkan mana yang berubah status. Jangan mengganti acuan sebelum melapor ke saya.
- **Pencatatan:**
  - setiap eksperimen tercatat di `experiments/log.csv` lewat pipeline,
  - Status ditulis di file Riset pada bagian "## 11. Eksperimen 17 Okt+" (buat jika belum ada),
  - satu baris hasil di `CLAUDE.md` jika acuan berubah,
  - ID eksperimen baru dimulai dari E200.
- **Kalau total kenaikan masih jauh dari target, jangan berhenti:**
  - pelajari kenapa hipotesis yang gagal itu gagal,
  - kembali ke data mentah dan susun putaran hipotesis berikutnya,
  - tanyakan ke saya hal-hal yang masih belum jelas.

## Laporan

Selalu perbarui ringkasan satu halaman di awal `report.md`, berisi:

- temuan terpenting,
- hipotesis teratas beserta hasilnya,
- kenaikan CV yang sudah terkumpul dibanding target,
- rencana berikutnya,
- pertanyaan yang masih terbuka untuk saya.
