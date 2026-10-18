# LAMPIRAN — Ide dari sumber lain

> **Kirim ini HANYA setelah** Claude Code menyerahkan daftar hipotesisnya sendiri (akhir Langkah 3 PROMPT UTAMA). Kalau dikirim lebih awal, Claude Code akan terpaku pada daftar ini. Kalau Langkah 3 sudah lewat, kirim sekarang. Tempel semua isi di bawah garis sebagai pesan berikutnya.

---

Berikut daftar ide dari sumber lain. Ini bukan perintah dan belum tentu benar.

Cara memakainya:

1. Bandingkan dengan daftarmu. Gabungkan ide yang sama, dan buang ide yang terbantah oleh buktimu.
2. Urutkan ulang semuanya dengan kriteria yang sama: bukti × dampak × kebaruan. Biaya tidak ikut menentukan urutan.
3. Untuk ide yang belum punya bukti, cari dulu buktinya di data sebelum diuji.
4. Tanyakan ke saya kalau ada yang tidak jelas.

## Daftar ide

1. **Learning curve (diagnostik).** Ukur AUC model terbaik saat dilatih dengan 25%, 50%, 75%, dan 100% data latih di tiap fold. Kalau kurvanya masih naik tajam, data tambahan (ide 4) berpotensi besar.

2. **Audit ID dan urutan baris di tabel pendukung.** Periksa `loan_id`, `order_id`, `installment_id`, `bureau_observation_id`, dan posisi baris di file terhadap target. Selama ini pemeriksaan seperti ini baru dilakukan di tabel aplikasi. Kalau ditemukan pola, laporkan, jangan dipakai.

3. **Model hazard level cicilan.**
   - Labelnya milik cicilan itu sendiri (telat ≥ 30 hari), bukan target customer. Jumlahnya sekitar 1,36 juta label.
   - Kovariat diambil as-of satu bulan sebelum jatuh tempo: belanja, saldo, riwayat bayar, dan biro.
   - Model diterapkan pada kondisi customer saat cutoff untuk menghasilkan peluang telat pada cicilan pinjaman baru, lalu dijadikan fitur.
   - Butuh builder fitur as-of yang bisa dihitung untuk sembarang tanggal.

4. **Pinjaman lama sebagai data berlabel** (E126 di Riset bagian 10).
   - Ada sekitar 61.505 pinjaman lama yang jendela labelnya sudah lengkap.
   - Fitur dihitung as-of tanggal keputusan pinjaman lama, dan DPD dihitung ulang.
   - Bisa dipakai sebagai skor PD (fitur) atau sebagai baris latih tambahan.

5. **Kelas laten atau persona.** Gaussian mixture tanpa target pada fitur perilaku kunci semua customer, dengan posterior sebagai fitur. Coba juga KNN target mean (nested, E127).

6. **Artefak angka.** Pembulatan dan heaping, misalnya income deklarasi, `loan_amount`, dan `requested_amount` yang bulat. Periksa juga bagian pecahan skor biro.

7. **Konsistensi tujuan pinjaman dengan belanja.** Contoh pasangan: medical ↔ health, education ↔ education, home_improvement ↔ home. Periksa juga lonjakan belanja menjelang pengajuan.

8. **Pinjam untuk menutup utang.** Bandingkan sisa pokok pinjaman lama yang masih berjalan (dihitung dari jadwal anuitas) dengan `loan_amount` baru. Tambahkan juga jarak hari ke cicilan berikutnya dan apakah customer sedang menunggak saat mengajukan.

9. **Kalender dan proxy gajian.** Gunakan tanggal dalam bulan saat mengajukan. Perkirakan tanggal gajian tiap customer dari modus tanggal bayar cicilannya.

10. **Analisis kasus salah (E120).** Baca timeline lengkap dari FN dan FP terbesar.

11. **Fitur otomatis.** OpenFE atau Featuretools dengan cutoff per aplikasi (E123/E124).

12. **Objektif ranking LightGBM.** Coba `rank_xendcg` dan `lambdarank` sebagai varian GBDT.

13. **Pseudo-label test dan refit 100% data.** Pseudo-label harus nested per fold (E160); refit 100% data adalah E145. Dampaknya kecil, jadi kerjakan di minggu terakhir.

## Masukan tambahan dari mentor

14. **Model berbeda dengan preprocessing minimal, plus AutoML sebagai pembanding.** Latih beberapa model (CatBoost, XGBoost, AutoGluon mode kualitas tertinggi) pada fitur yang hampir mentah dan pada matriks fitur kita, dengan fold yang sama supaya OOF bisa dibandingkan. Tujuannya:
    - mengecek apakah feature engineering kita justru membuang informasi,
    - mendapat batas atas kasar,
    - mendapat varian kuat yang berbeda untuk submission.

    AutoGluon sulit menghasilkan output identik antar-run, jadi hanya boleh masuk notebook final kalau lolos uji MD5.

15. **Pola missing dan nilai aneh: jangan buru-buru dibersihkan atau dibuang.**
    - Petakan kombinasi pola missing lintas tabel, misalnya punya biro tapi tidak punya rekening, atau punya aplikasi lama tapi tanpa cicilan.
    - Cari nilai nol, nilai maksimum, dan nilai yang terpotong.
    - Cek pelanggaran rumus antar kolom, misalnya closing ≠ opening + inflows − outflows, order_total ≠ Σ item_total, utilization ≠ revolving_outstanding / limit, atau approved > requested.
    - Uji apakah pola-pola itu berkaitan dengan target.

16. **Loop fitur berbantuan LLM (gaya CAAFE).** Kamu sendiri mengusulkan 10–20 fitur yang masuk akal secara domain kredit di setiap putaran, menulis kodenya, lalu mengujinya dengan screen atau CV. Hasil tiap putaran menjadi umpan balik untuk putaran berikutnya. Ulangi beberapa putaran.

17. **Fitur "aneh" otomatis dalam jumlah besar.** Selain OpenFE/Featuretools (ide 11), coba transformasi massal: rasio dan selisih antar pasangan fitur teratas, rank di dalam kelompok kategori, frekuensi nilai, dan jarak ke median kelompok. Saring semuanya dengan screen. Dengan komputer yang kuat, ruang pencariannya bisa dibuat besar.

18. **Explainable AI untuk analisis.** Jalankan SHAP pada model terbaik untuk:
    - kasus FN dan FP terbesar,
    - nilai interaksi SHAP, untuk menemukan pasangan fitur yang layak dijadikan fitur interaksi eksplisit,
    - plot dependence, untuk mencari pola janggal (tidak monoton, berbentuk tangga, atau punya ambang tajam) yang bisa menandakan aturan pembangkit data.

    Hasil SHAP juga dipakai untuk presentasi final.

19. **Post-processing yang relevan untuk KS.** Thresholding dan kalibrasi biasa tidak mengubah KS, karena KS hanya bergantung pada urutan prediksi. Yang bisa mengubah KS adalah urutan antar kelompok:
    - kalibrasi ulang per segmen (misalnya punya atau tidak punya riwayat, atau tier bunga) pada OOF secara nested, lalu skor semua segmen digabung kembali,
    - model kedua yang khusus mengurutkan ulang baris di sekitar ambang KS (cari ambangnya dari OOF), dilatih secara nested.

    Nilai keduanya dengan aturan bootstrap biasa.
