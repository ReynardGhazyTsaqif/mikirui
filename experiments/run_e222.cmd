@echo off
rem E222 (19 Okt malam): F*-CPU (528 + ecm + rp_ev + ab_nonin) + paket mekanisme (x_auxpd, H08/H09, ramp change-point),
rem LGB GBDT tuned 10-fold bersarang, 3 seed. Perkiraan 45 menit. Lalu blend dengan DART E219 untuk diuji di public.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
echo [%date% %time%] mulai E222 >> experiments\p3_stdout.log
"C:\Program Files\Python312\python.exe" e121.py --cv --use e121_ordermodel_v1.npz --basefs all --extra rp_events_v1.parquet+ab_nonin_v1.parquet+x_auxpd_v1.parquet+cand_h08h09_v1.parquet+cand_ramp_v1.parquet --subfolds 2 --name E222 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E222 selesai >> experiments\p3_stdout.log
