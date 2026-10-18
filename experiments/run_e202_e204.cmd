@echo off
rem Rantai job 18 Okt (berurutan, hemat RAM): E202 H05 10-fold bersarang (+-33 mnt) -> E203 sensitivitas parameter
rem (+-40 mnt) -> E204 H21 inflow non-gaji (+-15 mnt). Semua di atas set E200 (E122x1 + rp_ev). Total +-1,5 jam.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
set USE=e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz
echo [%date% %time%] mulai E202 >> experiments\e121_stdout.log
%PY% e121.py --cv --use %USE% --basefs all --extra rp_events_v1.parquet --subfolds 2 --name E202 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] mulai E203 >> experiments\e203_stdout.log
%PY% e203_sens.py --use %USE% --extra rp_events_v1.parquet >> experiments\e203_stdout.log 2>&1
echo [%date% %time%] mulai E204 >> experiments\e121_stdout.log
%PY% e121.py --cv --use %USE% --basefs all --extra rp_events_v1.parquet+cand_h21_v1.parquet --name E204 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E202-E204 selesai >> experiments\e121_stdout.log
