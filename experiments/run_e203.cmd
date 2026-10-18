@echo off
rem E203 (H12 pendahuluan): sensitivitas parameter LGB di set E200, 8 konfigurasi x 5 fold, lr 0,05, 1 seed. Perkiraan 30-50 menit.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
"C:\Program Files\Python312\python.exe" e203_sens.py --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --extra rp_events_v1.parquet >> experiments\e203_stdout.log 2>&1
echo [%date% %time%] launcher E203 selesai >> experiments\e203_stdout.log
