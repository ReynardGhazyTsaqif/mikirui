@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
"C:\Program Files\Python312\python.exe" e121.py --cv --study cat --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --basefs all --seeds 42 --name E150c_scr >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E150c_scr selesai >> experiments\e121_stdout.log
