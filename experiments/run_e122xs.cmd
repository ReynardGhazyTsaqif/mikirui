@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
"C:\Program Files\Python312\python.exe" e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_shuffle_v1r.npz --basefs all --name E122xs >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E122xs selesai >> experiments\e121_stdout.log
