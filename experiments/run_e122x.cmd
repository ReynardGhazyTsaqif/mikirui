@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --basefs all --name E122x1 >> experiments\e121_stdout.log 2>&1
%PY% e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_all_v1r.npz --basefs all --name E122x2 >> experiments\e121_stdout.log 2>&1
%PY% e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz+e122x_all_v1r.npz --basefs all --name E122x3 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E122x selesai >> experiments\e121_stdout.log
