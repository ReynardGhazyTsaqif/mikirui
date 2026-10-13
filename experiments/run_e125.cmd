@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% e121.py --cv --use e121_ordermodel_v1.npz --name E125 --basefs all --extra x_pred_v1.parquet >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E125 selesai >> experiments\e121_stdout.log
