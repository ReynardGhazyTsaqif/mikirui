@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% e121.py --cv --use e121_ordermodel_v1.npz --name E150_xgb --basefs all --study xgb >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E150_xgb selesai >> experiments\e121_stdout.log
%PY% e121.py --cv --use e121_ordermodel_v1.npz --name E150_cat --basefs all --study cat >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E150_cat selesai >> experiments\e121_stdout.log
