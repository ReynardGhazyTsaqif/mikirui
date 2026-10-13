@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% tune_e146.py --trials 60 --budget 60 >> experiments\tune_e146_stdout.log 2>&1
%PY% e121.py --cv --use e121_ordermodel_v1.npz --name E146 --basefs all --study lgb_e146 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E146 selesai >> experiments\e121_stdout.log
