@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% e130.py >> experiments\e130_stdout.log 2>&1
%PY% e121.py --cv --use e121_ordermodel_v1.npz --name E130b --basefs sel_tuned >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E130b selesai >> experiments\e121_stdout.log
