@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
powershell -NoProfile -Command "Wait-Process -Id 16744 -ErrorAction SilentlyContinue"
echo [%date% %time%] launcher E150_xgb selesai >> experiments\e121_stdout.log
%PY% e141.py --rounds 3000 --lr 0.05 --seeds 42 --name E141_scr >> experiments\e141_stdout.log 2>&1
echo [%date% %time%] launcher E141_scr selesai >> experiments\e141_stdout.log
%PY% e121.py --cv --use e121_ordermodel_v1.npz --name E150_cat --basefs all --study cat >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E150_cat selesai >> experiments\e121_stdout.log
