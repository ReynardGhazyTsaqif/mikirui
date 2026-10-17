@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1r.npz --basefs all --name E122r >> experiments\e121_stdout.log 2>&1
%PY% e121.py --cv --use e121_ordermodel_v1r.npz+e122_gru_v1r.npz --basefs all --name E122rr >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E122r selesai >> experiments\e121_stdout.log
