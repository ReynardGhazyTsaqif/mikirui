@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% e141.py --rounds 5000 --lr 0.05 --seeds 42 43 44 --name E141 >> experiments\e141_stdout.log 2>&1
echo [%date% %time%] launcher E141 selesai >> experiments\e141_stdout.log
