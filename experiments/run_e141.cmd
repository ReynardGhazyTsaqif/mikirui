@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% e141.py --rounds 3000 --lr 0.05 --seeds 42 --name E141_scr >> experiments\e141_stdout.log 2>&1
echo [%date% %time%] launcher E141_scr selesai >> experiments\e141_stdout.log
