@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% e128.py --seg nohist hist --lr 0.05 --seeds 42 --tag _scr >> experiments\e128_stdout.log 2>&1
echo [%date% %time%] launcher E128 screen selesai >> experiments\e128_stdout.log
