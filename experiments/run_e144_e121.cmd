@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% refit.py --model lgb --fs b3 --rounds 15000 --tag _r15k --seeds 45 46 >> experiments\refit_stdout.log 2>&1
%PY% e121.py --build >> experiments\e121_stdout.log 2>&1
%PY% e121.py --build --shuffle >> experiments\e121_stdout.log 2>&1
%PY% e121.py --cv --rounds 15000 --seeds 42 43 44 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E144+E121 selesai >> experiments\e121_stdout.log
