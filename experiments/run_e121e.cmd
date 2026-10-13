@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% e121.py --build --aggx --tag _e >> experiments\e121e_stdout.log 2>&1
echo [%date% %time%] launcher E121e build selesai >> experiments\e121e_stdout.log
