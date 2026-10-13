@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
"C:\Program Files\Python312\python.exe" refit.py --model lgb --fs b3 selb2 --rounds 15000 --tag _r15k >> experiments\refit_stdout.log 2>&1
echo [%date% %time%] launcher E140a selesai, exit %errorlevel% >> experiments\refit_stdout.log
