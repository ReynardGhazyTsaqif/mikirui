@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
echo [%date% %time%] launcher mulai >> experiments\p5\stdout.log
"C:\Program Files\Python312\python.exe" tune.py >> experiments\p5\stdout.log 2>&1
echo [%date% %time%] launcher selesai, exit %errorlevel% >> experiments\p5\stdout.log
