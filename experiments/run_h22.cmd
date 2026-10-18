@echo off
rem H22 lokal (mentor ide 14): LGB/XGB/CatBoost tuned pada 239 kolom agregat naif/mentah, 5 fold, 1 seed. Perkiraan 1,5 jam.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
"C:\Program Files\Python312\python.exe" h22_nearraw.py --models lgb xgb cat >> experiments\h22_stdout.log 2>&1
echo [%date% %time%] launcher H22 selesai >> experiments\h22_stdout.log
