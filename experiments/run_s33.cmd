@echo off
rem s33 (18 Okt): kontribusi tiap grup sumber pada 28k vs 56k baris latih (11 grup x 5 fold x 2), +-35 mnt.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika\experiments\explore_17okt"
echo [%date% %time%] mulai s33 >> s33_stdout.log
"C:\Program Files\Python312\python.exe" s33_group_learning.py >> s33_stdout.log 2>&1
echo [%date% %time%] launcher s33 selesai >> s33_stdout.log
