@echo off
rem Setelah E207 (18 Okt): s32 ulang (H14 mekanisme tier, tanpa kolom pa_ turunan bunga/cicilan sekarang; +-20 mnt)
rem -> s33 kontribusi tiap grup sumber pada 28k vs 56k baris latih (11 grup x 5 fold x 2; +-35 mnt). Total +-55 mnt.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika\experiments\explore_17okt"
set PY="C:\Program Files\Python312\python.exe"
echo [%date% %time%] mulai s32 ulang >> s33_stdout.log
%PY% s32_tier_mechanism.py >> s33_stdout.log 2>&1
echo [%date% %time%] mulai s33 >> s33_stdout.log
%PY% s33_group_learning.py >> s33_stdout.log 2>&1
echo [%date% %time%] launcher s33 selesai >> s33_stdout.log
