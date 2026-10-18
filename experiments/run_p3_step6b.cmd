@echo off
rem PROMPT 3 Langkah 6b: fold baru seed 2026, DART 5000 ronde non-nested 3 seed (NF_dart). Perkiraan 2,3 jam.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
"C:\Program Files\Python312\python.exe" experiments\prompt3\p3_newfold.py --model dart --seeds 42 43 44 >> experiments\p3_stdout.log 2>&1
echo [%date% %time%] launcher step6b selesai >> experiments\p3_stdout.log
