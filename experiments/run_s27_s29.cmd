@echo off

rem Rantai 18 Okt siang (setelah E204): E205 = E200 versi CPU saja (528 + ecm + rp_ev, tanpa fitur GRU GPU; +-20 mnt)

rem -> s27 uji kapasitas residual di atas OOF E200 (+-20 mnt) -> s29 kurva belajar set E200 (+-10 mnt) -> s31 H07 sub-model per sumber (+-15 mnt)
rem -> s32 H14 mekanisme tier (+-10 mnt). Total +-75 mnt.

cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"

set PY="C:\Program Files\Python312\python.exe"

echo [%date% %time%] mulai E205 >> experiments\e121_stdout.log

%PY% e121.py --cv --use e121_ordermodel_v1.npz --basefs all --extra rp_events_v1.parquet --name E205 >> experiments\e121_stdout.log 2>&1

cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika\experiments\explore_17okt"

echo [%date% %time%] mulai s27 >> s27_s29_stdout.log

%PY% s27_residual_capacity.py >> s27_s29_stdout.log 2>&1

echo [%date% %time%] mulai s29 >> s27_s29_stdout.log

%PY% s29_learning_curve_e200.py >> s27_s29_stdout.log 2>&1

echo [%date% %time%] mulai s31 >> s27_s29_stdout.log

%PY% s31_source_submodels.py >> s27_s29_stdout.log 2>&1

echo [%date% %time%] mulai s32 >> s27_s29_stdout.log
%PY% s32_tier_mechanism.py >> s27_s29_stdout.log 2>&1
echo [%date% %time%] launcher E205-s27-s29 selesai >> s27_s29_stdout.log

