@echo off
rem Rantai 18 Okt sore: s31 H07 sub-model per sumber (+-15 mnt) -> s32 H14 mekanisme tier (+-10 mnt)
rem -> E207 = LGB DART 5000 ronde x 3 seed (setting E141) pada set E205 / CPU saja (528 + ecm + rp_ev), +-2 jam.
rem Total +-2,5 jam.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika\experiments\explore_17okt"
set PY="C:\Program Files\Python312\python.exe"
echo [%date% %time%] mulai s31 >> s31_s32_stdout.log
%PY% s31_source_submodels.py >> s31_s32_stdout.log 2>&1
echo [%date% %time%] mulai s32 >> s31_s32_stdout.log
%PY% s32_tier_mechanism.py >> s31_s32_stdout.log 2>&1
echo [%date% %time%] s31-s32 selesai >> s31_s32_stdout.log
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
echo [%date% %time%] mulai E207 >> experiments\e141_stdout.log
%PY% e141.py --rounds 5000 --lr 0.05 --seeds 42 43 44 --use e121_ordermodel_v1.npz --extra rp_events_v1.parquet --name E207 >> experiments\e141_stdout.log 2>&1
echo [%date% %time%] launcher E207 selesai >> experiments\e141_stdout.log
