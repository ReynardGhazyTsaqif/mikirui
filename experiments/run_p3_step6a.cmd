@echo off
rem PROMPT 3 Langkah 6a: seed baru 101-103 untuk F* 10-fold (E210n, +-40 mnt) dan E122x1 (E122x1n, +-20 mnt);
rem fold baru seed 2026 GBDT non-nested 3 seed (NF_gbdt, +-20 mnt).
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
echo [%date% %time%] mulai E210n >> experiments\p3_stdout.log
%PY% e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --basefs all --extra rp_events_v1.parquet+ab_nonin_v1.parquet --subfolds 2 --seeds 101 102 103 --name E210n >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] mulai E122x1n >> experiments\p3_stdout.log
%PY% e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --basefs all --seeds 101 102 103 --name E122x1n >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] mulai NF_gbdt >> experiments\p3_stdout.log
%PY% experiments\prompt3\p3_newfold.py --model gbdt --seeds 42 43 44 >> experiments\p3_stdout.log 2>&1
echo [%date% %time%] launcher step6a selesai >> experiments\p3_stdout.log
