@echo off
rem PROMPT 3 Langkah 8: validasi aturan iterasi refit (+-15 mnt) -> E216 pseudo-label lunak nested pada F*, 3 seed (+-50 mnt).
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
echo [%date% %time%] mulai refit check >> experiments\p3_stdout.log
%PY% experiments\prompt3\p3_refit_check.py --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --extra rp_events_v1.parquet+ab_nonin_v1.parquet --seed 42 >> experiments\p3_stdout.log 2>&1
echo [%date% %time%] mulai E216 >> experiments\p3_stdout.log
%PY% e160_pseudo.py --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --extra rp_events_v1.parquet+ab_nonin_v1.parquet --mode soft --name E216 >> experiments\p3_stdout.log 2>&1
echo [%date% %time%] launcher step8 selesai >> experiments\p3_stdout.log
