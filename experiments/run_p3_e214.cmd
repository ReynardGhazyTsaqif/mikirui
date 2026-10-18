@echo off
rem PROMPT 3 Langkah 4d: E214 = CatBoost tuned (study tune_cat, setting E150c) pada F*, 5-fold 3 seed. Perkiraan 2,8 jam.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
"C:\Program Files\Python312\python.exe" e121.py --cv --study cat --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --basefs all --extra rp_events_v1.parquet+ab_nonin_v1.parquet --name E214 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E214 selesai >> experiments\e121_stdout.log
