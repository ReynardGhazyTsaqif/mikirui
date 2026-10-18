@echo off
rem PROMPT 3 Langkah 4b: E212 = LGB DART 5000 ronde x 3 seed (setting E141) pada F*. Perkiraan 2,3 jam.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
"C:\Program Files\Python312\python.exe" e141.py --rounds 5000 --lr 0.05 --seeds 42 43 44 --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --extra rp_events_v1.parquet+ab_nonin_v1.parquet --name E212 >> experiments\e141_stdout.log 2>&1
echo [%date% %time%] launcher E212 selesai >> experiments\e141_stdout.log
