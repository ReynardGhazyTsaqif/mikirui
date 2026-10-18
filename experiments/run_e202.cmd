@echo off
rem E202 = H05: 10-fold bersarang di 5 fold (tiap fold dibelah 2, model dilatih 90%% data), set E200 (E122x1 + rp_ev).
rem 30 model (10 x 3 seed), perkiraan 30-40 menit. Mean-fold AUC dihitung pada 5 fold asli.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
"C:\Program Files\Python312\python.exe" e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --basefs all --extra rp_events_v1.parquet --subfolds 2 --name E202 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E202 selesai >> experiments\e121_stdout.log
