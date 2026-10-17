@echo off
rem E141x: LGB DART 5000 ronde x 3 seed pada set E122x1 (528 kolom + ecm + seq + seqx). Perkiraan 1,5-2 jam.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
"C:\Program Files\Python312\python.exe" e141.py --rounds 5000 --lr 0.05 --seeds 42 43 44 --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --name E141x >> experiments\e141_stdout.log 2>&1
echo [%date% %time%] launcher E141x selesai >> experiments\e141_stdout.log
