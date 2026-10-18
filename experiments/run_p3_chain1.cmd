@echo off
rem PROMPT 3 Langkah 4a (18 Okt): F* = set E122x1 + rp_ev + ab_nonin.
rem E210 F* LGB 10-fold bersarang 3 seed (+-40 mnt) -> E209 F*-CPU (528 + ecm + rp_ev + ab_nonin) 5-fold 3 seed (+-15 mnt)
rem -> E211 F*-CPU 10-fold 3 seed (+-30 mnt). Total +-1,5 jam.
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
echo [%date% %time%] mulai E210 >> experiments\e121_stdout.log
%PY% e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --basefs all --extra rp_events_v1.parquet+ab_nonin_v1.parquet --subfolds 2 --name E210 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] mulai E209 >> experiments\e121_stdout.log
%PY% e121.py --cv --use e121_ordermodel_v1.npz --basefs all --extra rp_events_v1.parquet+ab_nonin_v1.parquet --name E209 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] mulai E211 >> experiments\e121_stdout.log
%PY% e121.py --cv --use e121_ordermodel_v1.npz --basefs all --extra rp_events_v1.parquet+ab_nonin_v1.parquet --subfolds 2 --name E211 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher p3 chain1 selesai >> experiments\e121_stdout.log
