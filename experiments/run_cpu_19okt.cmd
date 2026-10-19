@echo off
rem 19 Okt: jalur utama = keluarga CPU (E205 + E207, public 0,56428). Berurutan, total +-3,5 jam:
rem E209 GBDT set CPU + ab_nonin 5-fold (+-15 mnt) -> E211 versi 10-fold (+-25 mnt) -> cek aturan refit set CPU (+-20 mnt)
rem -> E216c pseudo-label nested set CPU (+-1 jam) -> E210 set GPU 10-fold (+-25 mnt) -> H22 fitur mentah (+-1,5 jam).
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
echo [%date% %time%] mulai E209 >> experiments\p3_stdout.log
%PY% e121.py --cv --use e121_ordermodel_v1.npz --basefs all --extra rp_events_v1.parquet+ab_nonin_v1.parquet --name E209 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] mulai E211 >> experiments\p3_stdout.log
%PY% e121.py --cv --use e121_ordermodel_v1.npz --basefs all --extra rp_events_v1.parquet+ab_nonin_v1.parquet --subfolds 2 --name E211 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] mulai refit check CPU >> experiments\p3_stdout.log
%PY% experiments\prompt3\p3_refit_check.py --use e121_ordermodel_v1.npz --extra rp_events_v1.parquet+ab_nonin_v1.parquet --seed 42 >> experiments\p3_stdout.log 2>&1
echo [%date% %time%] mulai E216c >> experiments\p3_stdout.log
%PY% e160_pseudo.py --use e121_ordermodel_v1.npz --extra rp_events_v1.parquet+ab_nonin_v1.parquet --mode soft --name E216c >> experiments\p3_stdout.log 2>&1
echo [%date% %time%] mulai E210 >> experiments\p3_stdout.log
%PY% e121.py --cv --use e121_ordermodel_v1.npz+e122_gru_v1r.npz+e122_gru_v1_bigrur.npz+e122_gru_v1_tfr.npz+e122x_rp_v1r.npz+e122x_ab_v1r.npz+e122x_pa_v1r.npz --basefs all --extra rp_events_v1.parquet+ab_nonin_v1.parquet --subfolds 2 --name E210 >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] mulai H22 >> experiments\p3_stdout.log
%PY% h22_nearraw.py --models lgb xgb cat >> experiments\h22_stdout.log 2>&1
echo [%date% %time%] launcher cpu 19okt selesai >> experiments\p3_stdout.log
