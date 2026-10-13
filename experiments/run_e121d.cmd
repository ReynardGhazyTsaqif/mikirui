@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
for %%T in (ab rp bu pa) do %PY% e121d.py --table %%T >> experiments\e121_stdout.log 2>&1
%PY% e121.py --cv --use e121_ordermodel_v1.npz+e121_ordermodel_v1_ab.npz+e121_ordermodel_v1_rp.npz+e121_ordermodel_v1_bu.npz+e121_ordermodel_v1_pa.npz --name E121d >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E121d selesai >> experiments\e121_stdout.log
