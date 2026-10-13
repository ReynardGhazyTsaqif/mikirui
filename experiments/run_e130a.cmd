@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
"C:\Program Files\Python312\python.exe" e121.py --cv --use e121_ordermodel_v1.npz --name E130a --basefs all >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E130a selesai >> experiments\e121_stdout.log
