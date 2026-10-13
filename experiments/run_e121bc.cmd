@echo off
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
set PY="C:\Program Files\Python312\python.exe"
%PY% e121.py --build --tag _b --leaves 255 --minchild 200 --lr 0.05 >> experiments\e121_stdout.log 2>&1
%PY% e121.py --cv --use e121_ordermodel_v1_b.npz --name E121b >> experiments\e121_stdout.log 2>&1
%PY% e121.py --build --tag _c --item >> experiments\e121_stdout.log 2>&1
%PY% e121.py --cv --use e121_ordermodel_v1.npz+e121_ordermodel_v1_c.npz --name E121c >> experiments\e121_stdout.log 2>&1
echo [%date% %time%] launcher E121bc selesai >> experiments\e121_stdout.log
