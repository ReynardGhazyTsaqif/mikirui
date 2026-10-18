@echo off
rem Antrean laptop mandiri (18 Okt sore, tanpa perlu sesi Claude): tunggu s33 selesai, lalu berurutan
rem rantai 4a (E210, E209, E211; +-1,5 jam) -> Langkah 8 (cek refit + E216 pseudo-label; +-1,1 jam)
rem -> Langkah 6a (E210n, E122x1n, NF_gbdt; +-1,3 jam) -> H22 (model fitur mentah; +-1,5 jam). Total +-5,5 jam.
rem Hentikan kapan saja dengan menutup jendela ini (hasil seed yang sudah selesai tetap tersimpan).
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
:tunggu
findstr /C:"launcher s33 selesai" "experiments\explore_17okt\s33_stdout.log" >nul 2>&1
if errorlevel 1 (
  timeout /t 60 /nobreak >nul
  goto tunggu
)
call "D:\Unand\Sems 7\Lomba\DSC\dsc-logika\experiments\run_p3_chain1.cmd"
call "D:\Unand\Sems 7\Lomba\DSC\dsc-logika\experiments\run_p3_step8.cmd"
call "D:\Unand\Sems 7\Lomba\DSC\dsc-logika\experiments\run_p3_step6a.cmd"
call "D:\Unand\Sems 7\Lomba\DSC\dsc-logika\experiments\run_h22.cmd"
cd /d "D:\Unand\Sems 7\Lomba\DSC\dsc-logika"
echo [%date% %time%] antrean laptop selesai >> experiments\p3_stdout.log
