@echo off
REM Build libinjection.dll on Windows using MSVC cl.exe.
REM Requires Microsoft Visual C++ Build Tools.

setlocal

set ROOT=%~dp0..
set SRC=%ROOT%\external\libinjection\src
set OUT_DIR=%ROOT%\lib
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

REM Try common vcvars locations
for %%V in (
  "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) do (
  if exist %%V (
    call %%V
    goto :build
  )
)
echo ERROR: cannot find vcvars64.bat (need MSVC Build Tools)
exit /b 1

:build
cd /d "%OUT_DIR%"
cl /LD /O2 /W3 /MD ^
   /D_CRT_SECURE_NO_WARNINGS ^
   /I "%SRC%" ^
   "%SRC%\libinjection_sqli.c" ^
   /link /DEF:"%~dp0..\external\libinjection\libinjection.def" /OUT:libinjection.dll /IMPLIB:libinjection.lib

if exist libinjection.dll (
    echo Build successful: %OUT_DIR%\libinjection.dll
) else (
    echo Build failed
    exit /b 1
)
