@echo off
setlocal EnableExtensions

title NCC BCI Console Launcher

set "REPO=E:\code\NCC-OI-BCI"
set "NODE_DIR=E:\Apps\node-v22.22.3-win-x64"
set "PNPM_DIR=%APPDATA%\npm"
set "PYTHON_EXE=D:\Apps\miniconda\envs\bci-dayloop\python.exe"
set "API_URL=http://127.0.0.1:8000/api/v1/health"
set "WEB_URL=http://127.0.0.1:3000"

set "PATH=%NODE_DIR%;%PNPM_DIR%;%PATH%"
set "PNPM_CMD=%PNPM_DIR%\pnpm.cmd"
if not exist "%PNPM_CMD%" set "PNPM_CMD=%NODE_DIR%\pnpm.cmd"

echo.
echo ==========================================
echo       NCC BCI Console Launcher
echo ==========================================
echo.

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python 3.11 executable was not found:
  echo         %PYTHON_EXE%
  goto :ERROR
)
if not exist "%NODE_DIR%\node.exe" (
  echo [ERROR] Node executable was not found:
  echo         %NODE_DIR%\node.exe
  goto :ERROR
)
if not exist "%REPO%\apps\console-api\app\main.py" (
  echo [ERROR] Console API entrypoint was not found:
  echo         %REPO%\apps\console-api\app\main.py
  goto :ERROR
)
if not exist "%REPO%\apps\console-web\package.json" (
  echo [ERROR] Console Web package.json was not found:
  echo         %REPO%\apps\console-web\package.json
  goto :ERROR
)
if not exist "%PNPM_CMD%" (
  echo [ERROR] pnpm.cmd was not found.
  echo         Checked: %PNPM_DIR%\pnpm.cmd
  echo         Checked: %NODE_DIR%\pnpm.cmd
  goto :ERROR
)

echo [OK] Python:
"%PYTHON_EXE%" --version
echo [OK] Node:
"%NODE_DIR%\node.exe" --version
echo [OK] pnpm:
call "%PNPM_CMD%" --version

if not exist "%REPO%\apps\console-web\node_modules" (
  echo.
  echo [SETUP] Installing Console Web dependencies...
  pushd "%REPO%\apps\console-web"
  call "%PNPM_CMD%" install
  if errorlevel 1 (
    popd
    echo [ERROR] pnpm install failed. The API and Web services were not started.
    goto :ERROR
  )
  popd
)

echo.
call :PORT_LISTENING 8000
if errorlevel 1 (
  echo [START] NCC BCI API :8000
  start "NCC BCI API :8000" /D "%REPO%\apps\console-api" cmd.exe /k call "%PYTHON_EXE%" -m uvicorn app.main:app --reload
) else (
  echo [OK] NCC BCI API :8000 is already running.
)

echo [WAIT] API health check...
call :WAIT_HTTP "%API_URL%" 30
if errorlevel 1 (
  echo [ERROR] API did not become healthy within 30 seconds.
  goto :ERROR
)

call :PORT_LISTENING 3000
if errorlevel 1 (
  echo [START] NCC BCI Web :3000
  start "NCC BCI Web :3000" /D "%REPO%\apps\console-web" cmd.exe /k call "%PNPM_CMD%" dev
) else (
  echo [OK] NCC BCI Web :3000 is already running.
)

echo [WAIT] Web availability check...
call :WAIT_HTTP "%WEB_URL%" 30
if errorlevel 1 (
  echo [ERROR] Web did not become available within 30 seconds.
  goto :ERROR
)

echo [OPEN] %WEB_URL%
start "" "%WEB_URL%"
echo.
echo Console is ready. API and Web run in their own windows.
exit /b 0

:PORT_LISTENING
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %~1 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
exit /b %errorlevel%

:WAIT_HTTP
set "WAIT_URL=%~1"
set /a WAIT_LIMIT=%~2
set /a WAIT_COUNT=0
:WAIT_HTTP_LOOP
powershell -NoProfile -Command "try { $response = Invoke-WebRequest -UseBasicParsing -Uri '%WAIT_URL%' -TimeoutSec 2; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { exit 0 } } catch {} ; exit 1" >nul 2>&1
if not errorlevel 1 exit /b 0
set /a WAIT_COUNT+=1
if %WAIT_COUNT% GEQ %WAIT_LIMIT% exit /b 1
timeout /t 1 /nobreak >nul
goto :WAIT_HTTP_LOOP

:ERROR
echo.
echo Launcher stopped. Review the error above and correct the local environment.
pause
exit /b 1
