@echo off
setlocal
cd /d "%~dp0"
call conda activate bci-dayloop
streamlit run web\app.py
endlocal

