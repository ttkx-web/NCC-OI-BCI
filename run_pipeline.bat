@echo off
setlocal
cd /d "%~dp0"
call conda activate bci-dayloop
python scripts\run_pipeline.py --config configs\day1_bnci_s01.yaml
endlocal

