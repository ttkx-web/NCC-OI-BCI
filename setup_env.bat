@echo off
setlocal
cd /d "%~dp0"
call conda env create -f environment.yml
call conda activate bci-dayloop
echo Base environment created. Install the approved PyTorch wheel, then run:
echo python -m pip install -e . --no-deps
echo See docs\server_deployment.md for the platform-specific GPU procedure.
endlocal
