@echo off
setlocal
cd /d "%~dp0"
call conda env create -f environment.yml
call conda activate bci-dayloop
python -m pip install -e .
echo Environment created. Activate with: conda activate bci-dayloop
endlocal
