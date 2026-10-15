@echo off
setlocal
cd /d "%~dp0"
set "TAXI_PYTHON=C:\Users\sijan\anaconda3\envs\taxi_rl\python.exe"
set "PATH=C:\Users\sijan\anaconda3\envs\taxi_rl;C:\Users\sijan\anaconda3\envs\taxi_rl\Library\bin;C:\Users\sijan\anaconda3\envs\taxi_rl\Scripts;%PATH%"
if not exist "%TAXI_PYTHON%" (
    echo The taxi_rl environment is missing. See README.md for setup.
    pause
    exit /b 1
)
"%TAXI_PYTHON%" -m jupyterlab "Taxi_Q_Learning_Mini_Project.ipynb"
