@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Creating Python 3.12 virtual environment...
    py -3.12 -m venv venv
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements-streamlit.txt
)
echo Starting Skin Disease Detection App...
venv\Scripts\python.exe -m streamlit run skin_app.py
