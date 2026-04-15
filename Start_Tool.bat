@echo off
title Bank Statement Sync Tool
cd /d "%~dp0"

echo Checking/installing requirements...
pip install -r requirements.txt -q

echo Starting the app...
streamlit run app.py

pause
