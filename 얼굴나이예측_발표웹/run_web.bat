@echo off
chcp 65001 >nul
cd /d "%~dp0"
"%~dp0..\전이학습\.venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
pause
