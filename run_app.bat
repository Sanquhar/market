@echo off
set PYTHONPATH=%cd%\src
streamlit run .\src\market\gui\interface.py
pause
