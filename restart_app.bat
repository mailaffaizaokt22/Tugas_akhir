@echo off
echo Clearing Streamlit cache...
if exist "%USERPROFILE%\.streamlit\cache" rmdir /s /q "%USERPROFILE%\.streamlit\cache"
if exist __pycache__ rmdir /s /q __pycache__
if exist .streamlit\cache rmdir /s /q .streamlit\cache

echo Starting Streamlit...
call venv\Scripts\activate && streamlit run app.py
