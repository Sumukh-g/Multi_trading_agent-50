@echo off
echo.
echo ========================================
echo   QuantEdge Pro Trading Dashboards
echo ========================================
echo.
echo Choose a dashboard to launch:
echo.
echo   1. Long-Term Dashboard (30-day predictions) - Port 8501
echo   2. Short-Term Dashboard (1-week predictions) - Port 8502
echo   3. Launch BOTH dashboards
echo   4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting Long-Term Dashboard on http://localhost:8501 ...
    start cmd /k "cd /d "%~dp0" && python -m streamlit run dashboard_longterm.py --server.port 8501"
    timeout /t 3 >nul
    start http://localhost:8501
)

if "%choice%"=="2" (
    echo.
    echo Starting Short-Term Dashboard on http://localhost:8502 ...
    start cmd /k "cd /d "%~dp0" && python -m streamlit run dashboard_shortterm.py --server.port 8502"
    timeout /t 3 >nul
    start http://localhost:8502
)

if "%choice%"=="3" (
    echo.
    echo Starting Long-Term Dashboard on http://localhost:8501 ...
    start cmd /k "cd /d "%~dp0" && python -m streamlit run dashboard_longterm.py --server.port 8501"
    echo Starting Short-Term Dashboard on http://localhost:8502 ...
    start cmd /k "cd /d "%~dp0" && python -m streamlit run dashboard_shortterm.py --server.port 8502"
    timeout /t 5 >nul
    start http://localhost:8501
    start http://localhost:8502
)

if "%choice%"=="4" (
    exit
)

echo.
echo Dashboards launched!
echo.
pause
