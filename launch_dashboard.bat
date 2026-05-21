@echo off
echo ======================================================================
echo LAUNCHING QUANTITATIVE TRADING DASHBOARD
echo ======================================================================
echo.
echo Starting Streamlit server...
echo.
echo The dashboard will open in your browser automatically.
echo Press Ctrl+C to stop the server.
echo.
streamlit run dashboard.py --server.port 8501 --browser.gatherUsageStats false
pause

