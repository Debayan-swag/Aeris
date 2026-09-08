@echo off
echo ============================================
echo TerraWatch AI - Streamlit Dashboard
echo ============================================
echo.
echo Starting application...
echo.
echo Make sure FastAPI server is running:
echo   uvicorn backend.main:app --reload
echo.
echo Press Ctrl+C to stop the dashboard
echo.
streamlit run streamlit_app.py
