@echo off
REM Start the DeepFake Detection Backend Server

echo ==========================================
echo   DeepFake Detection Backend Server
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if config.json exists
if not exist "app\config.json" (
    echo [WARNING] config.json not found in app directory
    echo Please create app\config.json with model configuration
    pause
)

REM Start server
echo [*] Starting FastAPI server...
echo [*] API will be available at: http://localhost:8000
echo [*] API docs at: http://localhost:8000/docs
echo.
echo Press CTRL+C to stop the server
echo.

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
