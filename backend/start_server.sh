#!/bin/bash
# Start the DeepFake Detection Backend Server

echo "=========================================="
echo "  DeepFake Detection Backend Server"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run: python -m venv venv"
    echo "Then: source venv/bin/activate"
    echo "Then: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "[*] Activating virtual environment..."
source venv/bin/activate

# Check if config.json exists
if [ ! -f "app/config.json" ]; then
    echo "[WARNING] config.json not found in app directory"
    echo "Please create app/config.json with model configuration"
fi

# Start server
echo "[*] Starting FastAPI server..."
echo "[*] API will be available at: http://localhost:8000"
echo "[*] API docs at: http://localhost:8000/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
