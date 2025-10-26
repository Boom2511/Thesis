# How to Run the DeepFake Detection System

## Quick Start (5 Minutes)

### Step 1: Navigate to Backend Directory

**IMPORTANT:** Always run from the `backend` directory, NOT `backend/app`!

```bash
cd backend
```

### Step 2: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Step 3: Start Backend Server

**Option A: Using Startup Script (Recommended)**

**Windows:**
```bash
start_server.bat
```

**Linux/Mac:**
```bash
chmod +x start_server.sh
./start_server.sh
```

**Option B: Manual Command**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
🚀 Starting Ensemble Deepfake Detection Service...
📋 Project Goals:
   1.2.1 ✓ Model >90% accuracy + Real-time processing
   1.2.2 ✓ Grad-CAM visual explanations
   1.2.3 ✓ Web app with image/video/webcam support
   1.2.4 ✓ Cross-dataset robustness evaluation
📦 Loading models...
✅ Service ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Test Backend

Open a new terminal:
```bash
curl http://localhost:8000/health
```

Or visit in browser:
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### Step 5: Start Frontend (Optional)

Open a **new terminal**:

```bash
cd frontend
npm install  # First time only
npm run dev
```

Visit: http://localhost:3000

---

## Detailed Setup (First Time)

### 1. Install Python Dependencies

```bash
cd backend

# Create virtual environment (first time only)
python -m venv venv

# Activate
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Models

Create or edit `backend/app/config.json`:

```json
{
  "device": "cpu",
  "models": {
    "efficientnet_b4": {
      "enabled": true,
      "path": "../efficientnet_b4.pth",
      "weight": 0.33
    },
    "xception": {
      "enabled": true,
      "path": "../xception.pth",
      "weight": 0.33
    },
    "f3net": {
      "enabled": true,
      "path": "../f3net.pth",
      "weight": 0.34
    },
    "effort": {
      "enabled": false,
      "path": "../effort_ff++.pth",
      "weight": 0.0
    }
  },
  "ensemble": {
    "method": "weighted_average"
  }
}
```

**Notes:**
- Set `device` to `"cuda"` if you have GPU
- Disable models you don't have weights for
- Paths are relative to `backend/app/` directory

### 3. Place Model Weights

Download or obtain model weights and place them in `backend/`:

```
backend/
├── efficientnet_b4.pth
├── xception.pth
├── f3net.pth
├── effort_ff++.pth (optional)
└── app/
    └── config.json
```

### 4. Test the System

```bash
# From backend directory
python -m uvicorn app.main:app --reload

# In another terminal, test:
curl -X POST "http://localhost:8000/api/detect/image" \
  -F "file=@test_image.jpg"
```

---

## Common Commands

### Backend

```bash
# Start server (from backend/)
python -m uvicorn app.main:app --reload

# Start server on different port
python -m uvicorn app.main:app --reload --port 8001

# Start with no reload (production)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run evaluation
python evaluate_model.py

# Run cross-dataset evaluation
python cross_dataset_eval.py

# Generate architecture diagrams
python create_model_diagrams.py
```

### Frontend

```bash
# Start development server (from frontend/)
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Clean rebuild
rm -rf .next node_modules
npm install
npm run dev
```

---

## API Endpoints

Once running, access these endpoints:

### Documentation
- **API Docs (Interactive):** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health
- **Model Info:** http://localhost:8000/api/models

### Detection
- **Image Detection:** `POST /api/detect/image`
- **Video Detection:** `POST /api/video/detect`
- **Batch Video:** `POST /api/video/detect_batch`
- **Webcam (WebSocket):** `WS /api/video/ws/webcam`

### Example Usage

**Image Detection:**
```bash
curl -X POST "http://localhost:8000/api/detect/image?generate_heatmap=true" \
  -F "file=@image.jpg" \
  -o result.json
```

**Video Detection:**
```bash
curl -X POST "http://localhost:8000/api/video/detect" \
  -F "file=@video.mp4" \
  -F "frame_skip=5" \
  -F "max_frames=100" \
  -o result.json
```

---

## Project Goals Testing

### Test Goal 1.2.1 (>90% Accuracy)

```bash
cd backend
python evaluate_model.py

# Provide dataset path when prompted
# Example: /path/to/dataset (with real/ and fake/ subdirectories)

# Results saved in: evaluation_results/
# Check: evaluation_report.json
```

### Test Goal 1.2.2 (Grad-CAM)

**Via Web UI:**
1. Go to http://localhost:3000
2. Upload an image
3. See Grad-CAM heatmap in results

**Via API:**
```bash
curl -X POST "http://localhost:8000/api/detect/image?generate_heatmap=true" \
  -F "file=@image.jpg" \
  | jq '.gradcam'  # Shows base64 heatmap
```

### Test Goal 1.2.3 (Web App)

**Image Upload:**
- Visit http://localhost:3000
- Drag and drop image
- View results with Grad-CAM

**Batch Processing:**
- Switch to "Batch Analysis" tab
- Upload multiple images
- Export results (CSV/JSON)

**Video Processing:**
```bash
curl -X POST "http://localhost:8000/api/video/detect" \
  -F "file=@video.mp4"
```

**Webcam (requires frontend integration):**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/video/ws/webcam');
ws.send(base64_frame);
```

### Test Goal 1.2.4 (Cross-Dataset)

```bash
cd backend
python cross_dataset_eval.py

# Edit script to configure datasets:
# - FaceForensics++
# - Celeb-DF
# - DFDC
# - Your custom datasets

# Results saved in: cross_dataset_results/
```

---

## Troubleshooting

### "ModuleNotFoundError"

**Problem:** Running from wrong directory

**Solution:**
```bash
# Always run from backend/, NOT backend/app/
cd backend  # Correct
python -m uvicorn app.main:app --reload
```

### "Config file not found"

**Problem:** Missing config.json

**Solution:**
```bash
# Create backend/app/config.json
# See "Configure Models" section above
```

### "Model weights not found"

**Problem:** Missing .pth files

**Solution:**
```bash
# Option 1: Disable models in config.json
{"models": {"xception": {"enabled": false}}}

# Option 2: Download/obtain weights and place in backend/
```

### Port already in use

**Solution:**
```bash
# Use different port
python -m uvicorn app.main:app --reload --port 8001
```

### Frontend can't connect

**Solution:**
```bash
# 1. Check backend is running
curl http://localhost:8000/health

# 2. Check CORS in backend/app/main.py

# 3. Check firewall settings
```

**More solutions:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## File Structure

```
deepfake-detection/
├── backend/
│   ├── app/
│   │   ├── main.py              ← Main FastAPI app
│   │   ├── config.json          ← Configuration
│   │   ├── models/              ← Model implementations
│   │   ├── services/            ← Detection & Grad-CAM services
│   │   └── api/                 ← Video/webcam API
│   ├── efficientnet_b4.pth      ← Model weights
│   ├── xception.pth
│   ├── f3net.pth
│   ├── evaluate_model.py        ← Accuracy evaluation
│   ├── cross_dataset_eval.py    ← Cross-dataset testing
│   ├── start_server.bat         ← Windows startup
│   ├── start_server.sh          ← Linux/Mac startup
│   ├── requirements.txt         ← Python dependencies
│   └── venv/                    ← Virtual environment
└── frontend/
    ├── app/                     ← Next.js pages
    ├── package.json             ← Node dependencies
    └── node_modules/
```

---

## Next Steps

Once running:

1. ✅ **Test with sample images**
   - Upload via web UI
   - Check Grad-CAM visualization

2. ✅ **Run evaluation scripts**
   - Test accuracy with your dataset
   - Verify >90% accuracy goal

3. ✅ **Test video processing**
   - Upload a video via API
   - Check frame-by-frame results

4. ✅ **Run cross-dataset evaluation**
   - Test robustness across datasets
   - Generate comparison reports

5. ✅ **Review documentation**
   - [README.md](README.md) - Overview
   - [PROJECT_GOALS.md](PROJECT_GOALS.md) - Goals achievement
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details
   - [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

---

## Production Deployment

For production:

```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Build frontend
cd frontend
npm run build
npm start
```

---

## Summary

**Minimum steps to run:**
```bash
1. cd backend
2. venv\Scripts\activate  (Windows) or source venv/bin/activate (Linux/Mac)
3. python -m uvicorn app.main:app --reload
4. Visit: http://localhost:8000/docs
```

**For full experience:**
```bash
Terminal 1 (Backend):
  cd backend
  venv\Scripts\activate
  python -m uvicorn app.main:app --reload

Terminal 2 (Frontend):
  cd frontend
  npm run dev

Visit: http://localhost:3000
```

---

**Status:** Ready to run! 🚀
**Documentation:** See README.md for full details
**Support:** See TROUBLESHOOTING.md for help
