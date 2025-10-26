# Troubleshooting Guide

Common issues and solutions for the DeepFake Detection System.

## Backend Issues

### Issue: "ModuleNotFoundError: No module named 'services'" or "No module named 'app'"

**Problem:** You're running the server from the wrong directory.

**Solution:**
```bash
# CORRECT - Run from backend directory
cd backend
python -m uvicorn app.main:app --reload

# OR use the startup script
# Windows:
start_server.bat

# Linux/Mac:
chmod +x start_server.sh
./start_server.sh

# WRONG - Don't run from backend/app
cd backend/app
python -m uvicorn app.main:app --reload  # This will fail!
```

---

### Issue: "Config file not found"

**Problem:** The `config.json` file is missing or in the wrong location.

**Solution:**
```bash
# Config should be at: backend/app/config.json

# Create from template:
cd backend/app
cp config.example.json config.json  # If example exists

# Or create manually:
cat > config.json << EOF
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
EOF
```

---

### Issue: "Model weights not found"

**Problem:** Model weight files (.pth) are missing.

**Solution:**

**Option 1:** Disable models you don't have weights for
```json
{
  "models": {
    "xception": {
      "enabled": false,  // Disable if no weights
      ...
    }
  }
}
```

**Option 2:** Download pre-trained weights
- Place `.pth` files in `backend/` directory
- Update paths in `config.json`

**Option 3:** Train your own models
- See training scripts (if available)
- Or use transfer learning from timm

---

### Issue: Port 8000 already in use

**Solution:**

**Windows:**
```bash
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F

# Or use a different port
python -m uvicorn app.main:app --reload --port 8001
```

**Linux/Mac:**
```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python -m uvicorn app.main:app --reload --port 8001
```

---

### Issue: "CUDA out of memory" or "RuntimeError: CUDA error"

**Solution:**

**Option 1:** Use CPU instead
```json
{
  "device": "cpu"
}
```

**Option 2:** Reduce batch size (if using batching)

**Option 3:** Use fewer models
```json
{
  "models": {
    "effort": {
      "enabled": false  // Disable large models
    }
  }
}
```

---

### Issue: Virtual environment issues

**Solution:**

**Windows:**
```bash
# Delete old venv
rmdir /s venv

# Create new venv
python -m venv venv

# Activate
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# Delete old venv
rm -rf venv

# Create new venv
python -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Frontend Issues

### Issue: "npm install" fails

**Solution:**
```bash
# Clear cache
npm cache clean --force

# Delete node_modules
rm -rf node_modules
rm package-lock.json

# Reinstall
npm install
```

---

### Issue: Frontend can't connect to backend

**Problem:** CORS or network issues.

**Solution:**

1. **Check backend is running:**
```bash
curl http://localhost:8000/health
```

2. **Check CORS settings in backend:**
```python
# In app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

3. **Check firewall:**
- Windows: Allow Python through firewall
- Linux: Check `ufw` or `iptables`

4. **Check API URL in frontend:**
- Should be: `http://localhost:8000`
- Not: `http://127.0.0.1:8000` (might cause issues)

---

### Issue: "Next.js build fails"

**Solution:**
```bash
# Clear Next.js cache
rm -rf .next

# Rebuild
npm run build

# Or just run dev mode
npm run dev
```

---

## Model Issues

### Issue: Low accuracy (<90%)

**Possible causes:**

1. **Using wrong weights:**
   - Ensure weights are for deepfake detection, not general classification

2. **Wrong preprocessing:**
   - Check image normalization
   - Verify input size (224x224 or 299x299)

3. **Not using ensemble:**
   - Enable multiple models in config.json
   - Use weighted ensemble

4. **Data quality:**
   - Ensure face is clearly visible
   - Check image quality/resolution

**Solution:**
```bash
# Test with evaluation script
cd backend
python evaluate_model.py

# Check individual model performance
# Verify config.json settings
```

---

### Issue: Grad-CAM not working

**Problem:** Grad-CAM fails to generate or shows errors.

**Solution:**

1. **Check if enabled:**
```bash
curl -X POST "http://localhost:8000/api/detect/image?generate_heatmap=true" \
  -F "file=@image.jpg"
```

2. **Check model compatibility:**
- Not all models support Grad-CAM
- Some require specific layer selection

3. **Fallback:**
```python
# In services/detection.py
generate_heatmap=False  # Disable if causing issues
```

---

## Performance Issues

### Issue: Slow inference (>5s per image)

**Solutions:**

1. **Use GPU:**
```json
{
  "device": "cuda"
}
```

2. **Reduce models:**
```json
{
  "models": {
    "effort": {"enabled": false}  // Disable slowest model
  }
}
```

3. **Optimize preprocessing:**
- Reduce image size before processing
- Use efficient data loading

4. **Batch processing:**
- Process multiple images at once
- Amortize overhead

---

### Issue: High memory usage

**Solutions:**

1. **Use CPU:**
```json
{"device": "cpu"}
```

2. **Reduce models:**
- Disable Effort model (304M params)
- Keep only 1-2 models active

3. **Clear cache:**
```python
import torch
torch.cuda.empty_cache()  # If using GPU
```

---

## Testing Issues

### Issue: Evaluation script fails

**Solution:**

1. **Check dataset structure:**
```
dataset/
├── real/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── fake/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

2. **Start with small sample:**
```python
python evaluate_model.py
# When prompted, enter path and use max_images=10
```

3. **Check dependencies:**
```bash
pip install scikit-learn matplotlib seaborn
```

---

### Issue: Cross-dataset evaluation fails

**Solution:**

1. **Configure datasets properly:**
```python
datasets = {
    'dataset1': {
        'path': '/full/path/to/dataset',
        'real_subdir': 'real',
        'fake_subdir': 'fake',
        'max_images': 50  # Start small
    }
}
```

2. **Check paths are absolute:**
```python
from pathlib import Path
path = Path('dataset').resolve()  # Convert to absolute
```

---

## Video/Webcam Issues

### Issue: Video processing fails

**Solution:**

1. **Install video dependencies:**
```bash
pip install opencv-python imageio imageio-ffmpeg
```

2. **Check video format:**
- Supported: MP4, AVI, MOV
- Try converting to MP4 if issues

3. **Reduce frame processing:**
```bash
curl -X POST "http://localhost:8000/api/video/detect" \
  -F "file=@video.mp4" \
  -F "frame_skip=10"  # Process fewer frames
```

---

### Issue: WebSocket connection fails

**Solution:**

1. **Check WebSocket support:**
```bash
pip install websockets
```

2. **Test connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/video/ws/webcam');
ws.onopen = () => console.log('Connected!');
ws.onerror = (e) => console.error('Error:', e);
```

3. **Check firewall:**
- Allow WebSocket connections
- Port 8000 should be open

---

## Getting Help

If none of these solutions work:

1. **Check logs:**
```bash
# Backend logs show detailed errors
python -m uvicorn app.main:app --reload
```

2. **Enable debug mode:**
```python
# In app/main.py
app = FastAPI(..., debug=True)
```

3. **Test individual components:**
```bash
# Test model loading
python -c "from app.models.manager import EnsembleModelManager; m = EnsembleModelManager()"

# Test service
python -c "from app.services.detection import EnsembleDetectionService; s = EnsembleDetectionService()"
```

4. **Check documentation:**
- [README.md](README.md)
- [QUICKSTART.md](QUICKSTART.md)
- [PROJECT_GOALS.md](PROJECT_GOALS.md)

5. **Create an issue:**
- Include error messages
- Include system info (OS, Python version)
- Include steps to reproduce

---

## Quick Diagnostic Commands

```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip list | grep -E "torch|fastapi|uvicorn|pillow"

# Test imports
python -c "import torch; import fastapi; import PIL; print('OK')"

# Test backend
curl http://localhost:8000/health

# Check config
cat backend/app/config.json

# Check models
ls backend/*.pth

# Test model loading
cd backend
python -c "from app.models.manager import EnsembleModelManager; print('Models OK')"
```

---

## Still Having Issues?

Create a detailed issue report with:

1. **Error message** (full traceback)
2. **System info:**
   ```bash
   python --version
   pip list
   uname -a  # Linux/Mac
   systeminfo  # Windows
   ```
3. **Steps to reproduce**
4. **Expected vs actual behavior**
5. **Relevant config files**

Good luck! 🚀
