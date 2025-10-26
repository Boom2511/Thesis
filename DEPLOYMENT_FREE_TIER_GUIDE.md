# 100% Free Deployment Guide

**Goal**: Deploy your deepfake detection system with $0/month cost
**Challenge**: Your models are 1.44 GB (exceeds GitHub free tier)
**Solution**: Use alternative free platforms and external storage

---

## 🎯 Free Deployment Strategy

### Option 1: Hugging Face + Vercel (Recommended)
- **Backend**: Hugging Face Spaces (FREE, includes GPU!)
- **Frontend**: Vercel (FREE)
- **Models**: Stored in Hugging Face (FREE, unlimited)
- **Total Cost**: $0/month ✅

### Option 2: GitHub + Render Free + Vercel
- **Backend**: Render Free Tier (LIMITED but possible)
- **Frontend**: Vercel (FREE)
- **Models**: Google Drive or Dropbox (FREE)
- **Total Cost**: $0/month ✅

### Option 3: Railway + Vercel
- **Backend**: Railway (FREE $5 credit/month)
- **Frontend**: Vercel (FREE)
- **Models**: Railway storage
- **Total Cost**: $0/month ✅

---

## 🚀 Option 1: Hugging Face Spaces (RECOMMENDED)

### Why Hugging Face?
- ✅ **FREE** GPU access (2 vCPU, 16 GB RAM, T4 GPU)
- ✅ No model size limits
- ✅ Built for ML models
- ✅ Persistent storage included
- ✅ Automatic HTTPS
- ✅ No credit card required

### Limitations:
- ⚠️ May sleep after inactivity (restarts in 30s)
- ⚠️ Public by default (can make private with Pro account)

---

## Step-by-Step: Deploy to Hugging Face Spaces

### Part 1: Prepare for Hugging Face

#### 1.1 Create Hugging Face Account
1. Go to: https://huggingface.co/join
2. Sign up (FREE, no credit card)
3. Verify email

#### 1.2 Create Space
1. Go to: https://huggingface.co/spaces
2. Click: **Create new Space**
3. Configure:
   - **Name**: `deepfake-detection`
   - **License**: Apache 2.0
   - **Space SDK**: Gradio (or Docker)
   - **Visibility**: Public (or Private with HF Pro)
   - **Hardware**: CPU Basic (FREE) or upgrade to T4 GPU (FREE with grant)

#### 1.3 Get Free GPU Access
1. Apply for community GPU grant (usually instant approval)
2. Or start with CPU (FREE, slower but works)

---

### Part 2: Convert Backend to Gradio

Create `backend/app_gradio.py`:

```python
import gradio as gr
import torch
from PIL import Image
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from models.manager import EnsembleModelManager
from services.detection import detect_face_and_crop

# Initialize models
print("Loading models...")
model_manager = EnsembleModelManager("app/config.json")
print("Models loaded!")

def predict_deepfake(image):
    """
    Predict if image is real or fake

    Args:
        image: PIL Image or numpy array

    Returns:
        tuple: (label, confidence, heatmap)
    """
    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Detect face
        face_img = detect_face_and_crop(image)
        if face_img is None:
            return "No face detected", None, None

        # Convert to tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img_tensor = transform(face_img).unsqueeze(0)

        # Get prediction
        result = model_manager.predict_ensemble(img_tensor)

        # Extract results
        ensemble = result["ensemble"]
        is_fake = ensemble["prediction"] == "FAKE"
        confidence = ensemble["confidence"]

        # Format output
        label = f"{'FAKE' if is_fake else 'REAL'} ({confidence*100:.1f}% confidence)"

        # Individual models
        individual_results = []
        for model_name, pred in result["individual"].items():
            fake_prob = pred["fake_prob"]
            individual_results.append(f"{model_name}: {fake_prob*100:.1f}% FAKE")

        details = "\n".join(individual_results)

        # TODO: Generate heatmap if needed
        heatmap = None

        return label, details, heatmap

    except Exception as e:
        return f"Error: {str(e)}", None, None

# Create Gradio interface
demo = gr.Interface(
    fn=predict_deepfake,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Individual Models"),
        gr.Image(label="Heatmap (Coming Soon)", visible=False)
    ],
    title="Deepfake Detection System",
    description="""
    Upload an image to detect if it's a deepfake or real.

    **Models**: Xception + F3Net + Effort-CLIP ensemble
    **Accuracy**: ~95% on FaceForensics++

    ⚠️ First prediction may take 30-60 seconds (loading models)
    """,
    examples=[
        # Add example images here
    ],
    article="""
    ### About
    This is an ensemble deepfake detection system using three state-of-the-art models:
    - **Xception**: Depthwise separable convolutions
    - **F3Net**: Frequency-aware feature extraction
    - **Effort-CLIP**: Vision Transformer with CLIP backbone

    ### How it works
    1. Face detection with MTCNN
    2. Preprocessing (resize, normalize)
    3. Inference with 3 models
    4. Weighted ensemble prediction

    ### Limitations
    - Requires clear face in image
    - Works best with single face
    - May be fooled by very recent deepfake techniques
    """
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
```

---

### Part 3: Create Hugging Face Configuration

Create `app.py` in project root:

```python
"""
Hugging Face Spaces entry point
"""
import sys
sys.path.append('backend')

from backend.app_gradio import demo

# Launch
demo.launch()
```

Create `requirements_hf.txt`:

```txt
torch==2.1.0
torchvision==0.16.0
gradio==4.12.0
pillow==10.2.0
facenet-pytorch==2.5.3
timm==1.0.3
opencv-python-headless==4.9.0.80
numpy==1.24.3
```

Create `README.md` in root (Hugging Face card):

```markdown
---
title: Deepfake Detection
emoji: 🔍
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.12.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Deepfake Detection System

Ensemble of 3 state-of-the-art models for deepfake detection.

## Models
- Xception (84 MB)
- F3Net (87 MB)
- Effort-CLIP (1.2 GB)

## Accuracy
~95% on FaceForensics++ dataset
```

---

### Part 4: Push to Hugging Face

```powershell
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detection
cd deepfake-detection

# Copy files
Copy-Item -Path ../deepfake-detection/backend -Destination . -Recurse
Copy-Item -Path ../deepfake-detection/app.py -Destination .
Copy-Item -Path ../deepfake-detection/requirements_hf.txt -Destination requirements.txt

# Commit and push
git add .
git commit -m "Initial commit"
git push
```

**Hugging Face will automatically build and deploy!**

---

### Part 5: Deploy Frontend to Vercel (FREE)

Update frontend to point to Hugging Face:

**frontend/.env.local**:
```env
NEXT_PUBLIC_API_URL=https://YOUR_USERNAME-deepfake-detection.hf.space
```

Then deploy to Vercel (same process as before, completely FREE).

---

## 🚀 Option 2: Render Free + Google Drive

### Strategy
- Use Render Free Tier (512 MB RAM - TIGHT!)
- Store models on Google Drive (FREE 15 GB)
- Download models on startup
- Use only 1-2 models (reduce RAM)

### Part 1: Upload Models to Google Drive

1. Go to: https://drive.google.com
2. Create folder: `deepfake-models`
3. Upload model files:
   - xception_best.pth (84 MB)
   - f3net_best.pth (87 MB)
   - effort_clip (skip - too large for free tier)

4. Get shareable links:
   - Right-click file → Share → Anyone with link
   - Copy link ID from URL: `https://drive.google.com/file/d/FILE_ID/view`

### Part 2: Create Download Script

Create `backend/download_models.py`:

```python
import os
import gdown
from pathlib import Path

# Google Drive file IDs
MODELS = {
    "xception_best.pth": "YOUR_XCEPTION_FILE_ID",
    "f3net_best.pth": "YOUR_F3NET_FILE_ID",
}

def download_models():
    weights_dir = Path(__file__).parent / "app/models/weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    for filename, file_id in MODELS.items():
        filepath = weights_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(filepath), quiet=False)
            print(f"Downloaded {filename}")
        else:
            print(f"{filename} already exists")

if __name__ == "__main__":
    download_models()
```

Add to `requirements.txt`:
```txt
gdown==5.1.0
```

### Part 3: Modify config.json for Free Tier

**backend/app/config.json** (USE ONLY 2 MODELS):

```json
{
  "models": {
    "xception": {
      "weight": 0.5,
      "enabled": true
    },
    "efficientnet_b4": {
      "weight": 0.0,
      "enabled": false
    },
    "f3net": {
      "weight": 0.5,
      "enabled": true
    },
    "effort": {
      "weight": 0.0,
      "enabled": false
    }
  }
}
```

**Why**: 2 smaller models = ~400 MB RAM, fits in 512 MB (barely)

### Part 4: Update render.yaml for Free Tier

```yaml
services:
  - type: web
    name: deepfake-detection-backend
    runtime: python
    plan: free  # FREE!
    region: oregon

    buildCommand: |
      pip install --upgrade pip &&
      pip install -r backend/requirements.txt &&
      python backend/download_models.py

    startCommand: cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1

    healthCheckPath: /health
    autoDeploy: true

    envVars:
      - key: PYTHON_VERSION
        value: "3.10"
      - key: MODEL_DEVICE
        value: cpu
      - key: ENABLE_MLFLOW
        value: "false"
      - key: ALLOWED_ORIGINS
        value: https://*.vercel.app
```

**No persistent disk** (not available on free tier)
**Models download on each startup** (takes 2-3 minutes)

---

## 🚀 Option 3: Railway (FREE $5/month credit)

### Why Railway?
- ✅ $5 FREE credit per month (enough for low traffic)
- ✅ 512 MB RAM (can upgrade to 2 GB within free credit)
- ✅ Persistent storage (500 MB free)
- ✅ Better performance than Render free tier
- ✅ No sleep after inactivity

### Limitations:
- ⚠️ Credit card required (but not charged unless you exceed $5)
- ⚠️ After $5, service stops (safe)

### Deployment to Railway

1. Go to: https://railway.app
2. Sign up with GitHub (FREE)
3. Add credit card (optional, for over $5)
4. Click: **New Project** → **Deploy from GitHub repo**
5. Select: `deepfake-detection`
6. Configure:
   - Root directory: `backend`
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

7. Add environment variables (same as Render)

8. Add volume for models:
   - Storage: 500 MB (FREE)
   - Mount path: `/app/models/weights`

9. Deploy!

**Cost**: FREE (stays under $5/month for low traffic)

---

## 📊 Comparison Table

| Feature | Hugging Face | Render Free | Railway |
|---------|--------------|-------------|---------|
| Cost | $0 | $0 | $0 (up to $5) |
| RAM | 16 GB | 512 MB | 512 MB - 8 GB |
| GPU | Yes (FREE!) | No | No |
| Storage | Unlimited | Ephemeral | 500 MB |
| Sleep | Yes (30s wake) | Yes (instant wake) | No |
| Credit Card | No | No | Optional |
| Best For | ML Apps | Static APIs | Dynamic Apps |

---

## 🎯 Recommended Approach

### For Best Performance (FREE):
**Use Hugging Face Spaces**
- ✅ FREE GPU access
- ✅ Built for ML
- ✅ No model size limits
- ✅ No credit card needed

### For Traditional Setup (FREE):
**Use Railway + Vercel**
- ✅ More control
- ✅ Better than Render free tier
- ✅ Persistent storage
- ⚠️ Credit card required (but not charged)

### For No Credit Card (FREE):
**Use Render Free + Google Drive + Vercel**
- ✅ No credit card
- ✅ Completely free
- ⚠️ Very limited (2 models only)
- ⚠️ Slow startup (downloads models each time)

---

## 🚀 Quick Start: Hugging Face (Easiest)

```powershell
# 1. Create backend/app_gradio.py (see above)

# 2. Create app.py in root (see above)

# 3. Create requirements_hf.txt (see above)

# 4. Install HF CLI
pip install huggingface_hub

# 5. Login
huggingface-cli login

# 6. Create space on HF website
# https://huggingface.co/spaces

# 7. Clone and push
git clone https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detection hf-space
cd hf-space

# Copy files
Copy-Item -Path ../backend -Destination . -Recurse
Copy-Item -Path ../app.py -Destination .
Copy-Item -Path ../requirements_hf.txt -Destination requirements.txt

# Push
git add .
git commit -m "Deploy to Hugging Face"
git push

# 8. Deploy frontend to Vercel (FREE)
# Update NEXT_PUBLIC_API_URL to your HF Space URL
```

**Total time**: 30 minutes
**Total cost**: $0/month forever

---

## ⚡ Performance Comparison

### Hugging Face (FREE GPU):
- First request: 10-20 seconds
- Subsequent: 0.5-1 second per image
- **FASTER THAN LOCAL!** ✅

### Railway (FREE $5 credit):
- First request: 30-60 seconds
- Subsequent: 2-5 seconds per image
- Similar to local CPU

### Render Free (512 MB):
- First request: 60-120 seconds
- Subsequent: 5-10 seconds per image
- May timeout on first request
- Only 2 models work

---

## 🎯 My Recommendation for You

**Best option**: Hugging Face Spaces + Vercel

**Why**:
1. ✅ **100% FREE** forever (no credit card needed)
2. ✅ **FREE GPU** (faster than your local CPU!)
3. ✅ No model size limits (all 3 models work)
4. ✅ Built for ML applications
5. ✅ Easy to deploy (Gradio interface)
6. ✅ Automatic HTTPS
7. ✅ Can get free GPU access with community grant

**Downsides**:
- ⚠️ May sleep after inactivity (wakes in 30s)
- ⚠️ Public by default (can upgrade to private for $9/month if needed)

---

## 📝 Next Steps

1. Choose deployment option (I recommend Hugging Face)
2. Follow the guide for that option
3. Deploy in 30-60 minutes
4. Enjoy your FREE deepfake detector!

---

**Total Cost**: $0/month ✅
**Credit Card Required**: No (for Hugging Face) ✅
**Performance**: Better than local (with GPU) ✅

---

Let me know which option you want to proceed with and I'll help you deploy!
