# 🚀 Deployment Guide

## 📋 Overview

This guide covers 3 deployment options:

1. **Hugging Face Spaces** (Recommended) - Free GPU, easy setup
2. **Vercel + Render** - Frontend + Backend separate
3. **Local Deployment** - Development/testing

---

## 🎯 Option 1: Hugging Face Spaces (Recommended)

**Best for:** Production deployment with FREE GPU access

### Prerequisites

- Hugging Face account
- Git LFS installed
- Model weights (included in repo)

### Steps

#### 1. Create New Space

```bash
# Go to https://huggingface.co/spaces
# Click "Create new Space"
# - Name: deepfake-detection
# - License: MIT
# - SDK: Gradio
# - Hardware: CPU Basic (upgrade to GPU if needed)
```

#### 2. Clone Space

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detection
cd deepfake-detection
```

#### 3. Copy Files

```bash
# Copy Gradio app
cp ../deepfake-detection/app_gradio.py app.py

# Copy backend
cp -r ../deepfake-detection/backend .

# Copy requirements
cp ../deepfake-detection/requirements.txt .
```

#### 4. Configure Git LFS

```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
```

#### 5. Push to Space

```bash
git add .
git commit -m "Deploy optimized ensemble (92.86% accuracy)"
git push
```

#### 6. Wait for Build

- Space will automatically build and deploy
- Check logs for any errors
- Should be live in ~5-10 minutes

#### 7. Test

Visit: `https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detection`

### Upgrade to GPU (Optional)

For faster inference:

1. Go to Space Settings
2. Change Hardware: **CPU Basic** → **T4 small** (free) or **A10G small** (paid)
3. Restart Space

---

## 🌐 Option 2: Vercel (Frontend) + Render (Backend)

**Best for:** Separate frontend/backend deployment

### Frontend (Vercel)

#### 1. Install Vercel CLI

```bash
npm i -g vercel
```

#### 2. Deploy Frontend

```bash
cd frontend
vercel
```

Follow prompts:
- Project name: `deepfake-detection-frontend`
- Framework: Next.js
- Deploy: Yes

#### 3. Set Environment Variable

In Vercel dashboard:
- Settings → Environment Variables
- Add: `NEXT_PUBLIC_API_URL` = `https://your-backend-url.onrender.com`

#### 4. Redeploy

```bash
vercel --prod
```

### Backend (Render)

#### 1. Create Account

Go to: https://render.com

#### 2. New Web Service

- Connect GitHub repo
- Select `backend` directory
- Name: `deepfake-detection-backend`

#### 3. Configure

```yaml
Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

#### 4. Add Environment Variables

```
PYTHON_VERSION=3.10
```

#### 5. Deploy

Click "Create Web Service"

**Note:** Free tier has:
- ⚠️ Slow startup (30-60 seconds)
- ⚠️ Spins down after 15 min inactivity
- ⚠️ No GPU

---

## 💻 Option 3: Local Deployment

### Development

```bash
# Terminal 1: Backend
cd backend
python -m uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

### Production (Local)

```bash
# Backend
cd backend
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Frontend
cd frontend
npm run build
npm start
```

---

## 🔧 Configuration

### Update API URL

**Frontend (.env.local):**
```bash
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

**Or hardcode in:** `frontend/app/page.tsx`

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
```

### Optimize Model Loading

**For faster startup, disable unused models:**

Edit `backend/app/config.json`:

```json
{
  "models": {
    "xception": { "enabled": true, "weight": 0.30 },
    "f3net": { "enabled": false, "weight": 0.00 },  // Disable if needed
    "effort": { "enabled": true, "weight": 0.70 }
  }
}
```

**Recommended for deployment:**
- Keep: Xception + Effort-CLIP (still 90%+ accuracy)
- Disable: F3Net (if memory/startup time is issue)

---

## 📊 Performance Comparison

| Platform | Startup Time | Inference Time | Cost | GPU |
|----------|--------------|----------------|------|-----|
| **HF Spaces (CPU)** | ~30s | ~3-5s | Free | ❌ |
| **HF Spaces (GPU)** | ~30s | ~1-2s | Free* | ✅ |
| **Render (Free)** | 30-60s | ~5-10s | Free | ❌ |
| **Vercel + Render** | Fast (FE) | ~5-10s | Free | ❌ |
| **Local (GPU)** | Instant | ~1s | Hardware | ✅ |

*GPU quota applies

---

## 🐛 Troubleshooting

### Hugging Face Spaces

**Build fails:**
- Check `requirements.txt` versions
- Ensure Git LFS is tracking `.pth` files
- Check logs in Space settings

**Out of memory:**
- Upgrade to GPU hardware
- Or disable F3Net model

**Slow inference:**
- Upgrade to GPU (T4 small is free!)
- Reduce image size in preprocessing

### Render

**Startup timeout:**
- Normal for free tier
- Wait 60 seconds before first request
- Consider upgrading to paid tier

**Models not loading:**
- Check Git LFS files uploaded
- Verify `backend/app/models/weights/` exists
- Check file sizes (should be ~1.3GB total)

### Vercel

**API calls failing:**
- Check `NEXT_PUBLIC_API_URL` environment variable
- Enable CORS in backend
- Check backend logs

---

## ✅ Verification Checklist

After deployment:

- [ ] Upload test image → Returns prediction
- [ ] Prediction confidence > 50%
- [ ] Individual model predictions shown
- [ ] Grad-CAM heatmap generated (if enabled)
- [ ] Processing time < 5 seconds
- [ ] Language switch works (EN/TH)
- [ ] Batch mode works (if needed)
- [ ] Video mode works (if needed)

---

## 📈 Next Steps

1. **Monitor Performance:**
   - Check inference times
   - Monitor error rates
   - Track user feedback

2. **Optimize:**
   - Enable GPU if slow
   - Cache model loading
   - Reduce image preprocessing

3. **Scale:**
   - Upgrade hardware if needed
   - Add load balancing
   - Consider CDN for frontend

---

## 📧 Support

Issues? Check:
- [README.md](README.md)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- GitHub Issues

---

**Happy Deploying! 🚀**
