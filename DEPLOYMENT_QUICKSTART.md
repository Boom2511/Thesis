# Deployment Quick Start Guide

**Target**: Deploy to Vercel (frontend) + Render (backend)
**Time Required**: 60-90 minutes
**Cost**: $7-25/month minimum

---

## ⚠️ CRITICAL WARNINGS

### 1. Git LFS Bandwidth Limit

**Your Model Files**: 1.44 GB total
- xception_best.pth: 84 MB
- f3net_best.pth: 87 MB
- effort_clip_L14_trainOn_FaceForensic.pth: 1.2 GB
- effnb4_best.pth: 68 MB (disabled)

**GitHub Free Tier**:
- Storage: 1 GB (YOU EXCEED THIS!)
- Bandwidth: 1 GB/month

**⚠️ YOUR MODELS EXCEED FREE TIER STORAGE!**

**Solutions**:
1. **Upgrade GitHub** to Pro ($4/month) - includes 50 GB LFS
2. **Use external storage** (AWS S3, Google Cloud) - ~$1/month
3. **Upload directly to Render** via SSH (no git)

**Recommended**: Option 1 (GitHub Pro) - easiest

---

### 2. Render Free Tier Won't Work

**Your Requirements**:
- RAM: 1.5-2 GB (3 models + PyTorch)
- First request: 30-60 seconds (model loading)
- CPU: Sustained usage

**Free Tier Limits**:
- RAM: 512 MB ❌
- Timeout: 30 seconds ❌
- CPU: Sleeps after inactivity ❌

**Minimum Plan**: Starter ($7/month)
**Recommended**: Standard ($25/month) - 2 GB RAM

---

## Pre-Deployment Checklist

Run through [PRE_DEPLOYMENT_CHECKLIST.md](PRE_DEPLOYMENT_CHECKLIST.md) first!

**Critical items**:
- [ ] .gitignore updated (no secrets!)
- [ ] .env.example files created
- [ ] Git LFS configured
- [ ] GitHub LFS bandwidth sufficient
- [ ] Render.yaml created
- [ ] vercel.json created
- [ ] Tested locally

---

## Step-by-Step Deployment

### Phase 1: Prepare Repository (15 minutes)

#### 1.1 Clean Repository

```powershell
# Remove cache and build files
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
Remove-Item -Recurse -Force backend/mlruns -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force frontend/node_modules -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force frontend/.next -ErrorAction SilentlyContinue
```

#### 1.2 Verify Git LFS

```powershell
# Check LFS files
git lfs ls-files

# Should show:
# - xception_best.pth
# - f3net_best.pth
# - effort_clip_L14_trainOn_FaceForensic.pth
# - effnb4_best.pth
```

#### 1.3 Check for Secrets

```powershell
# Search for common secrets
git grep -i "password"
git grep -i "api_key"
git grep -i "secret"

# Should return nothing sensitive!
```

#### 1.4 Verify Configuration Files

```powershell
# Check files exist
ls render.yaml
ls frontend/vercel.json
ls backend/.env.example
ls frontend/.env.local.example
```

---

### Phase 2: Push to GitHub (10 minutes)

#### 2.1 Stage All Files

```powershell
git add .
```

#### 2.2 Review Changes

```powershell
git status
git diff --cached --name-only
```

**Verify NOT staging**:
- .env files (should be in .gitignore)
- node_modules/
- mlruns/
- __pycache__/
- .next/

#### 2.3 Commit and Push

```powershell
git commit -m "chore: prepare for deployment

- Add render.yaml for Render deployment
- Add vercel.json for Vercel deployment
- Update .gitignore for production
- Add environment variable templates
- Fix all critical bugs (classifier loading, frontend display)
- Create comprehensive documentation

Ready for production deployment."

git push origin main
```

#### 2.4 Verify LFS Push

```powershell
git lfs ls-files
# All files should show * (uploaded)

git lfs status
# Should show: "Objects to be pushed to origin/main: (none)"
```

---

### Phase 3: Deploy Backend to Render (30-45 minutes)

#### 3.1 Create Render Account

1. Go to: https://render.com
2. Sign up (free account to start)
3. Connect GitHub account

#### 3.2 Create Web Service

1. Dashboard → **New +** → **Web Service**
2. Select: `deepfake-detection` repository
3. Click: **Connect**

#### 3.3 Configure Service

**Auto-detected from render.yaml:**
- Name: deepfake-detection-backend
- Build Command: (from yaml)
- Start Command: (from yaml)

**Manual settings:**
- **Plan**: Starter ($7/month) minimum
- **Region**: Oregon (or closest to you)

#### 3.4 Add Environment Variables

Render should auto-detect from render.yaml, but verify:

```
MODEL_DEVICE=cpu
ENABLE_MLFLOW=false
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000
```

Note: Will update ALLOWED_ORIGINS after frontend deployed

#### 3.5 Add Persistent Disk

**Important**: Verify disk configured:
- Name: model-weights-storage
- Mount Path: `/opt/render/project/src/backend/app/models/weights`
- Size: 10 GB

#### 3.6 Deploy

1. Click: **Create Web Service**
2. Wait for build (20-40 minutes first time)

**Monitor logs for**:
```
[OK] Xception model loaded
[OK] F3Net model loaded
[OK] Effort-CLIP model loaded
[OK] Loaded 3 models
Service ready!
```

#### 3.7 Test Backend

Get your URL: `https://your-service-name.onrender.com`

```powershell
# Test health endpoint
curl https://your-service-name.onrender.com/health

# Expected:
# {"status":"healthy","models_loaded":3,"device":"cpu"}
```

---

### Phase 4: Deploy Frontend to Vercel (15 minutes)

#### 4.1 Create Vercel Account

1. Go to: https://vercel.com
2. Sign up with GitHub
3. Authorize Vercel

#### 4.2 Import Project

1. Dashboard → **Add New...** → **Project**
2. Find: `deepfake-detection`
3. Click: **Import**

#### 4.3 Configure Project

**Root Directory**:
- Click **Edit**
- Set: `frontend`
- Click **Continue**

**Framework**: Next.js (auto-detected)

#### 4.4 Add Environment Variables

Click **Environment Variables** tab.

**Add these** (replace with YOUR Render URL):

| Name | Value |
|------|-------|
| `NEXT_PUBLIC_API_URL` | `https://your-service-name.onrender.com` |
| `NEXT_PUBLIC_WS_URL` | `wss://your-service-name.onrender.com` |

**Important**:
- HTTP → https://
- WebSocket → wss://
- No trailing slash!

#### 4.5 Deploy

1. Click: **Deploy**
2. Wait for build (5-10 minutes)

**Monitor for errors**

#### 4.6 Get Frontend URL

Your URL: `https://your-project-name.vercel.app`

---

### Phase 5: Connect Frontend & Backend (5 minutes)

#### 5.1 Update Backend CORS

1. Go to Render dashboard
2. Select your backend service
3. Click **Environment** tab
4. Find `ALLOWED_ORIGINS`
5. Update to:
   ```
   https://your-project-name.vercel.app,https://*.vercel.app
   ```
6. Save → Auto-redeploys

#### 5.2 Wait for Redeploy

Render will automatically redeploy (2-5 minutes)

---

### Phase 6: Test Production (10 minutes)

#### 6.1 Open Frontend

Visit: `https://your-project-name.vercel.app`

#### 6.2 Test Image Upload

1. Upload a test image
2. Wait for processing (10-30 seconds first request)
3. Verify:
   - ✅ Image uploads successfully
   - ✅ Detection completes
   - ✅ Results display (NOT 1%!)
   - ✅ Heatmap shows correctly
   - ✅ No CORS errors in console

#### 6.3 Check Browser Console

Press F12 → Console

**Should NOT see**:
- ❌ CORS errors
- ❌ 404 errors
- ❌ Network errors

**OK to see**:
- ✅ Slow response time (CPU is slow)
- ✅ Loading states

#### 6.4 Test on Mobile

1. Open on phone: `https://your-project-name.vercel.app`
2. Verify:
   - ✅ Layout responsive
   - ✅ Images not cut off
   - ✅ Detection works

---

## Post-Deployment Tasks

### 1. Update Documentation

Update [README.md](README.md) with:
- ✅ Live demo URL
- ✅ Deployment status
- ✅ Known limitations

### 2. Set Up Monitoring

**Render**:
- Check: Metrics tab (CPU, memory, requests)
- Set up: Alerts for high memory usage

**Vercel**:
- Check: Analytics tab
- Monitor: Page load times

### 3. Custom Domain (Optional)

**Vercel** (frontend):
1. Project Settings → Domains
2. Add your domain
3. Configure DNS

**Render** (backend):
1. Service Settings → Custom Domains
2. Add your domain
3. Configure DNS

**Don't forget**: Update CORS after adding custom domains!

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Backend build fails | Upgrade to Starter plan ($7/month) |
| Models not found | Check Git LFS, verify disk mount path |
| CORS errors | Update ALLOWED_ORIGINS on Render |
| Frontend shows 1% | Check environment variables, redeploy |
| Timeout on first request | Normal - models loading (30-60s) |
| Slow performance | Expected on CPU (3-10x slower than GPU) |
| Out of memory | Upgrade to Standard plan ($25/month) |

---

## Cost Summary

### Minimum Setup (Testing/Low Traffic)
- GitHub: Free (but LFS limited - YOU EXCEED!)
- GitHub Pro: $4/month (recommended - includes LFS)
- Render Starter: $7/month
- Vercel: Free

**Total**: $11/month

### Recommended Production Setup
- GitHub Pro: $4/month (for LFS)
- Render Standard: $25/month (2 GB RAM)
- Vercel Pro: $20/month (analytics, support)

**Total**: $49/month

### Budget Option
- GitHub Pro: $4/month
- Render Starter: $7/month (tight but works)
- Vercel: Free

**Total**: $11/month

---

## Performance Expectations

### Backend (Render Starter - CPU)
- First request: 30-60 seconds (model loading)
- Subsequent requests: 2-5 seconds per image
- Accuracy: ~95% (same as local)
- Limitation: CPU only (no GPU)

### Frontend (Vercel)
- Page load: 1-3 seconds
- Image upload: Instant
- Results display: Depends on backend

### Overall User Experience
- Upload → Results: 5-10 seconds (after initial load)
- Mobile: Fully responsive
- Concurrent users: 10-20 (Starter plan)

---

## Next Steps

1. ✅ Both services deployed
2. ✅ Frontend ↔ Backend connected
3. ✅ End-to-end tested

**Now**:
- Monitor performance (first 24 hours)
- Check error logs daily
- Optimize if needed
- Consider upgrades based on usage

---

## Support

**Detailed guides**:
- Backend: [DEPLOYMENT_GUIDE_RENDER.md](DEPLOYMENT_GUIDE_RENDER.md)
- Frontend: [DEPLOYMENT_GUIDE_VERCEL.md](DEPLOYMENT_GUIDE_VERCEL.md)
- Full checklist: [PRE_DEPLOYMENT_CHECKLIST.md](PRE_DEPLOYMENT_CHECKLIST.md)

**External resources**:
- Render: https://render.com/docs
- Vercel: https://vercel.com/docs
- Next.js: https://nextjs.org/docs

---

**Last Updated**: 2025-10-19
**Status**: Ready to deploy 🚀
