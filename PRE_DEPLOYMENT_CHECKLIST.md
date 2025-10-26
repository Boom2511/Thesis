# Pre-Deployment Checklist

**Last Updated**: 2025-10-19
**Target Platforms**:
- Frontend: Vercel
- Backend: Render.com

---

## CRITICAL: Before Pushing to GitHub

### 1. Security & Sensitive Files

#### ✅ Check .gitignore (MUST DO FIRST)

**Current Status**: .gitignore needs updates

**Action Required**: Update .gitignore to include:

```bash
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Virtual Environment
venv/
.venv/
env/
.env/
ENV/
env.bak/
venv.bak/

# Environment Variables (CRITICAL - DO NOT COMMIT)
.env
.env.*
.env.local
.env.production
.env.development
*.env

# IDE/Editor
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Node.js / Frontend
node_modules/
.next/
out/
build/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.pnpm-debug.log*

# MLflow
mlruns/
mlartifacts/

# Jupyter Notebook
.ipynb_checkpoints/

# pytest
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
desktop.ini

# Temporary files
*.tmp
*.temp
*.bak
*.swp
*~

# Model weights (already in Git LFS - keep this comment)
# *.pth files are tracked by Git LFS (see .gitattributes)

# Documentation (KEEP - these should be committed)
# *.md files should be committed
```

#### ⚠️ CRITICAL: Remove Secrets (if any)

**Check for secrets in code:**

```powershell
# Search for common secret patterns
git grep -i "password"
git grep -i "secret"
git grep -i "api_key"
git grep -i "token"
git grep -E "[0-9]+-[0-9A-Za-z_]{40}" # GitHub tokens
```

**If found:**
1. Remove from code
2. Move to environment variables
3. Add to .env.example (with dummy values)

#### 🔒 Verify No Credentials in Git History

```powershell
# Check if any .env files were previously committed
git log --all --full-history -- "*.env*"

# If found, you MUST clean git history (dangerous!)
# Consider: git filter-repo or BFG Repo-Cleaner
```

---

### 2. Git LFS Configuration

#### ✅ Current Status: Git LFS Configured

**Tracked files:**
- backend/app/models/weights/effnb4_best.pth (622 MB)
- backend/app/models/weights/effort_clip_L14_trainOn_FaceForensic.pth (1.2 GB)
- backend/app/models/weights/f3net_best.pth (188 MB)
- backend/app/models/weights/xception_best.pth (85 MB)

**Total size**: ~2.1 GB

**Action Required**: None - Git LFS already configured

**Verify LFS:**
```powershell
git lfs ls-files
# Should show all .pth files with * (pointer stored in repo)
```

---

### 3. Environment Variables Setup

#### Create .env.example Files

You need to create template files (with dummy values) to commit to repo.

**Backend .env.example:**
```env
# Backend Configuration
PORT=8000
HOST=0.0.0.0

# CORS Origins (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000,https://your-vercel-app.vercel.app

# Model Configuration
MODEL_DEVICE=cuda  # or 'cpu' for Render
ENABLE_MLFLOW=true

# MLflow Configuration (optional)
MLFLOW_TRACKING_URI=file:///app/mlruns
MLFLOW_EXPERIMENT_NAME=deepfake_detection

# File Upload Limits
MAX_FILE_SIZE_MB=10
ALLOWED_EXTENSIONS=jpg,jpeg,png,mp4,avi,mov

# Security
RATE_LIMIT_PER_MINUTE=60
```

**Frontend .env.local.example:**
```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# For production on Vercel:
# NEXT_PUBLIC_API_URL=https://your-render-backend.onrender.com
# NEXT_PUBLIC_WS_URL=wss://your-render-backend.onrender.com
```

**Create actual .env files (DO NOT COMMIT):**
- backend/.env (for local development)
- frontend/.env.local (for local development)

---

### 4. Update .gitignore

**Action Required:**

```powershell
# Backup current .gitignore
Copy-Item .gitignore .gitignore.bak

# Update .gitignore (see section 1 above for full content)
```

---

### 5. Clean Up Unnecessary Files

#### Files to Remove/Keep

**REMOVE before pushing (if exist):**
```powershell
# Remove Python cache
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# Remove .pyc files
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force

# Remove MLflow data (too large for git)
Remove-Item -Recurse -Force backend/mlruns -ErrorAction SilentlyContinue

# Remove node_modules if accidentally added
Remove-Item -Recurse -Force frontend/node_modules -ErrorAction SilentlyContinue

# Remove .next build folder
Remove-Item -Recurse -Force frontend/.next -ErrorAction SilentlyContinue
```

**KEEP (important documentation):**
- All *.md files
- diagrams/ folder
- requirements.txt
- package.json
- All source code files

---

### 6. Update Requirements Files

#### Backend requirements.txt

**Verify requirements.txt is production-ready:**

```powershell
cd backend
python -m pip freeze > requirements-full.txt
# Review and keep only necessary packages in requirements.txt
```

**Current requirements.txt should include:**
```
torch==2.8.0
torchvision==0.20.0
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
pillow==10.2.0
facenet-pytorch==2.5.3
timm==1.0.3
opencv-python==4.9.0.80
python-dotenv==1.0.0
mlflow>=2.10.0
numpy==1.24.3
pydantic==2.5.3
```

#### Frontend package.json

**Verify dependencies:**
- Remove any unused packages
- Ensure all required packages listed

---

### 7. Configuration Files for Deployment

#### Create render.yaml (for Render backend)

**Action Required**: Create this file in project root

```yaml
# render.yaml
services:
  - type: web
    name: deepfake-detection-backend
    runtime: python
    plan: free  # Change to 'standard' for better performance
    buildCommand: "pip install -r backend/requirements.txt"
    startCommand: "cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
      - key: MODEL_DEVICE
        value: cpu  # Render free tier doesn't have GPU
      - key: ENABLE_MLFLOW
        value: false  # Disable for free tier (storage limits)
    disk:
      name: model-storage
      mountPath: /opt/render/project/backend/app/models/weights
      sizeGB: 10  # Need space for 2.1GB models
```

#### Create vercel.json (for Vercel frontend)

**Action Required**: Create in frontend/ folder

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "nextjs",
  "regions": ["iad1"],
  "env": {
    "NEXT_PUBLIC_API_URL": "@api_url",
    "NEXT_PUBLIC_WS_URL": "@ws_url"
  }
}
```

---

### 8. Update CORS Configuration

**File**: backend/app/main.py

**Action Required**: Update CORS origins for production

```python
# In main.py, update origins:
origins = [
    "http://localhost:3000",
    "https://your-app-name.vercel.app",  # Add your Vercel domain
    "https://*.vercel.app",  # Allow all Vercel preview deployments
]
```

**Or use environment variable (RECOMMENDED):**

```python
# In main.py:
import os

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### 9. Git LFS Bandwidth Consideration

#### ⚠️ WARNING: GitHub LFS Bandwidth Limits

**Free Plan Limits:**
- Storage: 1 GB
- Bandwidth: 1 GB/month

**Your models**: 2.1 GB total

**Problem**: Exceeds free tier storage limit!

**Solutions:**

**Option A: Use GitHub LFS (Paid)**
- GitHub Pro: $4/month (50 GB bandwidth)
- Data packs: $5/month per 50 GB

**Option B: External Model Storage (RECOMMENDED for production)**
- Upload models to cloud storage (AWS S3, Google Cloud Storage, etc.)
- Download during deployment
- Update code to fetch models from URL

**Option C: Render Disk Storage**
- Use Render's persistent disk (included in paid plans)
- Upload models directly to Render via SSH
- Not tracked in git

---

### 10. Create Deployment Scripts

#### backend/download_models.py

**For Option B above:**

```python
import os
import urllib.request
from pathlib import Path

MODELS_URL = {
    "xception_best.pth": "https://your-storage-url/xception_best.pth",
    "f3net_best.pth": "https://your-storage-url/f3net_best.pth",
    "effort_clip_L14_trainOn_FaceForensic.pth": "https://your-storage-url/effort_clip.pth",
}

def download_models():
    weights_dir = Path(__file__).parent / "app/models/weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in MODELS_URL.items():
        filepath = weights_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}")
        else:
            print(f"{filename} already exists")

if __name__ == "__main__":
    download_models()
```

**Update render.yaml buildCommand:**
```yaml
buildCommand: "pip install -r backend/requirements.txt && python backend/download_models.py"
```

---

### 11. Test Locally Before Pushing

#### Backend Test

```powershell
cd backend

# Clean environment
Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue

# Fresh install
python -m venv test_venv
.\test_venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Test server
uvicorn app.main:app --reload

# Test endpoint
curl http://localhost:8000/health
```

#### Frontend Test

```powershell
cd frontend

# Clean install
Remove-Item -Recurse -Force node_modules -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .next -ErrorAction SilentlyContinue

npm install
npm run build
npm run start

# Test in browser: http://localhost:3000
```

---

### 12. Prepare README for Deployment

**Update README.md with:**

1. Live demo links (will add after deployment)
2. Deployment instructions
3. Environment variables needed
4. Known limitations (CPU vs GPU performance)

---

## GitHub Push Checklist

Before running `git push`:

- [ ] Updated .gitignore (see section 1)
- [ ] Removed all __pycache__ folders
- [ ] Removed mlruns/ folder
- [ ] Removed node_modules/
- [ ] Removed .next/ build folder
- [ ] No .env files in staging area
- [ ] Created .env.example files
- [ ] No API keys/secrets in code
- [ ] Git LFS properly configured
- [ ] render.yaml created
- [ ] vercel.json created
- [ ] CORS origins updated
- [ ] Tested locally (backend + frontend)
- [ ] All documentation up to date
- [ ] Committed recent fixes

**Verify staging area:**
```powershell
git status
git diff --cached  # Review what will be committed
```

**Final check:**
```powershell
# Check for secrets
git grep -i "password" HEAD
git grep -i "secret_key" HEAD
git grep -i "api_key" HEAD
```

---

## Deployment Steps

### Step 1: Push to GitHub

```powershell
# Stage all files
git add .

# Review what will be committed
git status

# Commit with descriptive message
git commit -m "Prepare for deployment: Update configs, fix bugs, add documentation"

# Push to GitHub
git push origin main
```

---

### Step 2: Deploy Backend to Render

1. **Go to**: https://dashboard.render.com
2. **New** → **Web Service**
3. **Connect GitHub repo**
4. **Settings:**
   - Name: deepfake-detection-backend
   - Environment: Python 3
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`

5. **Environment Variables** (Add these):
   ```
   MODEL_DEVICE=cpu
   ENABLE_MLFLOW=false
   ALLOWED_ORIGINS=https://your-vercel-app.vercel.app
   ```

6. **Advanced:**
   - Disk: Add persistent disk (10 GB) for models
   - Health Check Path: /health

7. **Create Web Service**

8. **Wait for deploy** (15-30 minutes first time - downloads 2.1GB models)

9. **Get backend URL**: https://your-app.onrender.com

---

### Step 3: Deploy Frontend to Vercel

1. **Go to**: https://vercel.com
2. **Import Project** → **Import Git Repository**
3. **Select GitHub repo**
4. **Configure:**
   - Framework Preset: Next.js
   - Root Directory: frontend
   - Build Command: `npm run build`
   - Output Directory: .next

5. **Environment Variables** (Add these):
   ```
   NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
   NEXT_PUBLIC_WS_URL=wss://your-backend.onrender.com
   ```

6. **Deploy**

7. **Get frontend URL**: https://your-app.vercel.app

---

### Step 4: Update CORS

**After getting Vercel URL:**

1. Go to Render dashboard
2. Environment → Add variable:
   ```
   ALLOWED_ORIGINS=https://your-app.vercel.app,https://*.vercel.app
   ```
3. Save → Redeploy

---

### Step 5: Test Production

1. Visit: https://your-app.vercel.app
2. Upload test image
3. Verify:
   - Image uploads successfully
   - Detection works (may be slow on Render free tier)
   - Heatmap displays correctly
   - No CORS errors in browser console

4. Test API directly:
   ```
   curl https://your-backend.onrender.com/health
   ```

---

## Common Deployment Issues

### Issue 1: Render Build Fails (Out of Memory)

**Symptom**: Build fails during PyTorch installation

**Solution**:
- Upgrade to paid plan (512 MB not enough)
- OR: Use Docker with pre-built PyTorch image

---

### Issue 2: Models Not Found

**Symptom**: `FileNotFoundError: weights not found`

**Solution**:
- Verify Git LFS pushed correctly: `git lfs ls-files`
- OR: Use external storage (see section 9, Option B)

---

### Issue 3: Vercel Build Timeout

**Symptom**: Frontend build exceeds 45 second limit

**Solution**:
- Remove unused dependencies
- Optimize images
- Upgrade Vercel plan

---

### Issue 4: CORS Errors

**Symptom**: `Access-Control-Allow-Origin` errors

**Solution**:
- Update ALLOWED_ORIGINS on Render
- Verify frontend URL matches exactly
- Include both www and non-www versions

---

### Issue 5: Slow Performance on Render Free Tier

**Expected**:
- 512 MB RAM is VERY limited for 3 models
- CPU inference is 3-10x slower than GPU
- May timeout on Render free tier (30s limit)

**Solutions**:
- Upgrade to paid plan (recommended)
- Reduce to 1-2 models only
- Implement request queuing

---

## Cost Estimation

### Free Tier (Testing Only)
- GitHub: Free (but LFS limited to 1 GB - your models are 2.1 GB)
- Vercel: Free (hobby plan)
- Render: Free (but insufficient for 3 models)

**Total**: $0/month (limited functionality)

---

### Recommended Production Setup
- GitHub Pro: $4/month (for LFS bandwidth)
- Vercel Pro: $20/month (better performance)
- Render Standard: $7/month (1 GB RAM - still tight)
- OR Render Pro: $25/month (2 GB RAM - recommended)

**Total**: $31-49/month

---

### Alternative: Cloud Storage for Models
- GitHub: Free
- Vercel: Free/Pro ($0-20/month)
- Render Standard: $7/month
- AWS S3: ~$1/month (storage only)
- CloudFront CDN: ~$1/month (model downloads)

**Total**: $9-28/month (more scalable)

---

## Post-Deployment Tasks

- [ ] Update README.md with live demo URLs
- [ ] Set up monitoring (Render/Vercel dashboards)
- [ ] Configure custom domain (optional)
- [ ] Set up error tracking (Sentry, etc.)
- [ ] Monitor performance and costs
- [ ] Set up CI/CD (GitHub Actions)
- [ ] Create backup strategy for models
- [ ] Document API for users

---

## Next Steps Document

After completing this checklist, see:
- [DEPLOYMENT_GUIDE_VERCEL.md](DEPLOYMENT_GUIDE_VERCEL.md) - Detailed Vercel guide
- [DEPLOYMENT_GUIDE_RENDER.md](DEPLOYMENT_GUIDE_RENDER.md) - Detailed Render guide
- [POST_DEPLOYMENT.md](POST_DEPLOYMENT.md) - After deployment tasks

---

**Status**: Ready to review and update files
**Last Updated**: 2025-10-19
