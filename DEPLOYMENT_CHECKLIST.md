# Deployment Checklist - Hugging Face + Vercel

**Print this and check off as you go!** ✅

---

## Part 1: Hugging Face Backend (30 minutes)

### ☐ Step 1: Hugging Face Account (3 min)
- ☐ Go to https://huggingface.co/join
- ☐ Sign up with email
- ☐ Verify email
- ☐ Go to https://huggingface.co/settings/tokens
- ☐ Create new token (Write access)
- ☐ **Copy token** and save it
- ☐ Note your username: `_________________`

### ☐ Step 2: Create Space (5 min)
- ☐ Go to https://huggingface.co/new-space
- ☐ Name: `deepfake-detection`
- ☐ SDK: **Gradio**
- ☐ Hardware: **CPU basic - Free**
- ☐ Visibility: **Public**
- ☐ Click "Create Space"
- ☐ Copy Space URL: `https://huggingface.co/spaces/__________/deepfake-detection`

### ☐ Step 3: Install & Login HF CLI (5 min)
```powershell
cd C:\Users\Admin\deepfake-detection
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```
- ☐ HF CLI installed
- ☐ Logged in successfully

### ☐ Step 4: Clone Space (2 min)
```powershell
git clone https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detection hf-space
cd hf-space
```
- ☐ Space cloned to `hf-space` folder

### ☐ Step 5: Copy Files (3 min)
```powershell
xcopy /E /I /Y ..\backend backend
copy /Y ..\app.py app.py
copy /Y ..\requirements_hf.txt requirements.txt
copy /Y ..\README_HF.md README.md
```
- ☐ Backend folder copied
- ☐ app.py copied
- ☐ requirements.txt copied
- ☐ README.md copied

### ☐ Step 6: Push to HF (5 min)
```powershell
git add .
git commit -m "Deploy deepfake detection system"
git push
```
- ☐ Files committed
- ☐ Pushed to Hugging Face
- ☐ Model files uploading (1.44 GB - takes 5-10 min)

### ☐ Step 7: Wait for Build (10 min)
- ☐ Open Space URL in browser
- ☐ Watch build logs
- ☐ See "Installing dependencies..."
- ☐ See "Downloading model weights..."
- ☐ See "[OK] Loaded 3 models successfully"
- ☐ See "✓ Running"

### ☐ Step 8: Test Backend (5 min)
- ☐ Upload test image in Gradio interface
- ☐ Click "🔍 Analyze Image"
- ☐ Get prediction result
- ☐ Verify it works!

### ☐ Step 9: (Optional) Request GPU
- ☐ Go to Space → Settings → Hardware
- ☐ Click "Request GPU grant"
- ☐ Fill form
- ☐ Submit request
- ☐ Wait for approval

**✅ Backend Complete!**
- Live URL: `https://YOUR_USERNAME-deepfake-detection.hf.space`

---

## Part 2: Vercel Frontend (15 minutes)

### ☐ Step 10: Configure Frontend (3 min)
```powershell
cd ..\frontend
```

Create `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=https://YOUR_USERNAME-deepfake-detection.hf.space
NEXT_PUBLIC_WS_URL=wss://YOUR_USERNAME-deepfake-detection.hf.space
```

- ☐ .env.local created
- ☐ URLs updated with YOUR username

### ☐ Step 11: (Optional) Test Locally (5 min)
```powershell
npm install
npm run dev
```
- ☐ Open http://localhost:3000
- ☐ Upload image
- ☐ Verify connects to HF backend
- ☐ Press Ctrl+C to stop

### ☐ Step 12: Push to GitHub
```powershell
cd ..
git add frontend/.env.local
git add .
git commit -m "Configure frontend for HF backend"
git push origin main
```
- ☐ Changes pushed to GitHub

### ☐ Step 13: Create Vercel Account (2 min)
- ☐ Go to https://vercel.com/signup
- ☐ Sign up with GitHub
- ☐ Authorize Vercel
- ☐ Choose Hobby plan (FREE)

### ☐ Step 14: Deploy to Vercel (5 min)
- ☐ Go to https://vercel.com/dashboard
- ☐ Click "Add New..." → "Project"
- ☐ Import `deepfake-detection` repo
- ☐ Set Root Directory: `frontend`
- ☐ Framework: Next.js (auto-detect)
- ☐ Add environment variables:
  - ☐ `NEXT_PUBLIC_API_URL` = `https://YOUR_USERNAME-deepfake-detection.hf.space`
  - ☐ `NEXT_PUBLIC_WS_URL` = `wss://YOUR_USERNAME-deepfake-detection.hf.space`
- ☐ Click "Deploy"
- ☐ Wait for build (3-5 min)

### ☐ Step 15: Test Complete System (5 min)
- ☐ Copy Vercel URL: `https://_____________________.vercel.app`
- ☐ Open in browser
- ☐ Upload test image
- ☐ Check browser console (F12) - no errors
- ☐ Get prediction result
- ☐ Test 2-3 more images
- ☐ Everything works!

**✅ Frontend Complete!**
- Live URL: `https://your-project.vercel.app`

---

## 🎉 DEPLOYMENT COMPLETE!

### Your Live System:

**Backend (Hugging Face):**
```
https://YOUR_USERNAME-deepfake-detection.hf.space
```

**Frontend (Vercel):**
```
https://your-project-name.vercel.app
```

### Summary:
- ✅ Backend deployed to Hugging Face (FREE GPU)
- ✅ Frontend deployed to Vercel (FREE)
- ✅ Both services connected
- ✅ System tested and working
- ✅ **Total cost: $0/month**

---

## 📊 Final Checks

- ☐ Backend responds to requests
- ☐ Frontend loads correctly
- ☐ Image upload works
- ☐ Predictions are accurate (not 1%!)
- ☐ No console errors
- ☐ Both URLs saved for sharing
- ☐ URLs added to README.md

---

## 🚀 Next Steps

- ☐ Share URLs with friends/users
- ☐ Monitor Hugging Face Space logs
- ☐ Monitor Vercel analytics
- ☐ Request GPU upgrade (if not done)
- ☐ Update documentation with live demo
- ☐ Celebrate! 🎉

---

## 📝 Important URLs to Save

| Service | URL |
|---------|-----|
| HF Space | `https://huggingface.co/spaces/________/deepfake-detection` |
| HF Live URL | `https://________-deepfake-detection.hf.space` |
| Vercel Dashboard | `https://vercel.com/dashboard` |
| Vercel Live URL | `https://__________.vercel.app` |
| GitHub Repo | `https://github.com/________/deepfake-detection` |

---

## 🆘 If Something Goes Wrong

### Backend not building?
1. Check Hugging Face Space → Logs
2. Verify model files uploaded (Git LFS)
3. Check requirements.txt has all dependencies

### Frontend not connecting?
1. Verify environment variables in Vercel
2. Check URLs match exactly (no typos)
3. Check browser console for errors

### Need help?
1. Check [STEP_BY_STEP_HUGGINGFACE_VERCEL.md](STEP_BY_STEP_HUGGINGFACE_VERCEL.md)
2. Review Hugging Face docs: https://huggingface.co/docs
3. Review Vercel docs: https://vercel.com/docs

---

**Total Time**: 45-60 minutes
**Total Cost**: $0/month
**Difficulty**: Easy (just follow steps!)

**Good luck! You got this! 🚀**
