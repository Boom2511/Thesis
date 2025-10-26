# C23 Weight Optimization Notebook - READY

## Summary

Created **[Weight_Optimization_C23.ipynb](Weight_Optimization_C23.ipynb)** - a Colab notebook optimized for your c23 dataset with corrected model loading and efficient data handling.

---

## Dataset Structure Discovered

Your extracted dataset at `C:\Users\Admin\Downloads\dataset_c23\`:

```
dataset_c23/
├── manipulated_sequences/
│   ├── Deepfakes/c23/frames/
│   │   ├── 000_003/  ← original_fake naming
│   │   ├── 001_870/
│   │   └── ...
│   ├── Face2Face/c23/frames/
│   ├── FaceSwap/c23/frames/
│   ├── FaceShifter/c23/frames/
│   ├── DeepFakeDetection/c23/frames/
│   └── NeuralTextures/c23/frames/
├── original_sequences/
│   └── youtube/c23/frames/
│       ├── 000/
│       ├── 001/
│       └── ...
├── test.json  ← [["original_id", "fake_id"], ...]
├── train.json
└── val.json
```

### Key Insight - Folder Naming:
- **test.json pairs:** `["953", "974"]` means original=953, fake=974
- **Fake folder name:** `953_974` (combined with underscore)
- **Original folder name:** `953` (just the ID)

---

## Fixes Applied

### 1. **Corrected Dataset Path**
```python
DATASET_ROOT = "/content/dataset_c23"  # Updated from dataset_
```

### 2. **Fixed Data Loading Function**
```python
def load_frame(video_id: str, is_fake: bool, method: str = "Deepfakes", original_id: str = None):
    if is_fake:
        # Fake videos: "original_fake" format (e.g., "953_974")
        folder_name = f"{original_id}_{video_id}"
        frames_dir = Path(DATASET_ROOT) / "manipulated_sequences" / method / "c23" / "frames" / folder_name
    else:
        # Original videos: just the ID (e.g., "953")
        frames_dir = Path(DATASET_ROOT) / "original_sequences" / "youtube" / "c23" / "frames" / video_id
    # ... load first frame
```

### 3. **Updated Data Loading Loop**
```python
for original_id, fake_id in sampled_pairs:
    # Load original
    original_img = load_frame(original_id, is_fake=False)

    # Load fake (pass original_id for correct folder lookup)
    fake_img = load_frame(fake_id, is_fake=True, method=method, original_id=original_id)
```

---

## Notebook Features

### ✅ Corrected Model Loading
- **Xception:** `backbone.` prefix removal, `fc` layer mapping
- **F3Net:** 12-channel input, FAD_head skip, Sequential→Linear mapping
- **Effort-CLIP:** Custom ViT (1024 dim), transformers implementation

### ✅ Efficient Sampling
- Samples **1000 pairs = 2000 images** (adjustable via `SAMPLE_SIZE`)
- Tries multiple manipulation methods (Deepfakes, Face2Face, FaceSwap)
- Conserves compute units: **~10-15 units** (not all 41)

### ✅ Grid Search Optimization
- Weight range: 0.0 to 1.0, step=0.1
- Tests ~66 combinations (weights sum to 1.0)
- Optimizes for **F1 score** (best for balanced datasets)
- Shows top 10 configurations

### ✅ Comprehensive Metrics
- Accuracy
- Precision
- Recall
- **F1 Score** (primary optimization target)
- AUC-ROC

### ✅ Config Generation
- Creates `config_optimized.json` with best weights
- Includes optimization metadata
- Ready to replace `backend/app/config.json`

---

## How to Use

### Step 1: Upload Dataset to Google Drive
```bash
# Your dataset is at: C:\Users\Admin\Downloads\dataset_c23.zip (21.1 GB)

# Option A: Use Google Drive Desktop app (recommended for large files)
# - Install Google Drive for Desktop
# - Drag dataset_c23.zip to your Drive folder

# Option B: Upload via browser (slow but works)
# - Go to drive.google.com
# - Upload dataset_c23.zip (will take time)
```

### Step 2: Open Notebook in Colab
1. Upload `Weight_Optimization_C23.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory
3. Runtime → Change runtime type → T4 GPU

### Step 3: Run Cells Sequentially
```python
# Cell 1: Mount Drive
# Cell 2: Install dependencies (timm, transformers, etc.)
# Cell 3: Extract dataset (if needed)
# Cell 4: Set DATASET_ROOT path
# Cell 5-13: Load models
# Cell 14-16: Load and sample test data
# Cell 17-19: Evaluate individual models
# Cell 20-21: Grid search for optimal weights
# Cell 22-23: Generate config_optimized.json
# Cell 24-25: Show summary and top 10 results
```

### Step 4: Download Results
- Download `config_optimized.json` from Colab
- Replace `backend/app/config.json` with optimized version
- Restart backend server

---

## Expected Output

### Individual Model Performance
```
Xception:
  Accuracy: 0.XXXX
  F1:       0.XXXX
  AUC:      0.XXXX

F3Net:
  Accuracy: 0.XXXX
  F1:       0.XXXX
  AUC:      0.XXXX

Effort-CLIP:
  Accuracy: 0.XXXX
  F1:       0.XXXX
  AUC:      0.XXXX
```

### Optimal Ensemble Weights
```
Weights:
  Xception:    0.XX
  F3Net:       0.XX
  Effort-CLIP: 0.XX

Metrics:
  Accuracy:  0.XXXX
  Precision: 0.XXXX
  Recall:    0.XXXX
  F1 Score:  0.XXXX
  AUC:       0.XXXX
```

### Top 10 Configurations Table
```
 w_xception  w_f3net  w_effort  f1      accuracy  precision  recall  auc
 0.4         0.3      0.3       0.XXXX  0.XXXX    0.XXXX     0.XXXX  0.XXXX
 0.5         0.2      0.3       0.XXXX  0.XXXX    0.XXXX     0.XXXX  0.XXXX
 ...
```

---

## Important Notes

### ⚠️ If Accuracy is Still ~50%

This confirms that **DeepfakeBench checkpoint weights are incompatible** with the dataset/architecture (as we discovered in previous testing).

**Solutions:**
1. **Train from scratch on c23 dataset** (recommended)
2. Use **Kaggle Notebooks** (FREE 30hrs/week GPU)
3. Use **Colab Pro** ($9.99/month, more compute)
4. Deploy with current weights (demo only, ~50% accuracy)

### ✅ If Accuracy is Good (>70%)

- Download `config_optimized.json`
- Replace `backend/app/config.json`
- Restart backend server
- Test with real images
- Celebrate! 🎉

---

## Compute Efficiency

**Estimated usage:**
- Model loading: ~2 compute units
- Data loading (1000 pairs): ~1 compute unit
- Individual model evaluation (3 models × 2000 images): ~6 compute units
- Grid search (~66 combinations): ~2 compute units
- **Total: ~11-13 compute units** (out of 41 remaining)

**Remaining after optimization:** ~28-30 compute units

---

## Files Created

1. **[Weight_Optimization_C23.ipynb](Weight_Optimization_C23.ipynb)** - Main optimization notebook
2. **[C23_NOTEBOOK_READY.md](C23_NOTEBOOK_READY.md)** - This summary document

---

## Next Steps

1. ✅ Upload `dataset_c23.zip` to Google Drive
2. ✅ Upload `Weight_Optimization_C23.ipynb` to Drive
3. ✅ Open notebook in Colab
4. ✅ Run all cells sequentially
5. ✅ Download `config_optimized.json`
6. ✅ Update `backend/app/config.json`
7. ✅ Test with real images

---

**Good luck with the optimization!** 🚀

If you encounter any issues, check:
- Dataset path matches extraction location
- All 3 model weights are in Google Drive
- GPU is enabled in Colab runtime
- Dependencies installed correctly
