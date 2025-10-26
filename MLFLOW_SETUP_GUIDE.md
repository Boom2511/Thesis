# 📊 MLflow Setup & Usage Guide

## Why MLflow Shows No Experiments

MLflow is running correctly, but you need to make predictions first for experiments to appear!

---

## ✅ How to See MLflow Experiments

### Step 1: Make Sure Backend is Running
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Upload Images to Create Experiments

1. Go to http://localhost:3000
2. Upload an image (single mode)
3. Wait for the result
4. ✅ MLflow will automatically log this prediction!

### Step 3: Check MLflow UI

1. Go to http://localhost:5000
2. You should now see the "deepfake_detection" experiment
3. Click on it to see your runs

---

## 📈 What You'll See in MLflow

### Experiments Page
- **deepfake_detection** experiment (created automatically)
- Number of runs (each image = 1 run)
- Last update time

### Run Details
Click on any run to see:

#### **Metrics** (Automatically Logged):
- `confidence` - Overall prediction confidence
- `fake_probability` - Probability of being fake
- `real_probability` - Probability of being real
- `processing_time_sec` - How long detection took
- `xception_fake_prob` - Xception model's fake probability
- `effort_fake_prob` - Effort model's fake probability
- Model-specific confidence scores

#### **Parameters** (Automatically Logged):
- `prediction` - FAKE or REAL
- `prediction_type` - image/video/webcam
- `models_used` - Which models were used
- `num_models` - Number of models in ensemble

#### **Tags**:
- Device used (CPU/GPU)
- Model names

---

## 🧪 Quick Test

### Test MLflow Logging (via cURL):

```bash
# Test image upload (replace with your image path)
curl -X POST http://localhost:8000/api/detect/image \
  -F "file=@path/to/your/image.jpg" \
  -F "generate_heatmap=true"
```

Then refresh MLflow UI at http://localhost:5000

---

## 📊 MLflow UI Navigation

### 1. Experiments Tab (Main Page)
- Lists all experiments
- Click "deepfake_detection" to see runs

### 2. Runs Page
- Shows all prediction runs
- Sortable by confidence, time, prediction
- Filter by FAKE/REAL

### 3. Run Details Page
Click any run to see:
- **Overview**: Prediction, confidence, time
- **Metrics**: Charts and graphs
- **Parameters**: Configuration
- **Artifacts**: Saved files (if any)
- **Tags**: Metadata

### 4. Compare Runs
- Select multiple runs (checkbox)
- Click "Compare"
- See side-by-side metrics

---

## 🎯 Common Use Cases

### 1. Track Prediction Accuracy
```python
# MLflow automatically logs each prediction
# View in UI: Metrics > confidence
```

### 2. Compare Model Performance
```python
# View in UI: Metrics > xception_fake_prob vs effort_fake_prob
```

### 3. Analyze Processing Time
```python
# View in UI: Metrics > processing_time_sec
# Filter slow predictions
```

### 4. Export Results
```python
from services.mlflow_service import MLflowService

mlflow = MLflowService()
stats = mlflow.get_experiment_stats()

print(f"Total predictions: {stats['total_runs']}")
print(f"Fake detected: {stats['total_fake']}")
print(f"Real detected: {stats['total_real']}")
```

---

## 🔧 Troubleshooting

### No Experiments Showing?

**Cause**: No predictions made yet

**Solution**:
1. Make sure backend is running (port 8000)
2. Upload at least one image
3. Refresh MLflow UI (http://localhost:5000)

### MLflow Service Not Initializing?

Check backend logs for:
```
✅ MLflow initialized - Experiment: deepfake_detection
```

If you see:
```
⚠️  MLflow initialization failed
```

Then MLflow is disabled. Check `backend/app/services/detection.py`

### Empty Run Details?

**Cause**: MLflow logging might be disabled

**Solution**:
```python
# In detection.py
service = EnsembleDetectionService(
    config_path="app/config.json",
    enable_mlflow=True  # ✅ Make sure this is True
)
```

---

## 📱 Advanced Features

### 1. Query Runs Programmatically

```python
import mlflow

# Get all runs
runs = mlflow.search_runs(
    experiment_names=["deepfake_detection"]
)

# Filter fake detections
fake_runs = runs[runs['params.prediction'] == 'FAKE']
print(f"Fake images: {len(fake_runs)}")

# Get high-confidence fakes
high_conf = fake_runs[fake_runs['metrics.confidence'] > 0.9]
print(f"High confidence fakes: {len(high_conf)}")
```

### 2. Export to CSV

```python
import mlflow
import pandas as pd

runs = mlflow.search_runs(experiment_names=["deepfake_detection"])

# Export to CSV
runs.to_csv('mlflow_results.csv', index=False)
```

### 3. Compare Models

```python
import mlflow

runs = mlflow.search_runs(experiment_names=["deepfake_detection"])

# Average confidence by model
xception_avg = runs['metrics.xception_confidence'].mean()
effort_avg = runs['metrics.effort_fake_prob'].mean()

print(f"Xception avg: {xception_avg:.2%}")
print(f"Effort avg: {effort_avg:.2%}")
```

---

## 🎨 MLflow UI Tips

### Customize Columns
- Click column headers to sort
- Click gear icon to show/hide columns
- Add custom columns for metrics

### Filter Runs
- Use search bar: `metrics.confidence > 0.9`
- Filter by parameters: `params.prediction = "FAKE"`
- Date range filters

### Download Data
- Click run name → "Download Run Data"
- Get metrics, params, artifacts as ZIP

---

## 🚀 Production Setup

### Use Database Backend

```bash
# Instead of files, use PostgreSQL
mlflow server \
    --backend-store-uri postgresql://user:pass@localhost/mlflow \
    --default-artifact-root s3://my-bucket/ \
    --host 0.0.0.0 \
    --port 5000
```

### Remote Tracking

```python
# In detection.py
mlflow_service = MLflowService(
    experiment_name="deepfake_detection",
    tracking_uri="http://mlflow-server:5000"  # Remote server
)
```

---

## ✅ Quick Checklist

Before expecting to see data in MLflow:

- [ ] Backend running (port 8000)
- [ ] MLflow UI running (port 5000)
- [ ] MLflow enabled in detection service
- [ ] At least one image uploaded
- [ ] Check backend logs for "MLflow initialized"
- [ ] Refresh MLflow UI

---

## 📝 Summary

**MLflow is working!** It just needs predictions to track.

**To see experiments**:
1. Upload images via the web UI
2. Each upload creates a new "run"
3. Refresh MLflow UI to see results

**What gets logged automatically**:
- ✅ Every image prediction
- ✅ Every video analysis
- ✅ Every webcam frame (optional)
- ✅ All metrics and parameters
- ✅ Model comparisons

**MLflow UI**: http://localhost:5000

**Happy tracking!** 🎉
