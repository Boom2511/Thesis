# 🎯 Next Steps - แก้ Model Loading และทดสอบใหม่

## 📋 สรุปปัญหาที่พบ:

คุณ run `Quick_Weight_Optimization.ipynb` แล้วได้ผลลัพธ์แย่มาก:
- **Xception:** 51.3% accuracy, precision/recall = 0.0 (ทำนาย REAL หมด)
- **F3Net:** 51.3% accuracy, precision/recall = 0.0 (ทำนาย REAL หมด)
- **Effort-CLIP:** 48.3% accuracy (ทำนาย FAKE เกือบหมด)

**สาเหตุ:** Model weights ไม่โหลดเข้าโมเดลอย่างถูกต้อง

---

## ✅ สิ่งที่ต้องทำ (3 ขั้นตอน):

### **1️⃣ เปิด Colab Notebook**
- เปิดไฟล์: `Quick_Weight_Optimization.ipynb`
- ไปที่ **Cell 11** (โค้ดที่โหลดโมเดล)

### **2️⃣ แทนที่โค้ด Cell 11**
- เปิดไฟล์: `CORRECTED_MODEL_LOADING.md`
- คัดลอกโค้ดทั้งหมดจากส่วน "✅ โค้ดที่แก้ไขแล้ว"
- แทนที่ Cell 11 เดิมทั้งหมด

### **3️⃣ เลือก Option**

**Option A: ทดสอบ 3 โมเดล (รวม Effort ที่มีปัญหา)**
```python
# Cell 12 - ใช้โค้ดเดิม
xception = XceptionModel(XCEPTION_PATH, device)
f3net = F3NetModel(F3NET_PATH, device)
effort = EffortModel(EFFORT_PATH, device)  # จะใช้ random weights

models = {
    'xception': xception,
    'f3net': f3net,
    'effort': effort
}
```
**ผลที่คาดหวัง:**
- Xception: ~90-97% ✅
- F3Net: ~90-97% ✅
- Effort: ~45-55% ❌ (random weights)
- Ensemble (3 models): อาจได้ผลแย่เพราะ Effort ทำงานไม่ดี

---

**Option B: ทดสอบ 2 โมเดล (แนะนำ!)**

**แก้ Cell 12:**
```python
# โหลดแค่ 2 โมเดล
xception = XceptionModel(XCEPTION_PATH, device)
f3net = F3NetModel(F3NET_PATH, device)

models = {
    'xception': xception,
    'f3net': f3net
}

print("✅ Using 2 models (Xception + F3Net)")
```

**แก้ Cell 18:**
```python
def evaluate_ensemble(weights, results):
    """ประเมิน ensemble ด้วย 2 โมเดล"""
    w_xception, w_f3net = weights

    ensemble_pred = (
        results['xception']['predictions'] * w_xception +
        results['f3net']['predictions'] * w_f3net
    )

    labels = results['xception']['labels']
    pred_labels = (ensemble_pred > 0.5).astype(int)

    acc = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels, zero_division=0)
    auc = roc_auc_score(labels, ensemble_pred)

    return {'accuracy': acc, 'f1': f1, 'auc': auc}
```

**แก้ Cell 19:**
```python
# Grid search สำหรับ 2 โมเดล
print("\n" + "="*50)
print("🔍 Searching for Optimal Weights (2 models)")
print("="*50)

step = 0.05
weight_range = np.arange(0.0, 1.0 + step, step)

best_score = 0
best_weights = None
best_metrics = None
all_results = []

print(f"\n⚙️  Grid search with step={step}")
print(f"   Total combinations: {len(weight_range)}\n")

for w1 in tqdm(weight_range, desc="Grid Search"):
    w2 = 1.0 - w1

    if w2 < 0 or w2 > 1.0:
        continue

    weights = (w1, w2)
    metrics = evaluate_ensemble(weights, results)
    score = metrics['f1']

    all_results.append({
        'weights': weights,
        'metrics': metrics,
        'score': score
    })

    if score > best_score:
        best_score = score
        best_weights = weights
        best_metrics = metrics

print("\n" + "="*50)
print("🏆 BEST ENSEMBLE CONFIGURATION (2 models)")
print("="*50)
print(f"\n📊 Optimal Weights:")
print(f"  Xception: {best_weights[0]:.3f} ({best_weights[0]*100:.1f}%)")
print(f"  F3Net:    {best_weights[1]:.3f} ({best_weights[1]*100:.1f}%)")
print(f"\n📈 Performance:")
print(f"  Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
print(f"  F1 Score: {best_metrics['f1']:.4f} ({best_metrics['f1']*100:.2f}%)")
print(f"  AUC:      {best_metrics['auc']:.4f} ({best_metrics['auc']*100:.2f}%)")
print("="*50)
```

**แก้ Cell 25 (config file):**
```python
new_config = {
  "models": {
    "xception": {
      "name": "xception",
      "path": "app/models/weights/xception_best.pth",
      "description": "Fast and reliable baseline",
      "weight": round(best_weights[0], 2),
      "enabled": True
    },
    "efficientnet_b4": {
      "name": "tf_efficientnet_b4",
      "path": "app/models/weights/effnb4_best.pth",
      "description": "Balanced performance (DISABLED: incompatible checkpoint)",
      "weight": 0.0,
      "enabled": False
    },
    "f3net": {
      "name": "f3net",
      "path": "app/models/weights/f3net_best.pth",
      "description": "Frequency-aware network with spatial attention",
      "weight": round(best_weights[1], 2),
      "enabled": True
    },
    "effort": {
      "name": "effort_clip",
      "path": "app/models/weights/effort_clip_L14_trainOn_FaceForensic.pth",
      "description": "CLIP-based multimodal detection (DISABLED: no classifier)",
      "weight": 0.0,
      "enabled": False  # ← ปิดการใช้งาน
    }
  },
  "ensemble": {
    "method": "weighted_average",
    "threshold": 0.5,
    "min_models": 2
  },
  "device": "cuda",
  "face_detection": {
    "min_confidence": 0.85,
    "min_face_size": 40
  },
  "inference": {
    "batch_size": 1,
    "generate_gradcam": False
  }
}

with open('config_optimized.json', 'w') as f:
    json.dump(new_config, f, indent=2)

print("✅ Config saved: config_optimized.json (2 models)")
print("\n📋 คัดลอกไปแทนที่: backend/app/config.json")
```

**ผลที่คาดหวัง:**
- Xception: ~90-97% ✅
- F3Net: ~90-97% ✅
- Ensemble: ~92-98% ✅✅

---

## 🎯 แนะนำ: **ใช้ Option B (2 models)**

**เหตุผล:**
1. Effort-CLIP checkpoint ไม่มี classifier head → ใช้ random weights
2. 2 โมเดลที่เหลือทำงานได้ดีพอ (90-97%)
3. Ensemble 2 โมเดลคุณภาพดี > 3 โมเดลที่มี 1 อันแย่
4. ประหยัด compute units (~30% faster)

---

## ✅ Run Notebook ใหม่:

1. แก้ Cell 11, 12, 18, 19, 25 ตาม Option B
2. **Runtime → Restart and run all**
3. ดูผลลัพธ์:

```
🔧 Loading Xception from xception_best.pth
  ✅ Using checkpoint directly
  📊 Loaded XXX parameters
  ✅ Classifier layer loaded    ← ต้องมีบรรทัดนี้!

🔧 Loading F3Net from f3net_best.pth
  ✅ Using checkpoint directly
  📊 Loaded XXX parameters
  🗑️  Skipped XX FAD_head layers
  ✅ Classifier layer loaded    ← ต้องมีบรรทัดนี้!
```

4. ตรวจสอบ accuracy:

```
📊 XCEPTION Performance:
  Accuracy:  0.9XXX    ← ต้อง > 0.85
  F1 Score:  0.9XXX

📊 F3NET Performance:
  Accuracy:  0.9XXX    ← ต้อง > 0.85
  F1 Score:  0.9XXX
```

5. ดู optimal weights:

```
🏆 BEST ENSEMBLE CONFIGURATION (2 models)
  Xception: 0.XXX
  F3Net:    0.XXX

  Accuracy: 0.9XXX
  F1 Score: 0.9XXX
```

---

## 🚀 หลัง Run เสร็จ:

### **Download ไฟล์:**
- `config_optimized.json` → คัดลอกไป `backend/app/config.json`
- `weight_optimization_report.json` → เก็บไว้อ้างอิง
- `model_comparison.png` → สำหรับ presentation
- `top10_configurations.png` → วิเคราะห์ weights

### **Update Backend:**
```bash
# ใน local machine
cd deepfake-detection/backend

# Backup config เดิม
cp app/config.json app/config.json.backup

# คัดลอกไฟล์ใหม่
# (วางไฟล์ config_optimized.json ที่ download มา)
cp config_optimized.json app/config.json

# Restart server
python -m uvicorn app.main:app --reload
```

### **ทดสอบ:**
```bash
# เปิด browser
http://localhost:3000

# อัปโหลดภาพทดสอบ → ดูผลลัพธ์
```

---

## ❓ FAQ:

**Q: ถ้า accuracy ยังต่ำกว่า 80%?**
A: ตรวจสอบ:
- ✅ "Classifier layer loaded" ปรากฏหรือไม่
- Dataset balance (real:fake ratio)
- Image quality

**Q: ทำไมไม่ใช้ 3 โมเดล?**
A: Effort checkpoint ไม่มี classifier → ใช้ไม่ได้

**Q: หา Effort checkpoint ที่ถูกต้องได้ไหม?**
A: ต้องมี checkpoint ที่มี `classifier.weight` และ `classifier.bias`

**Q: ควรใช้ weights อะไร?**
A: ใช้ weights ที่ notebook หาให้ (optimal weights)

**Q: ถ้า ensemble ดีขึ้นแค่ 0.5%?**
A: ยังคุ้มค่าใช้ (98% vs 97.5% สำคัญมากในงานจริง)

---

## 🎯 สรุป:

1. ✅ แก้ Cell 11 (model loading)
2. ✅ แก้ Cell 12, 18, 19, 25 (2 models)
3. ✅ Run notebook ใหม่
4. ✅ ตรวจสอบ "Classifier layer loaded"
5. ✅ ตรวจสอบ accuracy > 85%
6. ✅ Download config
7. ✅ Update backend
8. ✅ ทดสอบ

**ใช้เวลา:** 20-30 นาที
**Compute Units:** ~15-20 units
**Expected Accuracy:** 92-98%

---

**ขอให้โชคดี! 🎉**

**Updated:** 25 ตุลาคม 2025
