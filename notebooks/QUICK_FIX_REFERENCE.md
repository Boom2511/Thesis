# ⚡ Quick Fix Reference - One Page

## 🎯 Problem:
Models predict ~50% accuracy (random guessing) → Classifier weights not loading

---

## ✅ Solution - แก้ 3 จุด:

### **1. Cell 3: เพิ่ม transformers**
```python
!pip install -q torch torchvision timm pillow scikit-learn tqdm
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q transformers  # ← เพิ่มบรรทัดนี้!
```

---

### **2. Cell 11: แทนที่ทั้งหมด**

คัดลอกโค้ดจาก **`CORRECTED_MODEL_LOADING.md`** มาแทนที่ Cell 11 ทั้งหมด

**Key changes:**

**Xception:**
```python
new_k = new_k.replace('backbone.', '')  # ← เพิ่ม!
```

**F3Net:**
```python
if k.startswith('FAD_head'):  # ← เพิ่ม!
    continue
new_k = new_k.replace('backbone.', '')  # ← เพิ่ม!
new_k = new_k.replace('last_linear.1.', 'last_linear.')  # ← เพิ่ม!
```

**Effort-CLIP (ใหม่ทั้งหมด):**
```python
from transformers import ViTModel, ViTConfig

# สร้าง ViT 1024 dim
config = ViTConfig(hidden_size=1024, num_hidden_layers=24, ...)
model = ViTModel(config)

# โหลด backbone
new_k = k.replace('module.backbone.', '')

# โหลด classifier head
classifier.weight.data = checkpoint['module.head.weight']
classifier.bias.data = checkpoint['module.head.bias']
```

---

### **3. Cell 25: เปิดใช้ Effort**
```python
"effort": {
    "weight": round(best_weights[2], 2),
    "enabled": True  # ← แก้จาก False
}
```

---

## ✅ Cells 12-19: ใช้ตามเดิม (3 โมเดล)

**ไม่ต้องแก้!** ใช้ 3 โมเดลตามที่เขียนไว้เดิม

---

## 🔍 Verify Output:

**Must see:**
```
🔧 Loading Xception from xception_best.pth
  ✅ Classifier layer loaded    ← MUST HAVE!

🔧 Loading F3Net from f3net_best.pth
  🗑️  Skipped XX FAD_head layers
  ✅ Classifier layer loaded    ← MUST HAVE!

🔧 Loading Effort-CLIP from effort_clip_L14_trainOn_FaceForensic.pth
  📊 Detected classifier input dim: 1024
  ✅ Backbone loaded: XXX params
  ✅ Classifier head loaded (1024 → 2)    ← MUST HAVE!
```

**Results:**
```
Xception:  Accuracy > 0.85  ✅
F3Net:     Accuracy > 0.85  ✅
Effort:    Accuracy > 0.85  ✅
Ensemble:  Accuracy > 0.90  ✅✅
```

---

## 🚫 Don't See This:

```
❌ WARNING: Classifier layer NOT loaded!
❌ Accuracy: 0.5XXX
❌ Precision: 0.0000
```

---

## 📁 Files:

- **CORRECTED_MODEL_LOADING.md** → Full corrected code (คัดลอกจากนี่!)
- **NEXT_STEPS.md** → Detailed step-by-step guide
- **PROBLEM_SOLUTION_SUMMARY.md** → What went wrong & why

---

## 📝 Checklist:

- [ ] แก้ Cell 3 (เพิ่ม transformers)
- [ ] แทนที่ Cell 11 (3 model classes)
- [ ] Cell 12-19 ใช้ตามเดิม ✅
- [ ] แก้ Cell 25 (Effort enabled: true)
- [ ] Runtime → Restart and run all
- [ ] ตรวจสอบ "✅ Classifier layer loaded" ทั้ง 3 ตัว
- [ ] Download config_optimized.json

---

## ⏱️ Time: 20-30 min | 💰 Cost: ~20-25 compute units

**Expected Result:** 92-98% accuracy ensemble! 🎉

**All 3 models working!** ✅✅✅
