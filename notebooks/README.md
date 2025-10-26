# 📓 Colab Notebooks สำหรับ Deepfake Detection

## 🎯 Model_Weight_Optimization.ipynb

### วัตถุประสงค์:
หาค่า **weight ที่เหมาะสม** สำหรับ Ensemble Learning (3 โมเดล)

### ⚡ ประมาณการใช้ Compute Units:
- **5-10 units** สำหรับ dataset ขนาดเล็ก (~100-200 ภาพ)
- **15-20 units** สำหรับ dataset ขนาดกลาง (~500-1000 ภาพ)
- **30-40 units** สำหรับ dataset ขนาดใหญ่ (~2000+ ภาพ)

คุณมี **60.24 units** เหลือ → **เพียงพอ!**

---

## 🚀 วิธีใช้งาน (Step-by-Step)

### 1️⃣ เตรียมข้อมูล

#### Option A: ใช้ Google Drive
```
📁 Google Drive/
  └── test_data/
      ├── real/
      │   ├── real_001.jpg
      │   ├── real_002.jpg
      │   └── ...
      └── fake/
          ├── fake_001.jpg
          ├── fake_002.jpg
          └── ...
```

**แนะนำ:**
- Real images: 50-100 ภาพ
- Fake images: 50-100 ภาพ
- Total: **100-200 ภาพ** (เพื่อประหยัด compute units)

#### Option B: อัปโหลดโดยตรง
ใช้ `files.upload()` ใน Colab เพื่ออัปโหลดไฟล์ทีละน้อย

---

### 2️⃣ อัปโหลด Model Weights

ต้องมี **3 ไฟล์**:
```
1. xception_best.pth
2. f3net_best.pth
3. effort_clip_L14_trainOn_FaceForensic.pth
```

**ขนาดไฟล์รวม:** ประมาณ 500MB - 1GB

**วิธีอัปโหลด:**

```python
# ใน Colab cell
from google.colab import files
uploaded = files.upload()
```

หรือเก็บใน Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# ปรับ path
WEIGHTS_PATH = '/content/drive/MyDrive/model_weights/'
```

---

### 3️⃣ เปิด Colab และ Run

1. **อัปโหลด Notebook**:
   - ไปที่ [Google Colab](https://colab.research.google.com/)
   - File → Upload notebook
   - เลือก `Model_Weight_Optimization.ipynb`

2. **เลือก GPU Runtime**:
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU** (T4 หรือ A100)
   - Save

3. **Run ทีละ Cell**:
   - กด Shift+Enter เพื่อ run แต่ละ cell
   - หรือ Runtime → Run all

---

### 4️⃣ ปรับแต่งตามความต้องการ

#### ลดการใช้ Compute Units:

**A. ลดขนาด Dataset:**
```python
# จำกัดจำนวนภาพ
test_data = test_data[:100]  # ใช้แค่ 100 ภาพแรก
```

**B. ลดความละเอียดของ Grid Search:**
```python
# เปลี่ยนจาก step=0.05 เป็น 0.10
step = 0.10  # ทดสอบน้อยลง แต่เร็วกว่า
```

**C. Skip Face Detection:**
```python
# Comment out face detection (ถ้ารูปครอปใบหน้าแล้ว)
# boxes, _ = face_detector.detect(img)
```

#### เพิ่มความแม่นยำ:

**A. ใช้ Cross-Validation:**
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)
for train_idx, val_idx in kf.split(test_data):
    # Split และทดสอบแต่ละ fold
    ...
```

**B. Fine-tune Grid Search:**
```python
# หลังจากได้ weights คร่าวๆ แล้ว
# ค้นหาแบบละเอียดรอบๆ best weights
step = 0.01  # ละเอียดมากขึ้น
```

---

## 📊 ผลลัพธ์ที่ได้

### 1. **Optimal Weights**
```json
{
  "xception": 0.35,
  "f3net": 0.30,
  "effort_clip": 0.35
}
```

### 2. **Performance Report**
- Individual model metrics
- Ensemble performance
- Top 10 configurations

### 3. **Visualizations**
- `individual_model_performance.png`
- `weight_optimization_heatmap.png`
- `individual_vs_ensemble.png`

### 4. **Ready-to-use Config**
- `config_optimized.json` → คัดลอกไปใช้เลย!

---

## 💡 Tips & Best Practices

### ✅ DO:
- ใช้ dataset ที่หลากหลาย (หลายแหล่ง, หลายเทคนิค)
- ทดสอบกับข้อมูลที่ไม่เคยเห็น (unseen data)
- เก็บ log ทุกครั้งเพื่อเปรียบเทียบ
- ทดสอบกับ real-world data ก่อน deploy

### ❌ DON'T:
- ใช้ dataset เดียวกับที่ train (overfitting!)
- เลือก weights เพียงแค่ accuracy สูง (ดู F1, AUC ด้วย)
- ใช้ dataset ที่เล็กเกินไป (<50 ภาพ/class)
- ลืมทดสอบกับข้อมูลจริงก่อน deploy

### ⚠️ Common Issues:

**1. CUDA Out of Memory**
```python
# แก้: ลด batch size หรือใช้ CPU
device = torch.device('cpu')
```

**2. Model Loading Error**
```python
# ตรวจสอบ path และขนาดไฟล์
!ls -lh models/weights/
```

**3. Dataset Too Small**
```python
# อย่างน้อย 50 ภาพต่อ class
# แนะนำ 100+ ภาพต่อ class
```

---

## 📈 การตีความผลลัพธ์

### Accuracy vs F1 Score:

**เลือก F1 Score** เมื่อ:
- ข้อมูล imbalanced (จำนวน real ≠ fake)
- ต้องการสมดุลระหว่าง precision และ recall
- สำคัญกับทั้ง False Positive และ False Negative

**เลือก Accuracy** เมื่อ:
- ข้อมูลสมดุล (จำนวน real ≈ fake)
- ต้องการความถูกต้องโดยรวม

### AUC (Area Under Curve):
- **>0.95:** Excellent! 🎉
- **0.90-0.95:** Very good ✅
- **0.85-0.90:** Good ✅
- **0.80-0.85:** Fair ⚠️
- **<0.80:** Needs improvement ❌

---

## 🔄 การนำไปใช้งาน

### 1. ดาวน์โหลดไฟล์จาก Colab:
```python
from google.colab import files
files.download('config_optimized.json')
files.download('weight_optimization_report.json')
files.download('individual_vs_ensemble.png')
```

### 2. อัปเดท config ในโปรเจกต์:
```bash
# ใน local machine
cd deepfake-detection/backend/app
cp config.json config.json.backup  # backup เดิม
# แทนที่ด้วย config_optimized.json
cp ~/Downloads/config_optimized.json config.json
```

### 3. Restart Backend:
```bash
cd backend
python -m uvicorn app.main:app --reload
```

### 4. ทดสอบ:
```bash
# ทดสอบโหลดโมเดล
python -c "from app.models.manager import EnsembleModelManager; mgr = EnsembleModelManager(); print('✅ OK')"
```

---

## 📚 เอกสารเพิ่มเติม

- [IMPROVEMENTS_SUMMARY.md](../IMPROVEMENTS_SUMMARY.md) - การปรับปรุงล่าสุด
- [QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md) - คู่มือเริ่มต้นใช้งาน
- [ARCHITECTURE.md](../ARCHITECTURE.md) - สถาปัตยกรรมระบบ

---

## 💬 คำถามที่พบบ่อย (FAQ)

### Q: ต้องใช้ GPU หรือไม่?
**A:** แนะนำให้ใช้ GPU เพื่อความเร็ว แต่สามารถใช้ CPU ได้ (ช้ากว่า 10-20 เท่า)

### Q: Dataset ควรมีขนาดเท่าไหร่?
**A:** อย่างน้อย 50 ภาพต่อ class, แนะนำ 100-200 ภาพ

### Q: ใช้ Compute Units เท่าไหร่?
**A:** ประมาณ 5-20 units ขึ้นอยู่กับขนาด dataset และ grid search resolution

### Q: ผลลัพธ์แม่นยำแค่ไหน?
**A:** ขึ้นอยู่กับคุณภาพ test dataset - ยิ่ง diverse ยิ่งดี

### Q: ต้อง retrain โมเดลหรือไม่?
**A:** ไม่ต้อง! notebook นี้หาแค่ weights สำหรับ ensemble เท่านั้น

---

## 🎯 สรุป

Notebook นี้จะช่วยคุณ:
- ✅ หา **optimal weights** สำหรับ ensemble
- ✅ ประหยัด **compute units** (5-20 units)
- ✅ ได้ **config พร้อมใช้** ทันที
- ✅ เห็น **visualization** ชัดเจน
- ✅ ปรับปรุง **accuracy** ของระบบ

**เริ่มต้นได้เลย! 🚀**

---

**สร้างโดย:** Claude Code
**วันที่:** 24 ตุลาคม 2025
**เวอร์ชัน:** 1.0
