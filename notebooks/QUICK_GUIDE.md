# ⚡ คำสั่งด่วน - ทดสอบ Weight ด้วย Processed Data

## 📁 Dataset ของคุณ:
```
processed_data_split/
├── train/
├── val/
└── test/          ← ใช้อันนี้
    ├── original/   (Real)
    ├── Deepfakes/  (Fake)
    ├── FaceSwap/   (Fake)
    └── Face2Face/  (Fake)
```

---

## 🚀 ขั้นตอน (5 ขั้นตอน)

### 1️⃣ Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2️⃣ ตั้งค่า Path
```python
# ปรับ path ตามที่เก็บจริง
BASE_PATH = '/content/drive/MyDrive/DeepfakeProject/processed_data_split'
WEIGHTS_PATH = '/content/drive/MyDrive/DeepfakeProject/model_weights'

# หรืออัปโหลด weights ใน Colab
# WEIGHTS_PATH = '/content'
```

### 3️⃣ อัปโหลด Weights (ถ้ายังไม่มีใน Drive)
```python
from google.colab import files
uploaded = files.upload()
# เลือก 3 ไฟล์:
# - xception_best.pth
# - f3net_best.pth
# - effort_clip_L14_trainOn_FaceForensic.pth
```

### 4️⃣ Run All Cells
```
Runtime → Run all
```

### 5️⃣ ดาวน์โหลดผลลัพธ์
```python
from google.colab import files
files.download('config_optimized.json')     # ← สำคัญ!
files.download('weight_optimization_report.json')
files.download('model_comparison.png')
```

---

## ⚡ ใช้เวลา & Compute Units

**ใช้เวลา:** 15-25 นาที (ขึ้นอยู่กับจำนวนภาพ)

**Compute Units:**
- น้อย (< 500 images): ~5-10 units
- กลาง (500-1000 images): ~10-15 units
- เยอะ (> 1000 images): ~15-20 units

**คุณมี 60.24 units → เพียงพอ!** ✅

---

## 📊 Output ที่จะได้

### 1. **Optimal Weights**
```json
{
  "xception": 0.XX,
  "f3net": 0.XX,
  "effort_clip": 0.XX
}
```

### 2. **Performance Metrics**
```
Xception:    Accuracy: 0.9XXX, F1: 0.9XXX
F3Net:       Accuracy: 0.9XXX, F1: 0.9XXX
Effort-CLIP: Accuracy: 0.9XXX, F1: 0.9XXX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ensemble:    Accuracy: 0.9XXX, F1: 0.9XXX ⭐
```

### 3. **Config File**
`config_optimized.json` → คัดลอกไป `backend/app/config.json`

### 4. **Visualizations**
- `model_comparison.png` - เปรียบเทียบโมเดล
- `top10_configurations.png` - Top 10 weights

---

## 🔧 Tips

### ลดการใช้ Compute Units:
```python
# จำกัดจำนวนภาพ (ใน cell โหลด dataset)
test_data = test_data[:500]  # ใช้แค่ 500 ภาพแรก
```

### เพิ่ม GPU Speed:
```python
# Runtime → Change runtime type → GPU: T4 (หรือ A100)
```

### Debug:
```python
# ถ้า error → เช็คว่า path ถูกต้อง
!ls /content/drive/MyDrive/DeepfakeProject/processed_data_split/test
```

---

## ✅ Checklist

- [ ] Mount Drive สำเร็จ
- [ ] ปรับ `BASE_PATH` ให้ถูกต้อง
- [ ] มี 3 model weights files
- [ ] เลือก GPU runtime (T4)
- [ ] มี compute units เหลือ (> 10)

---

## 🎯 หลัง Run เสร็จ

### 1. ดาวน์โหลด config
```bash
# ใน local machine
cd deepfake-detection/backend/app
cp config.json config.json.backup
# แทนที่ด้วยไฟล์ที่ download มา
```

### 2. Restart Backend
```bash
cd backend
python -m uvicorn app.main:app --reload
```

### 3. ทดสอบ
```bash
# เปิด browser
http://localhost:3000
# อัปโหลดภาพทดสอบ
```

---

## 📚 ไฟล์ที่ต้องใช้

1. **Notebook:** `Quick_Weight_Optimization.ipynb`
2. **Model Weights:** 3 ไฟล์ (.pth)
3. **Dataset:** processed_data_split/ (มีอยู่แล้ว ✅)

---

## 💡 คาดหวังผลลัพธ์

จาก FaceForensics++ papers:
- Individual models: ~95-97% accuracy
- **Ensemble: ~98%+ accuracy** 🎯

ถ้าได้ใกล้เคียง → **โมเดลพร้อมใช้งานจริง!**

---

## ❓ FAQ

**Q: ต้องใช้ทั้ง train/val/test หรือไม่?**
A: ไม่ ใช้แค่ **test set** เพื่อประเมินความแม่นยำ

**Q: ต้อง extract frames หรือไม่?**
A: ไม่! คุณมี processed images พร้อมแล้ว ✅

**Q: ถ้า accuracy ต่ำกว่า 90%?**
A: ตรวจสอบ:
- Model weights ถูกต้องหรือไม่
- Dataset balance (real:fake ratio)
- Image quality

**Q: จำเป็นต้องใช้ weights ใหม่หรือไม่?**
A: ถ้า ensemble ดีขึ้น > 1% → ควรใช้
   ถ้าดีขึ้น < 0.5% → weights เดิมก็ใช้ได้

---

## 🚀 Ready!

**คัดลอก notebook ไป Colab แล้ว Run ได้เลย!**

ใช้เวลาแค่ **15-25 นาที** → ได้ **optimal weights** ทันที! 🎉

---

**Updated:** 24 ตุลาคม 2025
**Notebook:** `Quick_Weight_Optimization.ipynb`
