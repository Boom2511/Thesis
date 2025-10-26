# 🎬 FaceForensics++ Testing Guide

## ✅ สิ่งที่คุณมี:

```
📁 datasets/
  ├── original_sequences/youtube/c40/videos/     (300 videos - REAL)
  ├── manipulated_sequences/
  │   ├── Deepfakes/c40/videos/                  (100 videos - FAKE)
  │   ├── FaceSwap/c40/videos/                   (100 videos - FAKE)
  │   └── Face2Face/c40/videos/                  (100 videos - FAKE)
```

**รวม: 600 videos** 🎉

---

## 🎯 แผนการทดสอบ

### **Option 1: ทดสอบแบบเต็ม (แนะนำ)**

**ขั้นตอน:**
1. Extract frames จากทุกวิดีโอ (ทุก 60 frames = 2 วินาที)
2. Crop faces ด้วย MTCNN
3. ทดสอบ 3 โมเดล
4. หา optimal weights

**จำนวน frames ประมาณ:**
- 30 videos × 5 frames × 4 classes = **600 faces**

**ใช้ Compute Units:**
- Extract + Crop: ~10-15 units
- Testing: ~15-20 units
- **รวม: ~30-35 units** ✅ (จาก 60.24)

---

### **Option 2: ทดสอบแบบด่วน**

**ขั้นตอน:**
1. ใช้แค่ 10-15 videos ต่อ class
2. Extract 3-5 frames ต่อวิดีโอ
3. ทดสอบโมเดล

**จำนวน frames:**
- 15 videos × 3 frames × 4 classes = **180 faces**

**ใช้ Compute Units:**
- รวม: **~15-20 units** ⚡

---

## 🚀 วิธีใช้งาน

### 1️⃣ เปิด Notebook:
```python
# ใน Colab
Upload: Extract_and_Test_FaceForensics.ipynb
```

### 2️⃣ ปรับค่าพารามิเตอร์:

```python
# ในCell \"Extract Frames\"
NUM_VIDEOS_PER_CLASS = 30  # จำนวนวิดีโอต่อ class (ปรับได้)
FRAMES_PER_VIDEO = 5       # จำนวน frames ต่อวิดีโอ
FRAME_INTERVAL = 60        # ดึงทุก 60 frames
```

**ตัวอย่าง:**
- **เร็ว:** `NUM_VIDEOS=10`, `FRAMES=3` → 120 faces, ~15 units
- **กลาง:** `NUM_VIDEOS=20`, `FRAMES=5` → 400 faces, ~25 units
- **เต็ม:** `NUM_VIDEOS=30`, `FRAMES=5` → 600 faces, ~35 units

### 3️⃣ อัปโหลด Model Weights:

ต้องมี 3 ไฟล์:
1. `xception_best.pth` (~100MB)
2. `f3net_best.pth` (~100MB)
3. `effort_clip_L14_trainOn_FaceForensic.pth` (~350MB)

**วิธีอัปโหลด:**
- Option A: เก็บใน Google Drive → mount drive
- Option B: ใช้ `files.upload()` ใน Colab

### 4️⃣ Run All Cells:

Runtime → Run all → รอผลลัพธ์

---

## 📊 Output ที่จะได้:

### 1. **Cropped Faces:**
```
📁 datasets/cropped_faces/
  ├── real/     (150-300 images)
  └── fake/     (450-900 images)
```

### 2. **Performance Report:**
```json
{
  "individual_models": {
    "xception": {"accuracy": 0.XXX, "f1": 0.XXX},
    "f3net": {"accuracy": 0.XXX, "f1": 0.XXX},
    "effort": {"accuracy": 0.XXX, "f1": 0.XXX}
  },
  "best_ensemble": {
    "weights": {
      "xception": 0.XX,
      "f3net": 0.XX,
      "effort_clip": 0.XX
    },
    "metrics": {
      "accuracy": 0.XXX,
      "f1": 0.XXX,
      "auc": 0.XXX
    }
  }
}
```

### 3. **Config File:**
`config_optimized.json` → พร้อมใช้ทันที!

### 4. **Visualizations:**
- Individual model performance
- Weight optimization heatmap
- Individual vs Ensemble comparison
- Sample cropped faces

---

## 💡 Tips สำหรับ FaceForensics++

### ✅ ข้อดี:
- **Dataset มาตรฐาน** - ใช้ในงานวิจัยทั่วไป
- **หลากหลาย** - 4 เทคนิค deepfake
- **คุณภาพดี** - c40 compression (สมดุลระหว่างคุณภาพและขนาด)
- **มีวิดีโอเยอะ** - 600 videos เพียงพอสำหรับการทดสอบ

### ⚠️ ข้อควรระวัง:
- **ไม่ใช่ training set** - ใช้สำหรับ evaluation เท่านั้น
- **อาจมีภาพที่ตรวจจับใบหน้าไม่ได้** - ปกติ ~10-15%
- **Compression artifacts** - อาจส่งผลต่อความแม่นยำ
- **Face detection threshold** - ปรับเป็น 0.90 เพื่อคุณภาพ

### 🎯 แนะนำ:
1. **เริ่มจาก 10-15 videos ก่อน** เพื่อทดสอบ pipeline
2. **ตรวจสอบ sample images** หลัง crop เพื่อดูคุณภาพ
3. **ปรับ face detection threshold** ถ้าตรวจจับใบหน้าได้น้อยเกินไป
4. **เก็บ extracted frames** ไว้ใน Drive เพื่อใช้ซ้ำได้

---

## 🔧 Troubleshooting

### ❌ ปัญหา: Extract frames ช้า
**แก้:** ลด `NUM_VIDEOS_PER_CLASS` หรือ `FRAMES_PER_VIDEO`

### ❌ ปัญหา: ตรวจจับใบหน้าได้น้อย
**แก้:** ลด `MIN_CONFIDENCE` จาก 0.90 → 0.85 หรือ 0.80

### ❌ ปัญหา: CUDA Out of Memory
**แก้:**
```python
device = torch.device('cpu')
# หรือ
face_detector = MTCNN(device='cpu')
```

### ❌ ปัญหา: Video codec error
**แก้:**
```python
!apt-get update
!apt-get install -y ffmpeg
```

---

## 📈 ผลลัพธ์ที่คาดหวัง

### Baseline Performance:
จาก FaceForensics++ papers:
- **Xception:** ~95% accuracy
- **F3Net:** ~96% accuracy
- **Effort-CLIP:** ~97% accuracy
- **Ensemble:** ~**98%+ accuracy** 🎯

### Your Results:
อาจแตกต่างขึ้นอยู่กับ:
- จำนวน test samples
- Compression level (c40)
- Face detection threshold
- Mix ของ fake types

---

## 🎓 เข้าใจ Dataset

### FaceForensics++ Types:

**1. Deepfakes:**
- Face swap using deep learning
- High quality, realistic
- Most challenging to detect

**2. FaceSwap:**
- Traditional face swap
- Visible artifacts
- Easier to detect

**3. Face2Face:**
- Facial reenactment
- Transfer expressions
- Medium difficulty

**4. Original (Real):**
- YouTube videos
- No manipulation
- Ground truth

### Why Test All 4?
- **Generalization:** โมเดลต้องตรวจจับได้หลากหลาย
- **Robustness:** ไม่ bias ต่อเทคนิคใดเทคนิคหนึ่ง
- **Real-world:** fake ในโลกจริงมีหลายแบบ

---

## 📊 การวิเคราะห์ผล

### ถ้า Accuracy สูง (>95%):
✅ โมเดลเรียนรู้ patterns ได้ดี
✅ FaceForensics++ เหมาะสำหรับ evaluation
✅ พร้อม deploy (แต่ควรทดสอบ real-world data ก่อน)

### ถ้า Accuracy ปานกลาง (85-95%):
⚠️ ยังพอใช้ได้
⚠️ อาจต้องปรับ weights
⚠️ ควรเพิ่ม test samples

### ถ้า Accuracy ต่ำ (<85%):
❌ มีปัญหา - ตรวจสอบ:
- Face detection quality
- Model weights compatibility
- Preprocessing pipeline
- Dataset balance

---

## ✅ Checklist ก่อน Run

- [ ] มี GPU runtime (T4 หรือดีกว่า)
- [ ] มี 3 model weights files
- [ ] Mount Google Drive สำเร็จ
- [ ] Dataset path ถูกต้อง
- [ ] มี compute units เหลือ (>30 units)
- [ ] ตรวจสอบ disk space (~2-3GB)

---

## 🚀 Ready to Go!

**คำแนะนำสุดท้าย:**
1. เริ่มจากแบบ **Quick Test** (10 videos) ก่อน
2. ดู sample cropped faces เพื่อตรวจสอบคุณภาพ
3. ถ้าพอใจ ค่อย run แบบเต็ม (30 videos)
4. เก็บผลลัพธ์และ config ไว้

**ใช้เวลาประมาณ:**
- Quick: 20-30 นาที
- Full: 45-60 นาที

**ขอให้โชคดี! 🎉**

---

**Updated:** 24 ตุลาคม 2025
**Notebook:** `Extract_and_Test_FaceForensics.ipynb`
