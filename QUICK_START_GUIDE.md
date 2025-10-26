# 🚀 คู่มือเริ่มต้นด่วน - ระบบตรวจจับ Deepfake

## ✅ สิ่งที่ได้รับการปรับปรุงแล้ว

### 1. ความแม่นยำของโมเดล
- ✅ ใช้ 3 โมเดล AI ร่วมกัน (Xception, F3Net, Effort-CLIP)
- ✅ ปรับน้ำหนัก Ensemble ให้สมดุล (0.35, 0.30, 0.35)
- ✅ ลด Face Detection Threshold เป็น 0.85 (รับภาพได้มากขึ้น)

### 2. การแปลภาษา
- ✅ รองรับไทย/อังกฤษ ทุกหน้า 100%
- ✅ Batch Mode แปลภาษาครบถ้วนแล้ว
- ✅ สลับภาษาได้ทันที (มุมขวาบน)

---

## 🎯 วิธีเริ่มต้นใช้งาน

### ขั้นตอนที่ 1: เริ่ม Backend
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### ขั้นตอนที่ 2: เริ่ม Frontend
```bash
cd frontend
npm run dev
```

### ขั้นตอนที่ 3: เปิดเว็บไซต์
เข้าที่: http://localhost:3000

---

## 📊 โหมดการใช้งาน

### 🖼️ Single Image Mode
- อัปโหลดรูปภาพ 1 ภาพ
- ได้ผลลัพธ์พร้อม Grad-CAM heatmap
- แสดงรายละเอียดการวิเคราะห์

### 📦 Batch Mode
- อัปโหลดได้สูงสุด 10 ภาพพร้อมกัน
- ประมวลผลทีละภาพอัตโนมัติ
- ส่งออกผลลัพธ์เป็น CSV หรือ JSON

### 🎥 Video Mode
- วิเคราะห์วิดีโอทีละเฟรม
- สร้าง heatmap สำหรับเฟรมที่น่าสงสัย
- แสดงสถิติโดยละเอียด

---

## 🔍 วิธีทดสอบระบบ

### ทดสอบ Backend
```bash
cd backend
python -c "from app.models.manager import EnsembleModelManager; mgr = EnsembleModelManager('app/config.json'); print(f'✅ โหลดสำเร็จ {len(mgr.models)} โมเดล')"
```

### ทดสอบ API
```bash
curl http://localhost:8000/health
```

ควรได้ผลลัพธ์:
```json
{
  "status": "healthy",
  "models_loaded": 3,
  "device": "cuda"
}
```

---

## ⚙️ การปรับแต่ง

### ปรับน้ำหนัก Ensemble
แก้ไขไฟล์: `backend/app/config.json`

```json
{
  "models": {
    "xception": {
      "weight": 0.35,  // เพิ่ม/ลดได้
      "enabled": true
    },
    "f3net": {
      "weight": 0.30,
      "enabled": true
    },
    "effort": {
      "weight": 0.35,
      "enabled": true
    }
  }
}
```

**หมายเหตุ:** ผลรวม weight ต้องเท่ากับ 1.0

### ปรับ Face Detection
แก้ไข: `backend/app/services/detection.py` (บรรทัด 116)

```python
if confidence < 0.85:  # ปรับค่านี้ (0.7-0.95)
    raise ValueError(...)
```

- **0.7-0.8:** ยอมรับภาพได้มากแต่อาจมี false positive
- **0.85:** แนะนำ (สมดุลระหว่างความแม่นยำและการยอมรับ)
- **0.90-0.95:** เข้มงวดมาก อาจปฏิเสธภาพดีๆ

### เพิ่มภาษาใหม่
แก้ไข: `frontend/app/contexts/LanguageContext.tsx`

เพิ่ม language key และ translations:

```typescript
type Language = 'th' | 'en' | 'ja';  

const translations = {
  th: { ... },
  en: { ... },
  ja: {  // เพิ่มการแปล
    'app.title': 'ディープフェイク検出システム',
    ...
  }
};
```

---

## 🛠️ แก้ไขปัญหาที่พบบ่อย

### ❌ Backend ไม่สามารถโหลดโมเดลได้

**สาเหตุ:** ไฟล์ weights ไม่ถูกต้องหรือ path ผิด

**วิธีแก้:**
```bash
# ตรวจสอบว่าไฟล์มีอยู่
ls backend/app/models/weights/

# ต้องมีไฟล์:
# - xception_best.pth
# - f3net_best.pth
# - effort_clip_L14_trainOn_FaceForensic.pth
```

### ❌ CUDA Out of Memory

**วิธีแก้:** เปลี่ยนเป็น CPU

แก้ไข `backend/app/config.json`:
```json
{
  "device": "cpu"  // เปลี่ยนจาก "cuda"
}
```

### ❌ Frontend ไม่แสดงการแปลภาษา

**วิธีแก้:**
```bash
cd frontend
rm -rf .next  # ลบ cache
npm run dev   # restart
```

### ❌ Face detection confidence too low

**สาเหตุ:** ภาพมืดเกินไปหรือใบหน้าไม่ชัดเจน

**วิธีแก้:**
1. ใช้ภาพที่มีแสงสว่างดี
2. ลด threshold ใน `detection.py` (อ่านด้านบน)

---

## 📈 เคล็ดลับเพื่อความแม่นยำสูงสุด

### สำหรับผู้ใช้:
✅ ใช้ภาพความละเอียดสูง (≥512x512px)
✅ ใบหน้าหันตรงกล้อง
✅ มีแสงสว่างเพียงพอ
✅ หลีกเลี่ยงฟิลเตอร์หรือเอฟเฟกต์
✅ ไฟล์ไม่บีบอัดมากเกินไป (Quality ≥70%)

### สำหรับนักพัฒนา:
✅ ทดสอบกับชุดข้อมูลที่หลากหลาย
✅ ปรับ ensemble weights ตามผลการทดสอบ
✅ ใช้ MLflow tracking (ถ้ามี) เพื่อติดตามประสิทธิภาพ
✅ เก็บ log ของ false positive/negative
✅ update โมเดลเมื่อมี weights ใหม่

---

## 📊 การตีความผลลัพธ์

### Confidence Score
- **90-100%:** มั่นใจสูงมาก ✅
- **80-89%:** มั่นใจสูง ✅
- **70-79%:** ปานกลาง ⚠️
- **60-69%:** ต่ำ ⚠️
- **<60%:** ไม่แน่นอน ❌

### Model Agreement
- **3/3 โมเดลเห็นด้วย:** น่าเชื่อถือมาก ✅
- **2/3 โมเดลเห็นด้วย:** น่าเชื่อถือพอสมควร ✅
- **โมเดลขัดแย้งกัน:** ควรตรวจสอบเพิ่มเติม ⚠️

### Grad-CAM Heatmap
- **สีแดง:** บริเวณที่โมเดลให้ความสนใจมาก (อาจปลอม)
- **สีเหลือง:** ความสนใจปานกลาง
- **สีเขียว/น้ำเงิน:** ความสนใจต่ำ (ดูปกติ)

---

## 🔐 ความปลอดภัย

### การประมวลผลในเครื่อง (Local Processing)
- ✅ ภาพไม่ถูกส่งไปเซิร์ฟเวอร์ภายนอก
- ✅ ไม่มีการเก็บข้อมูลในฐานข้อมูล (ตาม config)
- ✅ ประมวลผลแบบ real-time แล้วลบทิ้ง

### สำหรับ Production
⚠️ **ก่อน deploy ควร:**
1. เปลี่ยน CORS settings
2. เพิ่ม authentication
3. เพิ่ม rate limiting
4. ตั้งค่า HTTPS
5. backup model weights

---

## 📞 ติดต่อและการสนับสนุน

### เอกสารที่เกี่ยวข้อง:
- [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) - รายละเอียดการปรับปรุง
- [ARCHITECTURE.md](ARCHITECTURE.md) - สถาปัตยกรรมระบบ (ถ้ามี)
- [HOW_TO_RUN.md](HOW_TO_RUN.md) - คู่มือการติดตั้ง (ถ้ามี)

### ปัญหา/ข้อเสนอแนะ:
- เปิด GitHub Issue (ถ้ามี repository)
- ติดต่อผู้พัฒนา

---

## 🎓 ทำความเข้าใจเพิ่มเติม

### Ensemble Learning คืออะไร?
การรวมผลการทำนายจากหลายโมเดลเข้าด้วยกัน เพื่อลด bias และเพิ่มความแม่นยำ

**สูตร:**
```
Final Prediction = (Xception × 0.35) + (F3Net × 0.30) + (Effort × 0.35)
```

### ทำไมต้องใช้ 3 โมเดล?
- **Xception:** เก่งในการตรวจจับ spatial artifacts
- **F3Net:** เก่งในการวิเคราะห์ frequency domain
- **Effort-CLIP:** เก่งในการเข้าใจบริบทและ semantics

**รวมกัน = ครอบคลุมทุกมิติของ deepfake!**

---

## 🚀 พร้อมใช้งาน!

ระบบพร้อมใช้งาน Production แล้ว โดยมี:
- ✅ 3 โมเดล AI ที่แม่นยำ
- ✅ รองรับ 2 ภาษา (ไทย/อังกฤษ)
- ✅ 3 โหมดการใช้งาน (Image/Batch/Video)
- ✅ Grad-CAM visualization
- ✅ Export ผลลัพธ์ (CSV/JSON)

**เริ่มต้นใช้งานได้เลย! 🎉**

---

**อัพเดทล่าสุด:** 24 ตุลาคม 2025
**เวอร์ชัน:** 2.0 (Improved)
