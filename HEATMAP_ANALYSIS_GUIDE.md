# 🔥 Heatmap Analysis Guide
## ระบบวิเคราะห์ Heatmap และระบุบริเวณที่มีความผิดปกติ

> **อัปเดตล่าสุด:** 2025-10-21
> **สถานะ:** ✅ ใช้งานได้แล้ว

---

## 📋 สารบัญ

1. [ภาพรวมระบบ](#ภาพรวมระบบ)
2. [วิธีการทำงาน](#วิธีการทำงาน)
3. [บริเวณที่วิเคราะห์](#บริเวณที่วิเคราะห์)
4. [การแปลผล](#การแปลผล)
5. [ตัวอย่างการใช้งาน](#ตัวอย่างการใช้งาน)

---

## 🎯 ภาพรวมระบบ

ระบบ **Heatmap Analysis** เป็นฟีเจอร์ใหม่ที่วิเคราะห์ Grad-CAM heatmap อัตโนมัติ และบอกผู้ใช้ว่า:

- ✅ **โมเดลตรวจจับความผิดปกติบริเวณไหนของใบหน้า** (เช่น ปาก, ตา, จมูก, ขอบใบหน้า)
- ✅ **ระดับความสนใจ** (สูง, ปานกลาง, ต่ำ)
- ✅ **คำอธิบายเฉพาะแต่ละบริเวณ** (ทำไมถึงพบความผิดปกติ)
- ✅ **สรุปผลเป็นภาษาไทย** (เข้าใจง่าย)

---

## ⚙️ วิธีการทำงาน

### Backend (Python)

```
📁 backend/app/services/heatmap_analyzer.py
```

**ขั้นตอน:**

1. **รับ Grad-CAM heatmap** (224x224 pixels)
2. **แบ่งใบหน้าออกเป็น 9 บริเวณ:**
   - หน้าผาก (Forehead)
   - ตาซ้าย/ขวา (Left/Right Eye)
   - จมูก (Nose)
   - ปาก (Mouth)
   - แก้มซ้าย/ขวา (Left/Right Cheek)
   - คาง (Chin)
   - ขอบใบหน้า (Face Boundary)

3. **คำนวณค่าความสนใจ** ในแต่ละบริเวณ:
   - `avg_attention`: ค่าเฉลี่ย
   - `max_attention`: ค่าสูงสุด
   - `attention_level`: high / moderate / low

4. **จัดอันดับ** บริเวณตามความสนใจ (สูง → ต่ำ)

5. **สร้างคำอธิบาย** เฉพาะแต่ละบริเวณ:
   - ปาก → "มักพบความผิดปกติเนื่องจากการสร้างภาพปากขณะพูดที่ไม่เป็นธรรมชาติ"
   - ตา → "มักมีปัญหาเรื่องแสงสะท้อน, ขนตา"
   - ขอบใบหน้า → "มักเป็นจุดที่พบ artifact จากการผสมภาพ"

### Frontend (React/TypeScript)

```
📁 frontend/app/components/HeatmapViewer.tsx
```

**แสดงผล:**

1. **สรุปผลหลัก** (Summary):
   - "โมเดลตรวจพบความผิดปกติสูงสุดที่ **ปาก** (ระดับความสนใจ: 78.3%)"

2. **Top 3 บริเวณ** (Ranking):
   - 🥇 ปาก - 78.3%
   - 🥈 ขอบใบหน้า - 65.1%
   - 🥉 ตาซ้าย - 54.2%

3. **บริเวณที่ตรวจพบความผิดปกติ** (Suspicious Regions - เฉพาะ FAKE):
   - 🚨 แสดงรายละเอียดบริเวณที่มี attention > 60%

4. **ตารางทุกบริเวณ** (Expandable):
   - แสดงข้อมูลครบทุก 9 บริเวณ

---

## 📍 บริเวณที่วิเคราะห์

### ตำแหน่ง Bounding Boxes (224x224 pixels)

| บริเวณ | พิกัด (x1, y1, x2, y2) | คำอธิบาย |
|--------|----------------------|----------|
| **หน้าผาก** | (50, 20, 174, 80) | อาจมี texture หรือแสงไม่สม่ำเสมอ |
| **ตาซ้าย** | (50, 70, 100, 110) | แสงสะท้อน, ขนตา, การกระพริบตา |
| **ตาขวา** | (124, 70, 174, 110) | แสงสะท้อน, ขนตา, การกระพริบตา |
| **จมูก** | (90, 100, 134, 150) | การเชื่อมต่อ, เงาผิดปกติ |
| **ปาก** | (80, 150, 144, 190) | การสร้างภาพปากขณะพูด, ฟัน |
| **แก้มซ้าย** | (30, 110, 80, 160) | ผิวหนังไม่สม่ำเสมอ |
| **แก้มขวา** | (144, 110, 194, 160) | ผิวหนังไม่สม่ำเสมอ |
| **คาง** | (80, 180, 144, 220) | การเชื่อมต่อกับคอ |
| **ขอบใบหน้า** | ซ้าย + ขวา | Blending artifacts |

### เกณฑ์การตัดสิน

```python
HIGH_ATTENTION_THRESHOLD = 0.6      # >= 60% = ผิดปกติสูง
MODERATE_ATTENTION_THRESHOLD = 0.4  # >= 40% = ปานกลาง
```

---

## 📊 การแปลผล

### กรณี: ภาพ **FAKE**

#### ตัวอย่างผลลัพธ์:

```json
{
  "is_fake": true,
  "top_3_regions": [
    {
      "region_name_th": "ปาก",
      "avg_attention": 0.783,
      "attention_level": "high"
    },
    {
      "region_name_th": "ขอบใบหน้า",
      "avg_attention": 0.651,
      "attention_level": "high"
    },
    {
      "region_name_th": "ตาซ้าย",
      "avg_attention": 0.542,
      "attention_level": "moderate"
    }
  ],
  "explanation": {
    "summary_th": "โมเดลตรวจพบความผิดปกติสูงสุดที่ **ปาก** (ระดับความสนใจ: 78.3%)",
    "specific_explanation": "บริเวณปากมักพบความผิดปกติใน Deepfake เนื่องจากการสร้างภาพปากขณะพูดที่ไม่เป็นธรรมชาติ หรือฟันที่ผิดปกติ"
  }
}
```

#### การแสดงผลบน UI:

```
┌─────────────────────────────────────────────────┐
│ 🔍 การวิเคราะห์จาก AI                           │
├─────────────────────────────────────────────────┤
│                                                 │
│ โมเดลตรวจพบความผิดปกติสูงสุดที่ **ปาก**        │
│ (ระดับความสนใจ: 78.3%)                          │
│                                                 │
│ บริเวณปากมักพบความผิดปกติใน Deepfake...        │
│                                                 │
├─────────────────────────────────────────────────┤
│ บริเวณที่โมเดลให้ความสนใจมากที่สุด:             │
│                                                 │
│ 🥇 ปาก (Mouth)               78.3%  [สูง]      │
│ 🥈 ขอบใบหน้า (Face Boundary)  65.1%  [สูง]      │
│ 🥉 ตาซ้าย (Left Eye)          54.2%  [ปานกลาง]  │
│                                                 │
├─────────────────────────────────────────────────┤
│ 🚨 บริเวณที่ตรวจพบความผิดปกติ:                  │
│                                                 │
│ 🔴 ปาก: ระดับความสนใจ 78.3% -                  │
│    บริเวณปากมักพบความผิดปกติใน Deepfake...     │
│                                                 │
│ 🔴 ขอบใบหน้า: ระดับความสนใจ 65.1% -            │
│    ขอบใบหน้ามักเป็นจุดที่พบ artifact...         │
└─────────────────────────────────────────────────┘
```

### กรณี: ภาพ **REAL**

```
โมเดลพบว่าภาพนี้เป็นธรรมชาติ
การกระจายความสนใจปกติที่ ตาซ้าย, ตาขวา, จมูก

✅ ตาซ้าย: ระดับความสนใจ 45.2% (ปกติ)
✅ ตาขวา: ระดับความสนใจ 43.8% (ปกติ)
✅ จมูก: ระดับความสนใจ 38.1% (ปกติ)
```

---

## 💻 ตัวอย่างการใช้งาน

### 1. Backend API Response

```python
# เมื่อ API detect ภาพ
{
  "prediction": "FAKE",
  "confidence": 0.92,
  "gradcam": "data:image/jpeg;base64,...",

  # ✨ ส่วนใหม่!
  "heatmap_analysis": {
    "is_fake": true,
    "regions": [...],  # ทุกบริเวณ (9 บริเวณ)
    "suspicious_regions": [...],  # เฉพาะที่น่าสงสัย
    "top_3_regions": [...],  # Top 3
    "explanation": {
      "summary_th": "...",
      "specific_explanation": "..."
    }
  }
}
```

### 2. Frontend Component

```tsx
// ResultDisplay.tsx
<HeatmapViewer
  originalImage={result.original_image}
  heatmapImage={result.gradcam}
  isFake={isFake}
  heatmapAnalysis={result.heatmap_analysis}  // ✨ ส่ง prop ใหม่
/>
```

### 3. การเพิ่ม Visual Annotations (Optional)

```tsx
import AnnotatedHeatmap from './AnnotatedHeatmap';

<AnnotatedHeatmap
  heatmapImage={result.gradcam}
  suspiciousRegions={result.heatmap_analysis.suspicious_regions}
  showAnnotations={true}
/>
```

จะวาด bounding boxes และ labels บน heatmap โดยตรง!

---

## 🎨 UI Components

### ไฟล์ที่เกี่ยวข้อง:

```
frontend/app/components/
├── HeatmapViewer.tsx          # แสดง heatmap + การวิเคราะห์
├── AnnotatedHeatmap.tsx       # (Optional) วาด annotations
└── ResultDisplay.tsx          # เชื่อมต่อทุกอย่าง
```

### Features:

1. **Color-coded Badges:**
   - 🔴 สูง (High) - ความสนใจ >= 60%
   - 🟡 ปานกลาง (Moderate) - ความสนใจ >= 40%
   - 🟢 ต่ำ (Low) - ความสนใจ < 40%

2. **Ranking Medals:**
   - 🥇 อันดับ 1 - สีทอง
   - 🥈 อันดับ 2 - สีเงิน
   - 🥉 อันดับ 3 - สีทองแดง

3. **Expandable Table:**
   - ดูการวิเคราะห์ทุกบริเวณ (9 บริเวณ) ▼

4. **Responsive Design:**
   - Mobile-friendly
   - Tailwind CSS

---

## 🚀 การทดสอบ

### ทดสอบด้วย Backend:

```bash
cd backend
python -c "
from services.heatmap_analyzer import HeatmapAnalyzer
import numpy as np

analyzer = HeatmapAnalyzer()

# สร้าง mock heatmap
heatmap = np.random.rand(224, 224)

# วิเคราะห์
result = analyzer.analyze_heatmap(heatmap, is_fake=True)

print('Top 3 Regions:')
for r in result['top_3_regions']:
    print(f\"  {r['region_name_th']}: {r['avg_attention']:.1%}\")

print(f\"\\nSummary: {result['explanation']['summary_th']}\")
"
```

### ทดสอบด้วย Frontend:

1. อัปโหลดภาพ Deepfake
2. ดูผลลัพธ์ใน "การวิเคราะห์จาก AI"
3. ตรวจสอบ:
   - ✅ มีการแสดง Top 3 บริเวณ
   - ✅ มีคำอธิบายเป็นภาษาไทย
   - ✅ มี badges แสดงระดับความสนใจ
   - ✅ มีตารางทุกบริเวณ (expandable)

---

## 🔧 Troubleshooting

### ปัญหา: ไม่แสดงการวิเคราะห์

**สาเหตุ:**
- `heatmap_analysis` เป็น `null` หรือ `undefined`

**แก้ไข:**
1. ตรวจสอบ Backend:
   ```python
   # backend/app/services/detection.py
   # ต้องมี HeatmapAnalyzer initialized
   if self.heatmap_analyzer:
       heatmap_analysis = self.heatmap_analyzer.analyze_heatmap(cam, is_fake)
   ```

2. ตรวจสอบ API Response:
   ```bash
   curl http://localhost:8000/api/detect/image -F "file=@test.jpg"
   # ดูว่ามี "heatmap_analysis" ใน response หรือไม่
   ```

### ปัญหา: ค่าความสนใจต่ำเกินไป

**สาเหตุ:**
- Heatmap ไม่ชัดเจน / มี noise มาก

**แก้ไข:**
- ปรับ threshold:
   ```python
   # backend/app/services/heatmap_analyzer.py
   HIGH_ATTENTION_THRESHOLD = 0.5  # ลดจาก 0.6
   ```

---

## 📚 เอกสารเพิ่มเติม

- [TECHNICAL_SPECIFICATIONS.md](TECHNICAL_SPECIFICATIONS.md) - สถาปัตยกรรมระบบ
- [ARCHITECTURE.md](ARCHITECTURE.md) - โครงสร้างโปรเจค
- [HOW_TO_RUN.md](HOW_TO_RUN.md) - วิธีรันโปรเจค

---

## ✅ สรุป

### สิ่งที่ได้เพิ่มเข้ามา:

1. ✅ **Backend Service:** `heatmap_analyzer.py`
   - วิเคราะห์ 9 บริเวณของใบหน้า
   - คำนวณความสนใจแต่ละบริเวณ
   - สร้างคำอธิบายอัตโนมัติ

2. ✅ **Frontend Component:** `HeatmapViewer.tsx` (Updated)
   - แสดงการวิเคราะห์เป็นภาษาไทย
   - Top 3 ranking พร้อม medals
   - Suspicious regions พร้อมคำอธิบาย
   - Expandable table ทุกบริเวณ

3. ✅ **Optional Component:** `AnnotatedHeatmap.tsx`
   - วาด bounding boxes บน heatmap
   - แสดง labels ของแต่ละบริเวณ

4. ✅ **Integration:** `detection.py` + `ResultDisplay.tsx`
   - เชื่อมต่อ backend → frontend
   - ส่งข้อมูลผ่าน API response

---

**🎉 ตอนนี้ระบบสามารถบอกผู้ใช้ได้ว่า "โมเดลตรวจจับความผิดปกติบริเวณไหนของใบหน้า" แล้ว!**

---

**ผู้พัฒนา:** Claude Code
**วันที่:** 2025-10-21
**เวอร์ชัน:** 1.0
