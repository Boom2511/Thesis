# 🔧 แก้ไขการโหลดโมเดล - Model Weights ไม่ทำงาน

## 🐛 ปัญหาที่พบ:

1. **Xception & F3Net:** ทำนาย 0.50:0.50 ทุกภาพ (ไม่ได้เรียนรู้)
2. **Effort-CLIP:** Classifier head ไม่ถูกต้อง

---

## ✅ วิธีแก้ไข

### **Step 1: ตรวจสอบ Checkpoint Structure**

เพิ่ม cell นี้ก่อนโหลดโมเดล:

```python
import torch

# ตรวจสอบโครงสร้าง checkpoint
def inspect_checkpoint(path, name):
    print(f"\n{'='*60}")
    print(f"📦 Inspecting {name}")
    print('='*60)

    checkpoint = torch.load(path, map_location='cpu')

    if isinstance(checkpoint, dict):
        print(f"Type: dict")
        print(f"Keys: {list(checkpoint.keys())}")

        # หา state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"\n✅ Found 'model' key")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\n✅ Found 'state_dict' key")
        else:
            state_dict = checkpoint
            print(f"\n⚠️  Using checkpoint directly as state_dict")

        # แสดง layer แรกและสุดท้าย
        keys = list(state_dict.keys())
        print(f"\nTotal parameters: {len(keys)}")
        print(f"\nFirst 5 layers:")
        for k in keys[:5]:
            print(f"  {k}: {state_dict[k].shape}")
        print(f"\nLast 5 layers:")
        for k in keys[-5:]:
            print(f"  {k}: {state_dict[k].shape}")

    else:
        print(f"Type: {type(checkpoint)}")
        print(f"⚠️  Not a dict - might be state_dict directly")

    return checkpoint

# ตรวจสอบ 3 ไฟล์
xception_ckpt = inspect_checkpoint(XCEPTION_PATH, 'Xception')
f3net_ckpt = inspect_checkpoint(F3NET_PATH, 'F3Net')
effort_ckpt = inspect_checkpoint(EFFORT_PATH, 'Effort-CLIP')
```

---

### **Step 2: แก้โค้ดโหลด Xception**

แทนที่ class `XceptionModel` เดิมด้วย:

```python
class XceptionModel:
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model = self._load_model(weights_path)
        self.model.eval()

    def _load_model(self, weights_path: str) -> nn.Module:
        print(f"\n🔧 Loading Xception from {Path(weights_path).name}")

        # สร้างโมเดล
        model = timm.create_model('xception', pretrained=False, num_classes=2)

        # โหลด checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')

        # หา state_dict
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("  ✅ Using checkpoint['model']")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("  ✅ Using checkpoint['state_dict']")
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
                print("  ✅ Using checkpoint['net']")
            else:
                state_dict = checkpoint
                print("  ⚠️  Using checkpoint as state_dict")
        else:
            state_dict = checkpoint
            print("  ⚠️  Checkpoint is state_dict directly")

        # ทำความสะอาด keys
        new_state_dict = {}
        for k, v in state_dict.items():
            # ลบ prefix
            new_k = k.replace('module.', '')
            new_k = new_k.replace('model.', '')
            new_k = new_k.replace('encoder.', '')

            # Map classifier layer
            if 'fc.' in new_k:
                new_k = new_k.replace('fc.', 'last_linear.')
            elif 'classifier.' in new_k:
                new_k = new_k.replace('classifier.', 'last_linear.')
            elif 'head.' in new_k:
                new_k = new_k.replace('head.', 'last_linear.')

            new_state_dict[new_k] = v

        print(f"  📊 Loaded {len(new_state_dict)} parameters")

        # โหลดเข้าโมเดล
        result = model.load_state_dict(new_state_dict, strict=False)

        if result.missing_keys:
            print(f"  ⚠️  Missing keys: {len(result.missing_keys)}")
            if len(result.missing_keys) <= 5:
                for k in result.missing_keys:
                    print(f"     - {k}")

        if result.unexpected_keys:
            print(f"  ⚠️  Unexpected keys: {len(result.unexpected_keys)}")

        # ตรวจสอบว่า classifier โหลดหรือไม่
        classifier_loaded = any('last_linear' in k or 'fc' in k for k in new_state_dict.keys())
        if classifier_loaded:
            print("  ✅ Classifier layer loaded")
        else:
            print("  ❌ WARNING: Classifier layer NOT loaded!")
            print("  → Model will use random weights for final layer")

        model.to(self.device)
        return model

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor):
        image_tensor = image_tensor.to(self.device)
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        # ตรวจสอบว่า class 0 = REAL หรือ FAKE
        # จาก output ดูเหมือนว่า:
        # Class 0 = REAL
        # Class 1 = FAKE
        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

        return fake_prob, real_prob
```

---

### **Step 3: แก้โค้ดโหลด F3Net**

เหมือน Xception (ใช้โค้ดเดียวกัน):

```python
class F3NetModel:
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model = self._load_model(weights_path)
        self.model.eval()

    def _load_model(self, weights_path: str) -> nn.Module:
        print(f"\n🔧 Loading F3Net from {Path(weights_path).name}")

        # F3Net ใช้ Xception architecture
        model = timm.create_model('xception', pretrained=False, num_classes=2)

        checkpoint = torch.load(weights_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("  ✅ Using checkpoint['model']")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("  ✅ Using checkpoint['state_dict']")
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
                print("  ✅ Using checkpoint['net']")
            else:
                state_dict = checkpoint
                print("  ⚠️  Using checkpoint as state_dict")
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '').replace('model.', '').replace('encoder.', '')

            if 'fc.' in new_k:
                new_k = new_k.replace('fc.', 'last_linear.')
            elif 'classifier.' in new_k:
                new_k = new_k.replace('classifier.', 'last_linear.')
            elif 'head.' in new_k:
                new_k = new_k.replace('head.', 'last_linear.')

            new_state_dict[new_k] = v

        print(f"  📊 Loaded {len(new_state_dict)} parameters")

        result = model.load_state_dict(new_state_dict, strict=False)

        classifier_loaded = any('last_linear' in k or 'fc' in k for k in new_state_dict.keys())
        if classifier_loaded:
            print("  ✅ Classifier layer loaded")
        else:
            print("  ❌ WARNING: Classifier layer NOT loaded!")

        model.to(self.device)
        return model

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor):
        image_tensor = image_tensor.to(self.device)
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()
        return fake_prob, real_prob
```

---

### **Step 4: แก้โค้ดโหลด Effort-CLIP**

```python
class EffortModel:
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.clip_model, self.preprocess = self._load_clip()
        self.classifier = self._load_classifier(weights_path)

        self.clip_model.eval()
        self.classifier.eval()

    def _load_clip(self):
        print(f"\n🔧 Loading CLIP ViT-L/14")
        model, preprocess = clip.load("ViT-L/14", device=self.device)
        print("  ✅ CLIP loaded")
        return model, preprocess

    def _load_classifier(self, weights_path: str):
        print(f"\n🔧 Loading Effort classifier from {Path(weights_path).name}")

        checkpoint = torch.load(weights_path, map_location='cpu')

        # CLIP ViT-L/14 → 768 dimensions
        classifier = nn.Linear(768, 2).to(self.device)

        if isinstance(checkpoint, dict):
            # หา classifier weights
            if 'classifier' in checkpoint:
                classifier.load_state_dict(checkpoint['classifier'])
                print("  ✅ Loaded checkpoint['classifier']")
            elif 'head' in checkpoint:
                classifier.load_state_dict(checkpoint['head'])
                print("  ✅ Loaded checkpoint['head']")
            elif 'fc' in checkpoint:
                classifier.load_state_dict(checkpoint['fc'])
                print("  ✅ Loaded checkpoint['fc']")
            elif 'model' in checkpoint:
                # อาจเป็น full model
                state_dict = checkpoint['model']
                # หา classifier weights
                classifier_state = {k.replace('classifier.', ''): v
                                   for k, v in state_dict.items()
                                   if 'classifier' in k or 'fc' in k or 'head' in k}
                if classifier_state:
                    classifier.load_state_dict(classifier_state, strict=False)
                    print("  ✅ Loaded classifier from checkpoint['model']")
                else:
                    print("  ❌ WARNING: No classifier found in checkpoint!")
            else:
                print("  ⚠️  Unexpected checkpoint structure")
                print(f"     Keys: {list(checkpoint.keys())}")
        else:
            print("  ❌ Checkpoint is not a dict - cannot load classifier")

        return classifier

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor):
        # Resize สำหรับ CLIP (224x224)
        if image_tensor.shape[-2:] != (224, 224):
            image_tensor = torch.nn.functional.interpolate(
                image_tensor, size=(224, 224), mode='bilinear', align_corners=False
            )

        image_tensor = image_tensor.to(self.device)

        # CLIP encoding
        features = self.clip_model.encode_image(image_tensor)
        features = features.float()

        # Classifier
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)

        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

        return fake_prob, real_prob
```

---

### **Step 5: ทดสอบใหม่**

หลังแก้โค้ดแล้ว:

```python
# โหลดโมเดลใหม่
print("="*60)
print("🔄 Reloading models with fixed code...")
print("="*60)

xception = XceptionModel(XCEPTION_PATH, device)
f3net = F3NetModel(F3NET_PATH, device)
effort = EffortModel(EFFORT_PATH, device)

models = {
    'xception': xception,
    'f3net': f3net,
    'effort': effort
}

print("\n✅ Models reloaded!")

# ทดสอบอีกครั้ง
print("\n🧪 Testing with 1 REAL and 1 FAKE image...")

real_img = Image.open(test_data[0]['path']).convert('RGB')
fake_img = Image.open(test_data[6075]['path']).convert('RGB')

for img, label in [(real_img, "REAL"), (fake_img, "FAKE")]:
    img_tensor = transform(img).unsqueeze(0)
    print(f"\n{label} image:")
    for model_name, model in models.items():
        fake_prob, real_prob = model.predict(img_tensor)
        pred = 'FAKE' if fake_prob > 0.5 else 'REAL'
        correct = '✅' if pred == label else '❌'
        print(f"  {model_name:10s} → {pred:4s} (fake:{fake_prob:.3f}, real:{real_prob:.3f}) {correct}")
```

---

## 🎯 คาดหวัง:

หลังแก้ไข ควรเห็น:

```
REAL image:
  xception   → REAL (fake:0.05-0.30, real:0.70-0.95) ✅
  f3net      → REAL (fake:0.05-0.30, real:0.70-0.95) ✅
  effort     → REAL (fake:0.10-0.40, real:0.60-0.90) ✅

FAKE image:
  xception   → FAKE (fake:0.70-0.95, real:0.05-0.30) ✅
  f3net      → FAKE (fake:0.70-0.95, real:0.05-0.30) ✅
  effort     → FAKE (fake:0.60-0.90, real:0.10-0.40) ✅
```

**ไม่ใช่ 0.50:0.50 อีกต่อไป!**

---

## ⚠️ ถ้ายังไม่ได้:

แสดงผลจาก `inspect_checkpoint()` ให้ดู จะช่วยแก้ต่อ!
