# 🔍 Debug Model Loading - ตรวจสอบ Classifier

Accuracy ยังคงอยู่ที่ ~50% แม้จะแก้แล้ว!

## 🧪 เพิ่ม Debug Code ใน Colab:

Run Cell ใหม่นี้เพื่อตรวจสอบว่า classifier โหลดจริงหรือไม่:

```python
# ===================================
# DEBUG: ตรวจสอบ Classifier Weights
# ===================================

print("\n" + "="*60)
print("🔍 DEBUGGING CLASSIFIER WEIGHTS")
print("="*60)

# 1. ตรวจสอบ Xception
print("\n1️⃣ XCEPTION:")
print(f"   Model type: {type(xception.model)}")
print(f"   Has last_linear: {hasattr(xception.model, 'last_linear')}")
if hasattr(xception.model, 'last_linear'):
    print(f"   last_linear type: {type(xception.model.last_linear)}")
    print(f"   last_linear weight shape: {xception.model.last_linear.weight.shape}")
    print(f"   last_linear weight (first 5 values): {xception.model.last_linear.weight[0, :5]}")

    # ตรวจสอบว่า weights เป็น random หรือไม่
    weight_std = xception.model.last_linear.weight.std().item()
    weight_mean = xception.model.last_linear.weight.mean().item()
    print(f"   Weight mean: {weight_mean:.6f}")
    print(f"   Weight std:  {weight_std:.6f}")

    if abs(weight_mean) < 0.01 and weight_std < 0.05:
        print("   ⚠️  WARNING: Looks like RANDOM initialization!")
    else:
        print("   ✅ Looks like loaded weights")

# 2. ตรวจสอบ F3Net
print("\n2️⃣ F3NET:")
print(f"   Model type: {type(f3net.model)}")
print(f"   Has last_linear: {hasattr(f3net.model, 'last_linear')}")
if hasattr(f3net.model, 'last_linear'):
    print(f"   last_linear weight shape: {f3net.model.last_linear.weight.shape}")
    print(f"   last_linear weight (first 5 values): {f3net.model.last_linear.weight[0, :5]}")

    weight_std = f3net.model.last_linear.weight.std().item()
    weight_mean = f3net.model.last_linear.weight.mean().item()
    print(f"   Weight mean: {weight_mean:.6f}")
    print(f"   Weight std:  {weight_std:.6f}")

    if abs(weight_mean) < 0.01 and weight_std < 0.05:
        print("   ⚠️  WARNING: Looks like RANDOM initialization!")
    else:
        print("   ✅ Looks like loaded weights")

# 3. ตรวจสอบ Effort
print("\n3️⃣ EFFORT:")
print(f"   Classifier type: {type(effort.classifier)}")
print(f"   Classifier weight shape: {effort.classifier.weight.shape}")
print(f"   Classifier weight (first 5 values): {effort.classifier.weight[0, :5]}")

weight_std = effort.classifier.weight.std().item()
weight_mean = effort.classifier.weight.mean().item()
print(f"   Weight mean: {weight_mean:.6f}")
print(f"   Weight std:  {weight_std:.6f}")

if abs(weight_mean) < 0.01 and weight_std < 0.05:
    print("   ⚠️  WARNING: Looks like RANDOM initialization!")
else:
    print("   ✅ Looks like loaded weights")

print("\n" + "="*60)

# 4. ตรวจสอบ checkpoint อีกครั้ง
print("\n4️⃣ CHECKPOINT VERIFICATION:")
print("\nXception checkpoint keys (last_linear):")
xception_ckpt = torch.load(XCEPTION_PATH, map_location='cpu')
last_linear_keys = [k for k in xception_ckpt.keys() if 'last_linear' in k]
print(f"   Found {len(last_linear_keys)} keys:")
for k in last_linear_keys:
    print(f"   - {k}: {xception_ckpt[k].shape}")

print("\nF3Net checkpoint keys (last_linear):")
f3net_ckpt = torch.load(F3NET_PATH, map_location='cpu')
last_linear_keys = [k for k in f3net_ckpt.keys() if 'last_linear' in k]
print(f"   Found {len(last_linear_keys)} keys:")
for k in last_linear_keys:
    print(f"   - {k}: {f3net_ckpt[k].shape}")

print("\nEffort checkpoint keys (head):")
effort_ckpt = torch.load(EFFORT_PATH, map_location='cpu')
head_keys = [k for k in effort_ckpt.keys() if 'head' in k]
print(f"   Found {len(head_keys)} keys:")
for k in head_keys:
    print(f"   - {k}: {effort_ckpt[k].shape}")

print("\n" + "="*60)
```

## 📊 ผลที่ควรเห็น:

### **ถ้า weights โหลดถูกต้อง:**
```
Weight mean: -0.034512 (ไม่ใกล้ 0)
Weight std:  0.234567 (> 0.1)
✅ Looks like loaded weights
```

### **ถ้า weights ยังเป็น random:**
```
Weight mean: 0.000123 (ใกล้ 0)
Weight std:  0.024567 (< 0.05)
⚠️  WARNING: Looks like RANDOM initialization!
```

---

## 🔑 สาเหตุที่เป็นไปได้:

1. **Checkpoint ไม่มี classifier weights จริงๆ**
2. **Key names ยังไม่ match** (ต้อง debug ต่อ)
3. **Timm Xception มี layer name แตกต่างจากที่คิด**
4. **Weights โหลดแล้ว แต่ถูก overwrite ด้วย random init**

---

Run debug code ด้านบนแล้วส่งผลมาให้ผมดูครับ!
