import sys
sys.path.append('app')
import torch
from models.manager import EnsembleModelManager

print('=== Testing Model Predictions ===\n')

manager = EnsembleModelManager('app/config.json')

print(f'Device: {manager.device}')
print(f'Models loaded: {list(manager.models.keys())}\n')

# Test with random image
print('Testing with random tensor (224x224x3)...\n')
dummy_img = torch.randn(1, 3, 224, 224)
result = manager.predict_ensemble(dummy_img)

print('=== RESULTS ===')
print(f"Ensemble: {result['ensemble']['prediction']}")
print(f"  Fake prob: {result['ensemble']['fake_prob']:.4f} ({result['ensemble']['fake_prob']*100:.1f}%)")
print(f"  Real prob: {result['ensemble']['real_prob']:.4f} ({result['ensemble']['real_prob']*100:.1f}%)")
print(f"  Confidence: {result['ensemble']['confidence']:.4f}\n")

print('Individual models:')
for model_name, model_result in result['individual'].items():
    fake_pct = model_result['fake_prob'] * 100
    print(f"  {model_name:15} {fake_pct:5.1f}% {model_result['prediction']}")

    # Check if still 1%
    if fake_pct < 2.0 or fake_pct > 98.0:
        print(f"    [WARNING] May still have 1% bug!")
    else:
        print(f"    [OK] Working correctly")

print('\n=== SUMMARY ===')
probs = [r['fake_prob'] for r in result['individual'].values()]
if all(p < 0.02 or p > 0.98 for p in probs):
    print('[FAILED] All models returning 1% or 99% (bug still exists)')
elif any(p < 0.02 or p > 0.98 for p in probs):
    print('[PARTIAL] Some models still have 1% bug')
else:
    print('[SUCCESS] All models returning realistic predictions!')
    print(f'   Probability range: {min(probs)*100:.1f}% - {max(probs)*100:.1f}%')
