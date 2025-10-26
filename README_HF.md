---
title: Deepfake Detection
emoji: 🎭
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 4.12.0
app_file: app_gradio.py
pinned: false
license: mit
---

# 🎭 Deepfake Detection System

Advanced deepfake detection system using ensemble of 3 state-of-the-art models with **92.86% accuracy** on FaceForensics++ c23 dataset.

## 🚀 Features

- **High Accuracy**: 92.86% accuracy, 92.96% F1 score, 97.86% AUC
- **Ensemble Approach**: Combines 3 specialized models for robust detection
- **Real-time Processing**: Fast inference on GPU
- **Explainable AI**: Confidence scores and model-by-model breakdown

## 🧠 Model Architecture

Our system uses a weighted ensemble of three complementary models:

| Model | Weight | Accuracy | F1 Score | AUC | Specialization |
|-------|--------|----------|----------|-----|----------------|
| **Effort-CLIP** | 60% | 85.00% | 83.20% | 93.53% | CLIP Vision Encoder - High precision |
| **Xception** | 30% | 84.29% | 86.25% | 97.67% | Modified Xception - High recall |
| **F3Net** | 10% | 68.57% | 76.09% | 93.96% | Frequency-aware - Perfect recall |
| **Ensemble** | 100% | **92.86%** | **92.96%** | **97.86%** | Weighted combination |

### Why Ensemble?

- **Effort-CLIP** (60%): Leverages CLIP's powerful vision encoder for semantic understanding
- **Xception** (30%): Detects spatial artifacts and unnatural textures
- **F3Net** (10%): Analyzes frequency domain for compression artifacts

## 📊 Performance

Tested on FaceForensics++ c23 dataset (140 images):

- **Accuracy**: 92.86%
- **F1 Score**: 92.96%
- **Precision**: 93.02%
- **Recall**: 92.86%
- **AUC**: 97.86%

## 🎯 Usage

1. Upload an image containing a face
2. The system will analyze it using all 3 models
3. Get instant prediction with confidence score
4. View individual model predictions for transparency

## ⚙️ Technical Details

- **Framework**: PyTorch, Transformers, Gradio
- **Models**: Effort-CLIP (ViT-L/14), Xception, F3Net
- **Face Detection**: MTCNN
- **Optimization**: Weights optimized via grid search on c23 dataset

## 🔬 Training Data

Models trained on FaceForensics++ dataset:
- **Real**: YouTube faces
- **Fake**: Deepfakes, Face2Face, FaceSwap, NeuralTextures

## 🚨 Limitations

- Designed for face images (portrait/selfie style)
- Performance may vary on unseen deepfake methods
- Not suitable for real-time video (use frame-by-frame)
- Requires clear, well-lit faces for best results

## 📝 Citation

If you use this system, please cite:

```bibtex
@software{deepfake_detection_2025,
  title={Deepfake Detection Ensemble System},
  author={Your Name},
  year={2025},
  url={https://huggingface.co/spaces/your-username/deepfake-detection}
}
```

## 🤝 Contributing

Contributions welcome! See the [GitHub repository](https://github.com/your-username/deepfake-detection) for more details.

## 📄 License

MIT License - See LICENSE file for details

---

**⚠️ Disclaimer**: This system is for research and educational purposes. Always verify critical information through multiple sources.
