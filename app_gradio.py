"""
Deepfake Detection - Hugging Face Spaces (Gradio)
Optimized ensemble model: 92.86% accuracy
"""

import gradio as gr
import torch
from PIL import Image
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.services.detection import EnsembleDetectionService

# Initialize service
print("Loading models...")
service = EnsembleDetectionService(config_path="backend/app/config.json", enable_mlflow=False)
print("Models loaded successfully!")

def predict(image):
    """
    Predict if image is real or fake

    Args:
        image: PIL Image

    Returns:
        Prediction, confidence, details
    """
    if image is None:
        return "Please upload an image", 0.0, {}

    # Process
    result = service.process(image, generate_heatmap=False)

    # Format output
    prediction = result['prediction']
    confidence = result['confidence']

    # Model details
    details = {
        f"{name} ({cfg['weight']*100:.0f}%)": f"{pred['fake_prob']*100:.1f}% FAKE"
        for name, pred in result['model_predictions'].items()
        for cfg in [service.model_manager.config['models'].get(name, {})]
    }

    # Confidence label
    if prediction == "FAKE":
        label = f"🚨 FAKE ({confidence*100:.1f}% confidence)"
        color = "red"
    else:
        label = f"✅ REAL ({confidence*100:.1f}% confidence)"
        color = "green"

    return label, confidence, details

# Create Gradio interface
with gr.Blocks(title="Deepfake Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 Deepfake Detection System

    **92.86% Accuracy** - Ensemble of 3 state-of-the-art models

    Upload an image to detect if it's real or manipulated using deepfake technology.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Image",
                type="pil",
                height=400
            )

            submit_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

        with gr.Column():
            prediction_output = gr.Textbox(
                label="Prediction",
                interactive=False
            )

            confidence_output = gr.Slider(
                label="Confidence Score",
                minimum=0,
                maximum=1,
                interactive=False
            )

            details_output = gr.JSON(
                label="Individual Model Predictions"
            )

    gr.Markdown("""
    ## 🧠 Model Architecture

    - **Effort-CLIP** (60%): CLIP Vision Encoder - High precision (94.55%)
    - **Xception** (30%): Modified Xception - High recall (98.57%)
    - **F3Net** (10%): Frequency-aware network - Perfect recall (100%)

    **Ensemble Method:** Weighted average optimized on FaceForensics++ c23

    ## 📊 Performance

    - Accuracy: **92.86%**
    - F1 Score: **92.96%**
    - AUC: **97.86%**

    ---

    Made with ❤️ using Gradio, PyTorch, and transformers
    """)

    # Connect events
    submit_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[prediction_output, confidence_output, details_output]
    )

    # Auto-run on upload
    image_input.change(
        fn=predict,
        inputs=[image_input],
        outputs=[prediction_output, confidence_output, details_output]
    )

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
