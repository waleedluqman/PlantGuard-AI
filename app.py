"""
PlantGuard AI - Plant Disease Detection System
Deep learning-based plant leaf disease classification using Convolutional Neural Networks.
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

# ─────────────────────────────────────────────
# Paths & Model Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
MODEL_PATH = WEIGHTS_DIR / "onnx" / "model.onnx"
CONFIG_PATH = WEIGHTS_DIR / "config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    MODEL_CONFIG = json.load(f)

ID2LABEL = {int(k): v for k, v in MODEL_CONFIG["id2label"].items()}

# Core 10 classes corresponding to the Bell Pepper, Potato, and Tomato study
CORE_10_IDS = [18, 19, 20, 21, 22, 28, 34, 35, 36, 37]

DISPLAY_NAMES = {
    18: "Bell Pepper – Bacterial Spot",
    19: "Bell Pepper – Healthy",
    20: "Potato – Early Blight",
    21: "Potato – Late Blight",
    22: "Potato – Healthy",
    28: "Tomato – Bacterial Spot",
    34: "Tomato – Target Spot",
    35: "Tomato – Yellow Leaf Curl Virus",
    36: "Tomato – Mosaic Virus",
    37: "Tomato – Healthy",
}

DISEASE_INFO = {
    "Bell Pepper – Bacterial Spot": {
        "severity": "Moderate",
        "cause": "Xanthomonas campestris pv. vesicatoria (Bacteria)",
        "treatment": "Apply copper-based bactericides early. Remove and destroy severely infected leaves and avoid working with wet plants.",
        "prevention": "Use certified disease-free seeds and transplants, practice 2–3 year crop rotation, and use drip irrigation instead of overhead watering.",
    },
    "Bell Pepper – Healthy": {
        "severity": "None",
        "cause": "No pathogen detected",
        "treatment": "No treatment required. Continue regular monitoring and balanced fertilization.",
        "prevention": "Maintain proper plant spacing, airflow, and consistent soil moisture.",
    },
    "Potato – Early Blight": {
        "severity": "Moderate",
        "cause": "Alternaria solani (Fungus)",
        "treatment": "Apply protectant fungicides (chlorothalonil, mancozeb, or copper) at first sign of concentric ring lesions. Prune lower infected foliage.",
        "prevention": "Rotate crops every 2–3 years, clear crop debris after harvest, and keep foliage dry by watering at the soil line.",
    },
    "Potato – Late Blight": {
        "severity": "Severe",
        "cause": "Phytophthora infestans (Oomycete)",
        "treatment": "Apply targeted anti-oomycete fungicides immediately. Remove and bag infected plants to prevent rapid spore spread to neighboring crops.",
        "prevention": "Plant certified blight-free seed tubers, choose resistant cultivars, and monitor closely during cool, humid weather.",
    },
    "Potato – Healthy": {
        "severity": "None",
        "cause": "No pathogen detected",
        "treatment": "No treatment required. Keep inspecting lower leaves during humid periods.",
        "prevention": "Maintain well-drained soil, adequate potassium/nitrogen balance, and clean field sanitation.",
    },
    "Tomato – Bacterial Spot": {
        "severity": "Moderate",
        "cause": "Xanthomonas perforans / X. euvesicatoria (Bacteria)",
        "treatment": "Spray copper hydroxide or copper octanoate combined with biological controls. Remove affected lower leaves.",
        "prevention": "Start with pathogen-free seeds, sanitize stakes and tools, and avoid overhead irrigation.",
    },
    "Tomato – Target Spot": {
        "severity": "Moderate",
        "cause": "Corynespora cassiicola (Fungus)",
        "treatment": "Apply recommended fungicides (e.g., azoxystrobin or chlorothalonil) and improve canopy ventilation by pruning suckers.",
        "prevention": "Space plants generously, mulch around the base to reduce soil splash, and rotate out of solanaceous crops.",
    },
    "Tomato – Yellow Leaf Curl Virus": {
        "severity": "Severe",
        "cause": "Tomato Yellow Leaf Curl Virus (TYLCV), transmitted by whiteflies",
        "treatment": "There is no chemical cure for infected plants—rogue and remove infected plants early to protect healthy crops, and control whitefly vectors.",
        "prevention": "Grow TYLCV-resistant varieties, install fine insect netting or reflective mulch, and use yellow sticky traps for whiteflies.",
    },
    "Tomato – Mosaic Virus": {
        "severity": "Moderate",
        "cause": "Tomato Mosaic Virus / Tobacco Mosaic Virus (ToMV / TMV)",
        "treatment": "Remove infected plants promptly and avoid touching healthy plants afterward. Disinfect hands, shears, and trellises with 10% milk or trisodium phosphate solution.",
        "prevention": "Plant virus-resistant cultivars, wash hands thoroughly before handling seedlings, and control weeds near greenhouses.",
    },
    "Tomato – Healthy": {
        "severity": "None",
        "cause": "No pathogen detected",
        "treatment": "No treatment required. Plant foliage looks vigorous and free of lesions.",
        "prevention": "Continue balanced watering, mulching, and regular inspection for early signs of pests or spots.",
    },
}


def get_disease_details(label_id: int, label_name: str) -> dict:
    """Return structured diagnosis metadata for both core 10 and extended 38 classes."""
    if label_name in DISEASE_INFO:
        return DISEASE_INFO[label_name]

    is_healthy = "healthy" in label_name.lower()
    if is_healthy:
        return {
            "severity": "None",
            "cause": "No pathogen detected",
            "treatment": "No treatment required. Maintain routine crop care and monitoring.",
            "prevention": "Ensure good airflow, balanced nutrition, and clean watering practices.",
        }

    severity = "Severe" if any(k in label_name.lower() for k in ["blight", "virus", "greening", "rot"]) else "Moderate"
    return {
        "severity": severity,
        "cause": f"Fungal, bacterial, or viral stress associated with {label_name}",
        "treatment": "Isolate severely affected leaves, improve canopy airflow, and apply an appropriate targeted fungicide or bactericide.",
        "prevention": "Use disease-free planting material, practice crop rotation, and avoid prolonged leaf wetness.",
    }


# ─────────────────────────────────────────────
# Load ONNX Inference Session
# ─────────────────────────────────────────────
print("Loading PlantGuard AI inference model...")
SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name
print("PlantGuard AI model ready.")


# ─────────────────────────────────────────────
# Preprocessing & Prediction
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize shortest edge to 256, center-crop 224x224, rescale, and normalize."""
    img = image.convert("RGB")
    w, h = img.size
    if w < h:
        new_w = 256
        new_h = int(round(256 * h / w))
    else:
        new_h = 256
        new_w = int(round(256 * w / h))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(arr, axis=0)  # NCHW


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


def predict(image: Image.Image, class_scope: str):
    """Run inference on a leaf image and return formatted diagnosis + probabilities."""
    if image is None:
        return (
            "_Upload or drop a leaf image on the left and click **Analyze Leaf** to view the diagnosis._",
            {},
        )

    start_time = time.perf_counter()
    input_tensor = preprocess_image(image)
    logits = SESSION.run(None, {INPUT_NAME: input_tensor})[0][0]
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    if class_scope.startswith("Core 10"):
        sub_logits = np.array([logits[i] for i in CORE_10_IDS], dtype=np.float32)
        probs = softmax(sub_logits)
        labels = [DISPLAY_NAMES[i] for i in CORE_10_IDS]
        ids = CORE_10_IDS
    else:
        probs = softmax(logits)
        ids = list(range(len(ID2LABEL)))
        labels = [DISPLAY_NAMES.get(i, ID2LABEL[i]) for i in ids]

    top_pos = int(np.argmax(probs))
    top_id = ids[top_pos]
    top_label = labels[top_pos]
    top_confidence = float(probs[top_pos]) * 100.0

    info = get_disease_details(top_id, top_label)
    severity = info["severity"]
    severity_icon = {"None": "✅", "Moderate": "⚠️", "Severe": "🚨"}.get(severity, "ℹ️")
    health_status = "Healthy" if severity == "None" else "Disease Detected"

    result_md = f"""## {severity_icon} {top_label}

| Metric | Value |
|---|---|
| **Status** | {health_status} |
| **Confidence** | **{top_confidence:.2f}%** |
| **Severity** | {severity} |
| **Inference Time** | {elapsed_ms:.1f} ms |

**Likely Cause:** {info['cause']}

---

### 💊 Recommended Treatment
{info['treatment']}

### 🛡️ Prevention Tips
{info['prevention']}
"""

    confidence_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return result_md, confidence_dict


# ─────────────────────────────────────────────
# UI Styling & Layout
# ─────────────────────────────────────────────
CSS = """
.gradio-container {
    max-width: 1120px !important;
    margin: auto !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.header-banner {
    background: linear-gradient(135deg, #14532d 0%, #16a34a 55%, #15803d 100%);
    padding: 1.75rem 2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    text-align: center;
    color: #ffffff;
    box-shadow: 0 8px 24px rgba(20, 83, 45, 0.18);
}
.header-banner h1 {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
    color: #ffffff;
}
.header-banner p {
    opacity: 0.94;
    font-size: 1rem;
    margin: 0.45rem 0 0 0;
    color: #ecfdf5;
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.25rem;
}
.stat-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
}
.stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #15803d;
}
.stat-label {
    font-size: 0.85rem;
    color: #374151;
    margin-top: 2px;
}
.footer-note {
    text-align: center;
    color: #6b7280;
    font-size: 0.85rem;
    margin-top: 1rem;
}
"""

HEADER_HTML = """
<div class="header-banner">
    <h1>🌿 PlantGuard AI</h1>
    <p>Automated Plant Leaf Disease Classification using Deep Convolutional Neural Networks</p>
</div>
"""

STATS_HTML = """
<div class="stats-grid">
    <div class="stat-box">
        <div class="stat-value">3 Core / 14 Total</div>
        <div class="stat-label">Crop Species</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">10 Core / 38 Total</div>
        <div class="stat-label">Disease Classes</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">97.7%</div>
        <div class="stat-label">Benchmark F1-Score</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">&lt; 100 ms</div>
        <div class="stat-label">CPU Inference Speed</div>
    </div>
</div>
"""

ARCHITECTURE_MD = """
## Model & Training Overview

**PlantGuard AI** classifies plant leaf images into healthy and diseased categories to support rapid, accessible crop disease screening.

### 1. Custom CNN Study (Notebooks)
The accompanying Jupyter notebooks (`PlantGuardAI-CNN-256x256.ipynb` and `PlantGuardAI-CNN-64x64.ipynb`) evaluate a custom Convolutional Neural Network trained on **4,522 PlantVillage images** across 10 core classes (75% train / 25% test split) with real-time data augmentation (rotation up to 30°, 15% height shift, 15% shear, 20% zoom, horizontal flip).

| Model Variant | Input Resolution | Final MaxPool | Optimizer & LR | Best F1-Score |
|---|---|---|---|---|
| **Hi-Res CNN** | 256 × 256 × 3 | 2 × 2 | Adam (`6e-4`) | **97.7%** |
| **Lo-Res CNN** | 64 × 64 × 3 | 4 × 4 | Adam (`6e-4`) | **97.2%** |

### 2. Live UI Inference Engine
For fast, reliable real-time demonstration in this web interface, **PlantGuard AI** runs a pre-trained **MobileNetV2** deep convolutional model exported to **ONNX Runtime** (`weights/onnx/model.onnx`), supporting both the **10 core project classes** (Bell Pepper, Potato, Tomato) and the **extended 38 PlantVillage classes**.
"""

CLASSES_MD = """
## Supported Crops & Conditions

### Core Project Classes (10 Classes — 4,522 Training Samples)
| Crop | Condition | Samples | Category |
|---|---|---|---|
| 🫑 Bell Pepper | Healthy | 499 | Healthy |
| 🫑 Bell Pepper | Bacterial Spot | 499 | Bacterial |
| 🥔 Potato | Healthy | 152 | Healthy |
| 🥔 Potato | Early Blight | 500 | Fungal |
| 🥔 Potato | Late Blight | 500 | Oomycete |
| 🍅 Tomato | Healthy | 500 | Healthy |
| 🍅 Tomato | Bacterial Spot | 500 | Bacterial |
| 🍅 Tomato | Target Spot | 500 | Fungal |
| 🍅 Tomato | Mosaic Virus | 373 | Viral |
| 🍅 Tomato | Yellow Leaf Curl Virus | 499 | Viral |

_Switch the **Classification Scope** selector on the Diagnose tab to **Extended 38 PlantVillage Classes** to also classify Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Raspberry, Soybean, Squash, and Strawberry leaves._
"""

with gr.Blocks(title="PlantGuard AI – Plant Disease Detection", analytics_enabled=False) as demo:
    gr.HTML(HEADER_HTML)
    gr.HTML(STATS_HTML)

    with gr.Tabs():
        with gr.Tab("🔬 Diagnose Leaf"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Leaf Image",
                        image_mode="RGB",
                        height=340,
                    )
                    scope_selector = gr.Radio(
                        choices=[
                            "Core 10 Classes (Bell Pepper, Potato, Tomato)",
                            "Extended 38 PlantVillage Classes",
                        ],
                        value="Core 10 Classes (Bell Pepper, Potato, Tomato)",
                        label="Classification Scope",
                    )
                    with gr.Row():
                        clear_btn = gr.ClearButton(
                            components=[image_input],
                            value="🗑️ Clear",
                        )
                        submit_btn = gr.Button(
                            "🔍 Analyze Leaf",
                            variant="primary",
                        )

                with gr.Column(scale=1):
                    diagnosis_output = gr.Markdown(
                        value="_Upload a leaf image on the left and click **Analyze Leaf** to get an instant diagnosis._",
                    )

            gr.Markdown("### 📊 Class Confidence Breakdown")
            confidence_output = gr.Label(
                num_top_classes=10,
                label="Top Predicted Classes",
            )

        with gr.Tab("🧠 Model & Architecture"):
            gr.Markdown(ARCHITECTURE_MD)

        with gr.Tab("🌱 Supported Diseases"):
            gr.Markdown(CLASSES_MD)

    submit_btn.click(
        fn=predict,
        inputs=[image_input, scope_selector],
        outputs=[diagnosis_output, confidence_output],
    )
    image_input.change(
        fn=predict,
        inputs=[image_input, scope_selector],
        outputs=[diagnosis_output, confidence_output],
    )
    scope_selector.change(
        fn=predict,
        inputs=[image_input, scope_selector],
        outputs=[diagnosis_output, confidence_output],
    )

    gr.HTML('<div class="footer-note">PlantGuard AI — Deep Learning Plant Disease Detection System</div>')


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"),
    )
