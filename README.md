# 🌿 PlantGuard AI — Plant Disease Detection System

<p align="center">
  <img src="https://img.shields.io/badge/F1--Score-97.7%25-brightgreen" />
  <img src="https://img.shields.io/badge/Core%20Classes-10-blue" />
  <img src="https://img.shields.io/badge/Inference-ONNX%20Runtime-orange" />
  <img src="https://img.shields.io/badge/UI-Gradio-purple" />
</p>

## Abstract

Crop diseases are one of the major threats to food security, and rapid identification remains challenging without specialized equipment. **PlantGuard AI** uses Convolutional Neural Networks (CNNs) to diagnose plant diseases from leaf photographs, achieving **97.7% F1-score** across 10 core classes spanning 3 crop species (Bell Pepper, Potato, and Tomato) and supporting up to 38 PlantVillage classes in the live web interface.

## Project Structure

```text
.
├── app.py                              # Gradio web UI & ONNX inference server
├── requirements.txt                    # Python dependencies
├── PlantGuardAI-CNN-256x256.ipynb      # Hi-Res (256x256) CNN training notebook
├── PlantGuardAI-CNN-64x64.ipynb        # Lo-Res (64x64) CNN training notebook
├── weights/                            # Pre-trained MobileNetV2 ONNX model & config
└── README.md
```

## Running the Project

```bash
pip install -r requirements.txt
python app.py
```

The web UI will start locally and open automatically at `http://127.0.0.1:7860`.

## Supported Core Plants & Diseases

| Plant | Condition | Samples |
|---|---|---|
| Bell Pepper | Healthy | 499 |
| Bell Pepper | Bacterial Spot | 499 |
| Potato | Healthy | 152 |
| Potato | Early Blight | 500 |
| Potato | Late Blight | 500 |
| Tomato | Healthy | 500 |
| Tomato | Bacterial Spot | 500 |
| Tomato | Target Spot | 500 |
| Tomato | Mosaic Virus | 373 |
| Tomato | Yellow Leaf Curl Virus | 499 |

## Custom CNN Architecture (Notebooks)

The training notebooks build a multi-block CNN with ReLU activations, batch normalization, max pooling, and dropout regularization:

| Layer | Activation |
|---|---|
| Conv2D (2×2, 64) | ReLU |
| Batch Normalization | — |
| Max Pooling (3×3) + Dropout (25%) | — |
| Conv2D (3×3, 64) × 2 + BatchNorm | ReLU |
| Max Pooling (2×2) + Dropout (25%) | — |
| Conv2D (3×3, 128) × 2 + BatchNorm | ReLU |
| Max Pooling (2×2 / 4×4) + Dropout (25%) | — |
| Flatten → Dense (1024) + BatchNorm + Dropout (20%) | ReLU |
| Dense (10) | Softmax |

## Benchmark Results

| Model | F1-Score |
|---|---|
| 256×256 (Hi-Res) | **97.7%** |
| 64×64 (Lo-Res) | 97.2% |

## Dataset

Images sourced from the [PlantVillage](https://plantvillage.psu.edu/) dataset — a public collection of diseased and healthy plant leaf images.
