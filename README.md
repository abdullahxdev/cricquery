# CricQuery, AI Cricket Commentator

A multimodal Visual Question Answering system for cricket images. Upload a cricket photo, ask a natural-language question, and receive a structured answer with a confidence score.

---

# Overview

CricQuery answers five structured cricket-analysis question types, along with a free-form fallback for general scene understanding.

| # | Question Type | Example Question | Model Used |
|---|---|---|---|
| Q1 | Shot Type | "What shot is being played?" | Fine-tuned ViT-base |
| Q2 | Player Role | "Is this a batsman or a bowler?" | CLIP Zero-Shot |
| Q3 | Handedness | "Is the batsman right- or left-handed?" | YOLOv8 + MediaPipe + CLIP fallback |
| Q4 | Foot Position | "Front foot or back foot?" | CLIP Zero-Shot |
| Q5 | Shot Intent | "Attacking or defensive?" | Rule-based inference on Q1 |
| Q6 | Free-form Reasoning | "Describe the scene." | BLIP-2 (Salesforce/blip2-opt-2.7b) |

---

# System Architecture

```text
Image + Question
       │
       ▼
[OpenCV CLAHE Preprocessing]
       │
       ▼
[CLIP Preflight Validation]
(cricket-image + camera-view verification)
       │
       ▼
[Question Router]
(regex + CLIP text similarity)
       │
   ┌───┴────┬────┬────┬────┬──────────────┐
   ▼        ▼    ▼    ▼    ▼              ▼
 Q1 ViT    Q2   Q3   Q4   Q5 Rule        Q6 BLIP-2
 Shot      CLIP YOLO  CLIP Uses Q1       Free-form
 Classifier      +Pose
       │
       └──────────────┬──────────────────┘
                      ▼
        [Annotated Output Renderer]
        OpenCV overlays + Gradio interface
```

---

# Technology Stack

- **Hugging Face Transformers** — ViT, CLIP, BLIP-2
- **YOLOv8 (Ultralytics)** — player and bat detection
- **OpenCV** — preprocessing and visualization
- **MediaPipe** — pose estimation
- **PyTorch** — training and inference
- **Albumentations** — image augmentation
- **Gradio** — interactive web UI

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Training the ViT Shot Classifier

## Kaggle API Setup

Generate a Kaggle API token from your Kaggle account settings.

Either upload `kaggle.json` in Colab or configure:

```python
import os

os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_key'
```

---

## Training Steps

1. Open:

```text
notebooks/01_train_vit_colab.ipynb
```

2. Enable GPU runtime in Colab

3. Run all cells

After training finishes, move:

```text
vit_cricket.pt
```

to:

```text
models/vit_cricket.pt
```

---

# Run the Demo

```bash
python app.py
```

Launches the Gradio interface at:

```text
http://localhost:7860
```

---

# Evaluation

```bash
python evaluate.py
```

Outputs:

- Per-question accuracy
- Overall VQA accuracy
- Confusion matrix for shot classification

---

# Project Structure

```text
FLP/
├── README.md
├── requirements.txt
├── app.py
├── evaluate.py
├── data/
│   ├── raw/
│   ├── splits/{train,val,test}/<class>/
│   └── test_vqa_pairs.json
├── examples/
├── models/
│   └── vit_cricket.pt
├── notebooks/
│   └── 01_train_vit_colab.ipynb
├── src/
│   ├── config.py
│   ├── question_router.py
│   ├── preflight.py
│   ├── pipeline.py
│   ├── visualize.py
│   └── inference/
│       ├── shot_classifier.py
│       ├── clip_qa.py
│       ├── yolo_handedness.py
│       ├── intent_rule.py
│       └── blip2_fallback.py
└── report/
    ├── REPORT_OUTLINE.md
    └── figures/
```

---

# Results

| Question Type | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Shot Type (Q1) | – | – | – |
| Player Role (Q2) | – | – | – |
| Handedness (Q3) | – | – | – |
| Foot Position (Q4) | – | – | – |
| Shot Intent (Q5) | – | – | – |
| **Overall** | **–** | **30** | **–** |

---

# Design Philosophy

CricQuery uses a routed multimodal architecture instead of relying on a single Vision-Language Model.

Different models handle different tasks:

- **ViT** → shot classification
- **CLIP** → zero-shot reasoning
- **YOLO + Pose Geometry** → spatial understanding
- **BLIP-2** → free-form scene description

This modular pipeline improves both accuracy and interpretability.

---

# Contributors

- [Malik Saad Hayat](http://github.com/saadhtiwana)
- [Muhammad Abdullah](http://github.com/abdullahxdev)
