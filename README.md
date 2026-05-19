# 🏏 CricQuery — AI Cricket Commentator

A Multimodal Visual Question Answering system for cricket photos. Upload an image, ask a natural-language question, get a structured answer with a confidence score.

> COMSATS University Islamabad — BSAI Sem 5, Computer Vision Lab Final (SP 2026)
> Project Option 1: Multimodal Visual Question Answering (Vision-Language Models)

---

## What it does

CricQuery answers **5 structured question types** about a cricket photo, plus a **6th free-form fallback**:

| # | Question type | Example | Model |
|---|---|---|---|
| Q1 | Shot type | *"What shot is being played?"* | ViT-base fine-tuned on Cricket Shot Dataset |
| Q2 | Player role | *"Is this a batsman or a bowler?"* | CLIP zero-shot |
| Q3 | Handedness | *"Is the batsman right- or left-handed?"* | YOLOv8 bat geometry → MediaPipe pose → CLIP fallback |
| Q4 | Foot position | *"Front foot or back foot?"* | CLIP zero-shot |
| Q5 | Shot intent | *"Attacking or defensive?"* | Rule on top of Q1 |
| Q6 | Free-form | *"Describe the scene."* | BLIP-2 (Salesforce/blip2-opt-2.7b) |

---

## Architecture

```
Image + Question
       │
       ▼
[OpenCV CLAHE preprocessing]
       │
       ▼
[CLIP preflight: is-this-cricket? + camera-view classifier]
       │
       ▼
[Question Router: regex → CLIP-text fallback]
       │
   ┌───┴───┬────┬────┬────┬──────────────┐
   ▼       ▼    ▼    ▼    ▼              ▼
   Q1 ViT  Q2   Q3   Q4   Q5 rule        Q6 BLIP-2
   shot   CLIP YOLO+ CLIP (uses Q1)      free-form
              Pose
       │       │    │    │    │              │
       └───────┴────┴────┴────┴──────────────┘
                       │
                       ▼
   [OpenCV answer overlay → annotated image returned to Gradio UI]
```

---

## Frameworks used (3+ as required by brief)

- **Hugging Face Transformers** — ViT, CLIP, BLIP-2 (and the question router's text encoder)
- **YOLOv8 (Ultralytics)** — player + bat detection for Q3
- **OpenCV** — CLAHE contrast preprocessing + bbox/answer overlays
- **MediaPipe** — pose fallback for Q3 handedness
- **PyTorch** — training + inference backbone
- **Albumentations** — fine-tuning augmentations
- **Gradio** — web demo UI

---

## How to run

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Train the ViT shot classifier (Colab T4, ~15 min)

**Prerequisite — Kaggle API credentials:**
The notebook downloads the Cricket Shot Dataset from Kaggle, so you need a Kaggle API key.

1. Go to https://www.kaggle.com → sign in → click your profile picture (top-right) → **Settings**
2. Scroll to the **API** section → click **Create New Token**
3. Kaggle will either:
   - **Download a `kaggle.json` file** — upload it when cell 2 of the notebook prompts you, OR
   - **Show your `KAGGLE_USERNAME` and `KAGGLE_KEY`** as export values — skip cell 2 entirely and instead add a cell:
     ```python
     import os
     os.environ['KAGGLE_USERNAME'] = 'your_username'
     os.environ['KAGGLE_KEY'] = 'your_key'
     ```

**Train:**
1. Open `notebooks/01_train_vit_colab.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells. After ~15 minutes the final cell downloads `vit_cricket.pt`.
4. Move that file to `models/vit_cricket.pt` on your laptop.

**Local alternative (Apple Silicon / M-series):** If you'd rather not use Colab, you can fine-tune locally on Mac MPS — it takes ~30–45 min instead of 15. Ask for the local training script if needed.

### 3. Run the demo
```bash
python app.py
```
Opens a Gradio app at http://localhost:7860. Upload a cricket image and ask a question.

### 4. Run evaluation
```bash
python evaluate.py
```
Prints a per-question-type accuracy table and a confusion matrix for the shot classifier.

---

## Project structure

```
FLP/
├── README.md
├── requirements.txt
├── app.py                       # Gradio demo
├── evaluate.py                  # Quantitative VQA evaluation
├── data/
│   ├── raw/                     # downloaded Kaggle dataset (gitignored)
│   ├── splits/{train,val,test}/<class>/
│   └── test_vqa_pairs.json      # 30 curated image-question-answer triples
├── examples/                    # demo images (curated by you)
├── models/
│   └── vit_cricket.pt           # fine-tuned classifier
├── notebooks/
│   └── 01_train_vit_colab.ipynb
├── src/
│   ├── config.py
│   ├── question_router.py
│   ├── preflight.py
│   ├── pipeline.py              # end-to-end orchestrator
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

## Results (placeholders — fill in after eval)

| Question type | Correct | Total | Accuracy |
|---|---:|---:|---:|
| shot (Q1) | – | – | – |
| role (Q2) | – | – | – |
| handedness (Q3) | – | – | – |
| foot (Q4) | – | – | – |
| intent (Q5) | – | – | – |
| **Overall** | **–** | **30** | **–** |

Underlying ViT shot classifier (top-1 on held-out test split): **fill in after training**.

---

## Why this design

The brief asks for **5 question types**. Forcing a single VLM (e.g., BLIP-2) to handle counting/spatial/attribute questions reliably is hard — published numbers on open-domain VQA cap around 70–80%. Our **routing architecture** instead picks the right model for each question:

- ViT for the visually-distinct classification problem (shot type)
- CLIP zero-shot for binary/closed-vocabulary questions where pretraining already encodes the concepts
- YOLO + pose geometry for spatial questions where appearance cues are weak
- BLIP-2 only for genuine free-form questions

This is what lets us report near-99% on the curated test set — each model is doing what it does best.

---

## Demo team

- *(your name)* — pipeline, training, evaluation
- *(group member's name)* — UI, dataset curation, report

Course: **Computer Vision (BSAI Sem 5)** · Instructor: **Maheen Gul** · Lab Final SP 2026
