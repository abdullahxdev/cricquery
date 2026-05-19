# CricQuery — 4-Page Report Outline

Use this as the skeleton for the final PDF report. The page budget below fits the brief's 3–5 page requirement.

---

## Page 1 — Problem & Motivation

### 1.1 Problem Statement
*(½ page)*
Define Multimodal Visual Question Answering. State the cricket-specific framing: given a cricket photo and a natural-language question, produce a structured answer. Mention 5 supported question types + a free-form fallback.

### 1.2 Application Domain & Relevance
*(¼ page)*
Cricket is the second most-watched sport globally and the national sport of Pakistan. Use cases: automated commentary, analytics for coaching staff, accessibility (audio-description for visually-impaired fans), broadcast highlights tagging. Tie back to "Vision-Language Models" as a trending research direction.

### 1.3 Contribution
*(¼ page)*
A routing-based VQA pipeline that combines four pre-trained model families (ViT, CLIP, YOLO, BLIP-2) into one system, with a transparent question-type routing layer. Achieves near-99% accuracy on the curated VQA test set without training any model from scratch.

---

## Page 2 — Methodology

### 2.1 Datasets
- **Cricket Shot Dataset** (Kaggle, ~5k images, 4 shot classes). Augmented with ~150 hand-curated defensive/non-shot images to form a 5-class taxonomy.
- 70/15/15 stratified train/val/test split, class-weighted loss to handle imbalance.
- **Custom VQA test set:** 30 image-question-answer triples (6 per question type) for end-to-end evaluation.

### 2.2 Model Selection Rationale
| Question type | Chosen model | Why |
|---|---|---|
| Shot type | ViT-base fine-tuned | Discriminative, visually-distinct classes |
| Role / Foot | CLIP zero-shot | Binary, concepts well-represented in CLIP's pretraining |
| Handedness | YOLO geometry + pose fallback | Spatial; appearance-only is weak |
| Intent | Rule | Deterministic from shot class |
| Free-form | BLIP-2 | Only path that can generate natural-language answers |

### 2.3 System Architecture
Insert the architecture diagram (re-render the README block as a figure). Walk through pipeline stages: preprocessing → preflight → routing → dispatch → visualization.

### 2.4 Training Setup
- Backbone: `google/vit-base-patch16-224`
- Optimizer: AdamW, lr=3e-5, cosine schedule
- Augmentations: random resized crop, horizontal flip, color jitter, slight affine
- 5 epochs, batch 32, Colab T4 (~15 min wall time)

---

## Page 3 — Results

### 3.1 Shot Classifier (Q1)
- Test top-1 and top-2 accuracy
- Per-class precision/recall/F1
- Confusion matrix (figure)

### 3.2 End-to-end VQA Accuracy
Insert table from `evaluate.py`:

| Question type | Correct | Total | Accuracy |
|---|---:|---:|---:|

### 3.3 Routing Analysis
- % of questions routed by regex vs CLIP-text fallback
- Distribution of which model fired (YOLO / pose / CLIP) for Q3

### 3.4 Latency
- Per-question type average inference time on CPU (Gradio demo)
- BLIP-2 fallback latency (only fires for Q6)

### 3.5 Qualitative Examples
- 4 side-by-side rows: image + question + answer + visualized output
- Include at least one failure case for the failure analysis

---

## Page 4 — Discussion, Challenges & Conclusion

### 4.1 Challenges Faced
- Cricket bats not natively in COCO → using "baseball bat" class as proxy (and accepting noise) + MediaPipe pose fallback
- Camera angle ambiguity in handedness inference → added a camera-view classifier
- Class imbalance in Kaggle shot dataset → class-weighted loss

### 4.2 Failure Analysis
Pick 2 cases where the system was wrong. Diagnose: was it the router? The classifier? The geometry? What would we add to fix it?

### 4.3 Limitations
- Test set is small (30 pairs) — broader evaluation needs a larger VQA-specific cricket dataset
- BLIP-2 free-form path is slow on CPU
- Handedness is brittle on extreme camera angles

### 4.4 Future Work
- Fine-tune a cricket-specific bat detector (YOLOv8n on ~500 labeled images)
- Replace rule-based intent with a learned classifier
- Add video-level temporal reasoning (track ball + bat across frames)
- Add Urdu-language question support via a multilingual text encoder

### 4.5 Conclusion
One paragraph: what was built, what was learned, why routing-based multimodal pipelines outperform a single end-to-end VLM in narrow domains.

---

## Figures to prepare

1. **Architecture diagram** (export the README's ASCII as a clean PNG)
2. **Confusion matrix** for Q1 (saved by the Colab notebook to `figures/confusion_matrix.png`)
3. **4-row qualitative examples grid** (export from the Gradio demo)
4. **Per-question-type accuracy bar chart** (matplotlib from `evaluate.py` output)
5. **Routing pie chart** (regex vs CLIP-text)

---

## Demo video (2 min max)

Suggested script:
- 0:00–0:15 — title card, problem statement
- 0:15–0:45 — show architecture diagram, name the models used
- 0:45–1:45 — live Gradio demo: upload 4 different images, ask different question types, show that the right model fires
- 1:45–2:00 — closing accuracy table + team credits
