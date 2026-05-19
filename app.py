"""Gradio web app — the demo interface for CricQuery.

Run:  python app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from PIL import Image

from src.pipeline import CricQueryPipeline

ENABLE_BLIP2 = os.environ.get("CRICQUERY_BLIP2", "1") == "1"
EXAMPLES_DIR = Path(__file__).parent / "examples"

PIPELINE = CricQueryPipeline(enable_blip2=ENABLE_BLIP2)

# Friendly labels for the 5 structured question types
QTYPE_LABEL = {
    "shot": "Shot type",
    "role": "Player role",
    "handedness": "Batsman handedness",
    "foot": "Foot position",
    "intent": "Shot intent",
    "freeform": "Free-form description",
    "preflight": "Image check",
}

# Buttons we show in the UI — the 5 supported types + one freeform demo
PRESET_QUESTIONS = [
    ("🏏 What shot is being played?", "What shot is being played?"),
    ("🎯 Is this a batsman or a bowler?", "Is this player a batsman or a bowler?"),
    ("✋ Is the batsman right-handed or left-handed?", "Is the batsman right-handed or left-handed?"),
    ("👟 Is this a front-foot or back-foot shot?", "Is this a front-foot or back-foot shot?"),
    ("⚔️ Is this an attacking or defensive shot?", "Is this an attacking or defensive shot?"),
    ("📝 Describe what is happening", "Describe what is happening in this image."),
]


def run(image: Image.Image, question: str):
    if image is None:
        return None, "⚠️ **Please upload a cricket image first.**", ""
    if not question or not question.strip():
        return None, "⚠️ **Please type a question or click one of the preset buttons.**", ""

    res = PIPELINE.answer(image, question)
    vis = PIPELINE.visualize(image, res)

    # Friendly headline
    qtype_pretty = QTYPE_LABEL.get(res.qtype, res.qtype.title())

    # Special-case freeform-disabled — give a helpful nudge
    if res.qtype == "freeform" and "disabled" in res.answer.lower():
        answer_md = (
            "### 💤 Free-form description is disabled\n\n"
            "Free-form answers use BLIP-2 (a 2.7B-parameter model) which is slow on CPU. "
            "To enable it, restart the app without the `CRICQUERY_BLIP2=0` flag.\n\n"
            "**Meanwhile, try one of the structured questions** "
            "(shot · role · handedness · foot · intent) — those answer in 1–2 seconds."
        )
    else:
        answer_md = (
            f"## **{res.answer}**\n\n"
            f"*Confidence: **{res.confidence:.1%}**  ·  Question type: **{qtype_pretty}***"
        )

    # Compact, plain-English technical breakdown (collapsed by default)
    details = (
        f"- **Question type:** `{res.qtype}` — the system classified your question into this category\n"
        f"- **Resolved by:** `{res.via}` — the model that produced the answer "
        f"(ViT = fine-tuned classifier · CLIP = zero-shot match · YOLO = bbox geometry · "
        f"rule = derived from the shot prediction · BLIP-2 = free-form generative)\n"
        f"- **Router:** `{res.extras.get('router_via', '-')}` "
        f"(confidence {res.extras.get('router_confidence', 0):.2f}) — how the question type was decided "
        f"(regex = keyword match · clip = text-similarity match)\n"
        f"- **Camera view:** `{res.extras.get('camera_view', '-')}` — image angle (used to flip handedness rule)"
    )
    return vis, answer_md, details


def list_examples():
    if not EXAMPLES_DIR.exists():
        return []
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(EXAMPLES_DIR.rglob(ext))
    return sorted(str(p) for p in files)


with gr.Blocks(title="CricQuery — AI Cricket Commentator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏏 CricQuery — AI Cricket Commentator

        Multimodal Visual Question Answering for cricket photos.
        **Upload an image → click a question button → get an answer with confidence.**
        """
    )

    with gr.Row():
        # LEFT: input
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="📷 Cricket image", height=380)
            q_in = gr.Textbox(
                label="Your question",
                placeholder="Type a question, or click one of the buttons below ↓",
                lines=1,
            )
            with gr.Row():
                go = gr.Button("Answer 🎯", variant="primary", scale=2)
                clr = gr.Button("Clear", scale=1)

            gr.Markdown("### Quick-pick questions")
            with gr.Column():
                for label, q in PRESET_QUESTIONS:
                    btn = gr.Button(label, size="sm")
                    btn.click(lambda x=q: x, None, q_in)

            ex = list_examples()
            if ex:
                gr.Examples(examples=[[e] for e in ex], inputs=[img_in], label="Click an example image")

        # RIGHT: output
        with gr.Column(scale=1):
            ans_md = gr.Markdown(
                "### 👋 Welcome\n\n"
                "Upload a cricket image on the left and click one of the **Quick-pick questions** to see the system answer."
            )
            img_out = gr.Image(label="🖼️ Annotated result", type="numpy", height=380)
            with gr.Accordion("🔍 Technical details (for viva / debugging)", open=False):
                dbg_md = gr.Markdown("Run a question to see how the answer was produced.")

    go.click(run, [img_in, q_in], [img_out, ans_md, dbg_md])
    clr.click(
        lambda: (None, "### 👋 Welcome\n\nUpload an image and ask a question.", "Run a question to see details."),
        None,
        [img_out, ans_md, dbg_md],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False, inbrowser=True)
