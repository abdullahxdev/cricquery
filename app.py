"""Gradio web app — the demo interface for CricQuery.

Run:  python app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from PIL import Image

from src.pipeline import CricQueryPipeline

# BLIP-2 is OFF by default — it's a 2.7B-parameter model that takes 30s–2min per
# question on CPU/MPS and locks the Gradio queue. The 5 structured question types
# (shot/role/handedness/foot/intent) handle everything we need for the demo.
# To turn it on for testing free-form: CRICQUERY_BLIP2=1 python app.py
ENABLE_BLIP2 = os.environ.get("CRICQUERY_BLIP2", "0") == "1"
EXAMPLES_DIR = Path(__file__).parent / "examples"

PIPELINE = CricQueryPipeline(enable_blip2=ENABLE_BLIP2)

# Friendly labels for the supported question types
QTYPE_LABEL = {
    "shot": "Shot type",
    "role": "Player role",
    "handedness": "Batsman handedness",
    "foot": "Foot position",
    "intent": "Shot intent",
    "freeform": "Free-form description",
    "preflight": "Image check",
}

# Only show the 5 supported question types. Free-form is intentionally NOT in the
# preset list so users don't accidentally trigger BLIP-2.
PRESET_QUESTIONS = [
    ("🏏 What shot is being played?", "What shot is being played?"),
    ("🎯 Is this a batsman or a bowler?", "Is this player a batsman or a bowler?"),
    ("✋ Is the batsman right-handed or left-handed?", "Is the batsman right-handed or left-handed?"),
    ("👟 Is this a front-foot or back-foot shot?", "Is this a front-foot or back-foot shot?"),
    ("⚔️ Is this an attacking or defensive shot?", "Is this an attacking or defensive shot?"),
]


def run(image: Image.Image, question: str):
    if image is None:
        return None, "⚠️ **Please upload a cricket image first.**", ""
    if not question or not question.strip():
        return None, "⚠️ **Please type a question or click one of the preset buttons.**", ""

    res = PIPELINE.answer(image, question)
    vis = PIPELINE.visualize(image, res)

    qtype_pretty = QTYPE_LABEL.get(res.qtype, res.qtype.title())

    # Friendly message for free-form (BLIP-2 disabled by default)
    if res.qtype == "freeform":
        answer_md = (
            "### 🤔 Free-form questions aren't supported in this demo\n\n"
            "This system specialises in **5 structured question types** about cricket photos.\n\n"
            "**Try one of these instead** (click a button below the image):\n"
            "- 🏏 What shot is being played?\n"
            "- 🎯 Is this a batsman or a bowler?\n"
            "- ✋ Is the batsman right-handed or left-handed?\n"
            "- 👟 Is this a front-foot or back-foot shot?\n"
            "- ⚔️ Is this an attacking or defensive shot?"
        )
    else:
        # Big, clean answer — no confidence number to avoid bad-looking values
        answer_md = (
            f"## ✅ {res.answer.title()}\n\n"
            f"*Question type: **{qtype_pretty}***"
        )

    # Technical details (collapsed by default — for viva)
    timing = res.extras.get("timing", {})
    details = (
        f"- **Question type:** `{res.qtype}` — which category the system put your question in\n"
        f"- **Resolved by:** `{res.via}` — the model that answered "
        f"(ViT · CLIP · YOLO · rule · BLIP-2)\n"
        f"- **Router:** `{res.extras.get('router_via', '-')}` "
        f"(conf {res.extras.get('router_confidence', 0):.2f}) — how the question type was decided\n"
        f"- **Internal confidence:** `{res.confidence:.1%}`  *(hidden from the headline to keep the demo clean)*\n"
        f"- **Camera view:** `{res.extras.get('camera_view', '-')}` — image angle used for handedness rule\n"
        f"- **Latency:** total `{timing.get('total_ms', 0)}ms`  "
        f"(preprocess {timing.get('preprocess_ms', 0)}ms, "
        f"route {timing.get('route_ms', 0)}ms, "
        f"answer {timing.get('answer_ms', 0)}ms)"
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
        **Upload an image → click a question button → get an answer.**
        """
    )

    with gr.Row():
        # LEFT: input
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="📷 Cricket image", height=380)
            q_in = gr.Textbox(
                label="Your question",
                placeholder="Click a button below — or type your own question",
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
