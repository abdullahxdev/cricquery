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

PRESET_QUESTIONS = [
    "What shot is being played?",
    "Is this player a batsman or a bowler?",
    "Is the batsman right-handed or left-handed?",
    "Is this a front-foot or back-foot shot?",
    "Is this an attacking or defensive shot?",
    "Describe what is happening in this image.",
]


def run(image: Image.Image, question: str):
    if image is None:
        return None, "Please upload an image first.", ""
    if not question or not question.strip():
        return None, "Please type a question.", ""
    res = PIPELINE.answer(image, question)
    vis = PIPELINE.visualize(image, res)
    details = (
        f"**Question type:** `{res.qtype}`  \n"
        f"**Resolved by:** `{res.via}`  \n"
        f"**Router:** `{res.extras.get('router_via', '-')}` "
        f"(conf {res.extras.get('router_confidence', 0):.2f})  \n"
        f"**Camera view:** `{res.extras.get('camera_view', '-')}`"
    )
    answer_md = f"### {res.answer}\n\n*Confidence: {res.confidence:.1%}*"
    return vis, answer_md, details


def list_examples():
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(str(p) for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


with gr.Blocks(title="CricQuery — AI Cricket Commentator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏏 CricQuery — AI Cricket Commentator
        Multimodal Visual Question Answering for cricket photos. Upload an image,
        ask a question, get a structured answer with confidence.

        **Supported question types:** shot · role · handedness · foot position · intent · free-form (BLIP-2)
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="Cricket image")
            q_in = gr.Textbox(label="Question", placeholder="e.g. What shot is being played?")
            with gr.Row():
                go = gr.Button("Answer 🎯", variant="primary")
                clr = gr.Button("Clear")
            gr.Markdown("**Try one of these:**")
            for q in PRESET_QUESTIONS:
                gr.Button(q, size="sm").click(lambda x=q: x, None, q_in)
            ex = list_examples()
            if ex:
                gr.Examples(examples=[[e] for e in ex], inputs=[img_in], label="Example images")
        with gr.Column(scale=1):
            img_out = gr.Image(label="Annotated result", type="numpy")
            ans_md = gr.Markdown()
            dbg_md = gr.Markdown()

    go.click(run, [img_in, q_in], [img_out, ans_md, dbg_md])
    clr.click(lambda: (None, "", "", ""), None, [img_in, img_out, ans_md, dbg_md])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False, inbrowser=True)
