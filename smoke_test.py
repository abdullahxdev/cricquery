"""Quick sanity check — run BEFORE doing anything else.

Verifies that:
  1. All Python imports resolve
  2. Hugging Face can download CLIP (small, fast)
  3. The question router classifies a sample sentence correctly
  4. The pipeline can be constructed (with a missing ViT model — it logs a warning)

Does NOT require the trained ViT, Kaggle data, or any cricket images.

Run:  python smoke_test.py
"""
import sys
import traceback


def check(label, fn):
    sys.stdout.write(f"[ .. ] {label}")
    sys.stdout.flush()
    try:
        fn()
        print(f"\r[ OK ] {label}")
        return True
    except Exception as e:
        print(f"\r[FAIL] {label}")
        print(f"       {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return False


def import_torch():
    import torch
    assert torch.tensor([1.0]).sum().item() == 1.0


def import_transformers():
    import transformers  # noqa: F401


def import_ultralytics():
    import ultralytics  # noqa: F401


def import_cv2():
    import cv2  # noqa: F401


def import_mediapipe():
    import mediapipe  # noqa: F401


def import_gradio():
    import gradio  # noqa: F401


def import_project_modules():
    from src import config  # noqa: F401
    from src import question_router  # noqa: F401
    from src import preflight  # noqa: F401
    from src import pipeline  # noqa: F401
    from src import visualize  # noqa: F401
    from src.inference import clip_qa  # noqa: F401
    from src.inference import shot_classifier  # noqa: F401
    from src.inference import yolo_handedness  # noqa: F401
    from src.inference import intent_rule  # noqa: F401
    from src.inference import blip2_fallback  # noqa: F401


def test_router_regex():
    from src.question_router import QuestionRouter
    r = QuestionRouter()
    expectations = {
        "What shot is being played?": "shot",
        "Is the batsman right-handed?": "handedness",
        "Is this a batsman or a bowler?": "role",
        "front foot or back foot?": "foot",
        "Is this attacking or defensive?": "intent",
    }
    for q, want in expectations.items():
        got = r.route(q).qtype
        assert got == want, f"router({q!r}) -> {got}, expected {want}"


def construct_pipeline():
    # Disable BLIP-2 so this is fast and offline-resilient
    from src.pipeline import CricQueryPipeline
    _ = CricQueryPipeline(enable_blip2=False)


if __name__ == "__main__":
    results = []
    print("Smoke test for CricQuery — should take ~30s on first run\n")
    results.append(check("Import torch", import_torch))
    results.append(check("Import transformers", import_transformers))
    results.append(check("Import ultralytics", import_ultralytics))
    results.append(check("Import opencv-python", import_cv2))
    results.append(check("Import mediapipe", import_mediapipe))
    results.append(check("Import gradio", import_gradio))
    results.append(check("Import project modules", import_project_modules))
    results.append(check("Question router regex tests", test_router_regex))
    results.append(check("Construct full pipeline (no BLIP-2, no ViT weights)",
                        construct_pipeline))

    n_pass = sum(results)
    n_total = len(results)
    print()
    if n_pass == n_total:
        print(f"✓ All {n_total} checks passed. You are ready to train + demo.")
        sys.exit(0)
    else:
        print(f"✗ {n_total - n_pass} of {n_total} checks failed. Fix these first.")
        sys.exit(1)
