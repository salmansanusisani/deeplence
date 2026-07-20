"""Agreement-aware ensemble image-classification signal for synthetic media."""

import os
import statistics
import threading

from PIL import Image


DEFAULT_MODEL_IDS = (
    "prithivMLmods/Deep-Fake-Detector-v2-Model",  # ViT, face/deepfake-focused
    "prithivMLmods/deepfake-detector-model-v1",   # SigLIP, broader synthetic-image signal
)
MODEL_IDS = tuple(
    value.strip()
    for value in os.environ.get("AUTHENTICHECK_MODELS", ",".join(DEFAULT_MODEL_IDS)).split(",")
    if value.strip()
)
MIN_ENSEMBLE_MODELS = 2
MAX_MODEL_DISAGREEMENT = 0.35
MODEL_SCOPE = (
    "Two independently loaded image classifiers are the primary signal. A result is decision-ready "
    "only when both models load and broadly agree; this does not validate every generator or edit type."
)

_models: list[tuple[str, object, object]] = []
_load_errors: dict[str, str] = {}
_load_lock = threading.Lock()
_load_attempted = False


def _try_load() -> None:
    global _load_attempted
    with _load_lock:
        if _load_attempted:
            return
        _load_attempted = True
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except Exception as error:
            for model_id in MODEL_IDS:
                _load_errors[model_id] = f"Required ML dependency is unavailable: {error}"
            return
        for model_id in MODEL_IDS:
            try:
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModelForImageClassification.from_pretrained(model_id)
                model.eval()
                _models.append((model_id, processor, model))
                print(f"[classifier] Loaded ensemble model: {model_id}")
            except Exception as error:
                _load_errors[model_id] = str(error)
                print(f"[classifier] Could not load {model_id}: {error}")


def loaded_model_count() -> int:
    if not _load_attempted:
        _try_load()
    return len(_models)


def is_available() -> bool:
    return loaded_model_count() >= MIN_ENSEMBLE_MODELS


def _fake_probability(model, processor, image: Image.Image) -> tuple[float, dict[str, float]]:
    import torch

    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        probabilities = model(**inputs).logits.softmax(dim=-1)[0].tolist()
    labels = {int(index): str(label) for index, label in model.config.id2label.items()}
    raw_probs = {labels[index]: round(float(value), 4) for index, value in enumerate(probabilities)}
    fake_terms = ("fake", "deepfake", "synthetic", "generated", "ai", "manipulated")
    fake_indices = [index for index, label in labels.items() if any(term in label.lower().replace("-", "_") for term in fake_terms)]
    if len(fake_indices) != 1:
        raise ValueError(f"Could not identify exactly one synthetic-media label: {labels}")
    return float(probabilities[fake_indices[0]]), raw_probs


def classify(pil_image: Image.Image) -> dict | None:
    """Run every available model and return a disagreement-aware ensemble score."""
    loaded_model_count()
    if not _models:
        return None
    individual = []
    for model_id, processor, model in _models:
        try:
            score, raw_probs = _fake_probability(model, processor, pil_image)
            individual.append({"model": model_id, "score": score, "raw_probs": raw_probs})
        except Exception as error:
            _load_errors[model_id] = f"Inference failed: {error}"
    if not individual:
        return None

    scores = [item["score"] for item in individual]
    spread = max(scores) - min(scores) if len(scores) > 1 else 1.0
    ensemble_complete = len(individual) >= MIN_ENSEMBLE_MODELS
    agreement = ensemble_complete and spread <= MAX_MODEL_DISAGREEMENT
    notes = [f"{item['model'].split('/')[-1]} estimates {item['score'] * 100:.1f}% synthetic-media risk." for item in individual]
    if not ensemble_complete:
        notes.append("Ensemble incomplete: no decision is made until both configured models are available.")
    elif not agreement:
        notes.append(f"Model disagreement is {spread * 100:.1f} points; the result is inconclusive.")
    else:
        notes.append(f"Models agree within {spread * 100:.1f} points.")
    return {
        "score": float(statistics.median(scores)),
        "notes": notes,
        "details": {
            "models": individual,
            "configured_model_ids": list(MODEL_IDS),
            "model_count": len(individual),
            "ensemble_complete": ensemble_complete,
            "agreement": agreement,
            "score_spread": round(spread, 4),
            "load_errors": _load_errors,
            "scope": MODEL_SCOPE,
        },
    }
