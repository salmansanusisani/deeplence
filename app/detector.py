"""
Core forensic analysis engine.

Combines several independent, explainable signals into one fused
"AI-generated probability" score:

  1. Metadata / provenance      - EXIF camera data, C2PA content credentials
  2. Error Level Analysis (ELA) - JPEG recompression inconsistency
  3. Frequency / FFT analysis   - GAN/diffusion upsampling artifacts
  4. Noise residual analysis    - real sensor noise vs "too clean" synthetic noise

Each signal returns a score from 0.0 (looks real) to 1.0 (looks AI-generated),
plus a human-readable note. Scores are fused with fixed weights into a final
percentage. This is a heuristic forensic prototype, not a trained deep model -
it is meant to be transparent and explainable, which is exactly what makes it
a strong hackathon demo (you can point at *why* a score was given).
"""

import io
import numpy as np
from PIL import Image, ImageChops, ExifTags
from scipy import ndimage

from app import classifier


# A neural classifier will resize every input to its preferred dimensions, but
# that cannot recreate the forensic detail lost in a tiny source thumbnail.
MIN_ANALYZABLE_EDGE = 128


def assess_image_suitability(pil_image: Image.Image) -> dict:
    """Report whether the source supports a detector decision.

    This is an eligibility check, not a real/fake signal: it never changes the
    risk score. It only prevents a misleading confident verdict on thumbnails.
    """
    width, height = pil_image.size
    shortest_edge = min(width, height)
    suitable = shortest_edge >= MIN_ANALYZABLE_EDGE
    if suitable:
        note = f"Source resolution is {width}x{height}; sufficient for a preliminary image-forensics assessment."
    else:
        note = (
            f"Source resolution is only {width}x{height}. Images below {MIN_ANALYZABLE_EDGE}px "
            "on the shortest edge are inconclusive because resizing cannot restore lost forensic detail."
        )
    return {
        "suitable_for_decision": suitable,
        "width": width,
        "height": height,
        "shortest_edge": shortest_edge,
        "notes": [note],
    }


# ---------------------------------------------------------------------------
# 1. Metadata / provenance signal
# ---------------------------------------------------------------------------
def analyze_metadata(pil_image: Image.Image, raw_bytes: bytes, provenance: dict | None = None) -> dict:
    score = 0.5  # neutral default if nothing conclusive is found
    notes = []
    details = {}

    # Look for C2PA / Content Credentials manifest markers embedded in the file
    c2pa_markers = [b"c2pa", b"C2PA", b"jumbf", b"contentcredentials"]
    has_c2pa = any(marker in raw_bytes for marker in c2pa_markers)
    details["c2pa_manifest_found"] = has_c2pa
    details["c2pa_validation"] = "not_present"

    if provenance:
        details["c2pa_validation"] = provenance.get("status", "not checked")
        details["provenance_issuer_hints"] = provenance.get("issuer_hints", [])
        if provenance.get("declared_ai_generation"):
            score = 0.98
            notes.append("C2PA validator report declares an AI-related issuer or generation action.")
            details["provenance_status"] = "declared_ai_provenance"
            return {"score": score, "notes": notes, "details": details}

    if has_c2pa:
        # A byte marker is not a validated Content Credential. A proper C2PA
        # validator must verify the manifest, signature, trust chain, and that
        # the binding still matches this asset. It can also describe AI edits.
        score = 0.5
        notes.append("Possible Content Credentials marker found, but it has not been cryptographically validated. This is not proof of authenticity.")
        details["c2pa_validation"] = "unvalidated_marker"

    # EXIF check
    exif_data = {}
    try:
        exif_raw = pil_image.getexif()
        if exif_raw:
            for tag_id, value in exif_raw.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[str(tag)] = str(value)[:100]
    except Exception:
        pass

    details["exif_tag_count"] = len(exif_data)
    details["exif_sample"] = dict(list(exif_data.items())[:6])

    camera_indicators = ["Make", "Model", "LensModel", "FNumber", "ExposureTime", "ISOSpeedRatings"]
    found_camera_tags = [t for t in camera_indicators if t in exif_data]
    details["camera_tags_found"] = found_camera_tags

    # AI generator tools sometimes leave their own software tag / prompt text
    software = exif_data.get("Software", "")
    ai_tool_hints = ["stable diffusion", "midjourney", "dall-e", "dalle", "comfyui",
                     "automatic1111", "leonardo", "firefly", "runway", "sora"]
    software_lower = software.lower()
    if any(hint in software_lower for hint in ai_tool_hints):
        score = 0.95
        notes.append(f"Metadata 'Software' tag references a known generative tool ({software}).")
        details["provenance_status"] = "generator_tool_declared"
        return {"score": score, "notes": notes, "details": details}

    if len(found_camera_tags) >= 2:
        score = 0.5
        notes.append(f"Camera EXIF data is present ({', '.join(found_camera_tags)}). It is useful context but can be copied or altered, so it is not proof of authenticity.")
        details["provenance_status"] = "camera_metadata_present"
    elif len(exif_data) == 0:
        score = 0.5
        notes.append("No EXIF metadata is available. This is common after screenshots, edits, and social-media uploads, so it is not evidence of AI generation.")
        details["provenance_status"] = "metadata_unavailable"
    else:
        score = 0.5
        notes.append("Partial/inconclusive metadata present.")
        details["provenance_status"] = "inconclusive"

    return {"score": score, "notes": notes, "details": details}


# ---------------------------------------------------------------------------
# 2. Error Level Analysis (ELA)
# ---------------------------------------------------------------------------
def error_level_analysis(pil_image: Image.Image) -> dict:
    rgb = pil_image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, "JPEG", quality=90)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(rgb, resaved)
    diff_arr = np.asarray(diff).astype(np.float32)

    mean_error = float(diff_arr.mean())
    std_error = float(diff_arr.std())
    h, w = diff_arr.shape[:2]

    # Adaptive block size: shrink for small images so we still get a
    # meaningful number of blocks instead of degenerating to 0-1 blocks
    # (which previously forced uniformity to a constant, falsely-high value
    # regardless of whether the image was real or fake).
    block_size = min(16, max(4, min(h, w) // 8))
    gray = diff_arr.mean(axis=2)
    block_means = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block_means.append(gray[y:y + block_size, x:x + block_size].mean())
    block_means = np.array(block_means)

    if len(block_means) < 4:
        # Not enough spatial resolution to make a reliable claim - stay neutral
        # rather than defaulting to a biased score.
        return {
            "score": 0.5,
            "notes": ["Image too small for reliable ELA analysis — signal skipped (neutral score)."],
            "details": {"mean_error": round(mean_error, 3), "std_error": round(std_error, 3), "uniformity": None},
            "heatmap_stats": {"max_error": float(diff_arr.max())}
        }

    uniformity = float(1.0 - min(block_means.std() / (block_means.mean() + 1e-6), 1.0))

    # Fuse: very low overall error + high uniformity => more "AI-like"
    score = 0.3 + 0.4 * uniformity + (0.2 if mean_error < 2.0 else 0.0)
    score = float(np.clip(score, 0.0, 1.0))

    notes = [f"ELA spatial uniformity: {uniformity:.2f} (higher = more suspicious), mean error level: {mean_error:.2f}."]

    return {
        "score": score,
        "notes": notes,
        "details": {"mean_error": round(mean_error, 3), "std_error": round(std_error, 3), "uniformity": round(uniformity, 3)},
        "heatmap_stats": {"max_error": float(diff_arr.max())}
    }


# ---------------------------------------------------------------------------
# 3. Frequency domain / FFT analysis
# ---------------------------------------------------------------------------
def frequency_analysis(pil_image: Image.Image) -> dict:
    gray = np.asarray(pil_image.convert("L")).astype(np.float32)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # radial average energy falloff
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    max_r = min(cx, cy)
    radial_profile = ndimage.mean(magnitude, labels=r, index=np.arange(0, max_r))

    # Natural photos: smooth, roughly monotonic falloff of energy from low to high freq.
    # GAN/diffusion outputs: often show unnatural bumps/periodic peaks (upsampling
    # checkerboard artifacts) or an artificially fast falloff (over-smoothed textures).
    if len(radial_profile) > 10:
        diffs = np.diff(radial_profile)
        # count sign changes in the tail (high-frequency) region as a proxy for periodic bumps
        tail = diffs[len(diffs) // 2:]
        sign_changes = int(np.sum(np.diff(np.sign(tail)) != 0))
        bumpiness = sign_changes / max(len(tail), 1)
    else:
        bumpiness = 0.0

    # high frequency energy ratio - unnaturally low can indicate over-smoothing typical
    # of diffusion decoders; unnaturally periodic can indicate GAN upsampling
    high_freq_energy = float(np.mean(radial_profile[int(max_r * 0.7):])) if max_r > 3 else 0.0
    low_freq_energy = float(np.mean(radial_profile[:int(max_r * 0.2)])) if max_r > 3 else 1.0
    hf_ratio = high_freq_energy / (low_freq_energy + 1e-6)

    score = 0.5
    notes = []
    if bumpiness > 0.35:
        score += 0.25
        notes.append(f"Periodic bumps detected in high-frequency spectrum (bumpiness={bumpiness:.2f}) — resembles GAN/upsampling artifacts.")
    if hf_ratio < 0.15:
        score += 0.2
        notes.append(f"High-frequency energy unusually low (ratio={hf_ratio:.2f}) — resembles over-smoothed diffusion output.")
    if not notes:
        notes.append(f"Frequency spectrum falloff looks broadly natural (bumpiness={bumpiness:.2f}, hf_ratio={hf_ratio:.2f}).")
        score = 0.35

    score = float(np.clip(score, 0.0, 1.0))
    return {
        "score": score,
        "notes": notes,
        "details": {"bumpiness": round(bumpiness, 3), "hf_ratio": round(hf_ratio, 3)}
    }


# ---------------------------------------------------------------------------
# 4. Noise residual analysis
# ---------------------------------------------------------------------------
def noise_residual_analysis(pil_image: Image.Image) -> dict:
    gray = np.asarray(pil_image.convert("L")).astype(np.float32)

    # high-pass filter (image minus a heavily blurred version) isolates noise/texture
    blurred = ndimage.gaussian_filter(gray, sigma=2)
    residual = gray - blurred

    # Real camera sensor noise is spatially correlated / textured due to optics,
    # demosaicing, and sensor pattern noise. AI-generated images frequently have
    # noise that is either near-zero (too "clean") or statistically flatter/more
    # uniform across the frame (no real sensor pattern).
    global_std = float(residual.std())
    h, w = residual.shape

    block_size = min(16, max(4, min(h, w) // 8))
    block_stds = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block_stds.append(residual[y:y + block_size, x:x + block_size].std())
    block_stds = np.array(block_stds)

    if len(block_stds) < 4:
        return {
            "score": 0.5,
            "notes": ["Image too small for reliable noise analysis — signal skipped (neutral score)."],
            "details": {"global_noise_std": round(global_std, 3), "spatial_variability": None}
        }

    variability = float(block_stds.std())  # how much noise level varies across regions

    score = 0.5
    notes = []
    if global_std < 1.5:
        score = 0.75
        notes.append(f"Noise floor unusually low (std={global_std:.2f}) — real camera sensor noise is typically higher.")
    elif variability < 0.5:
        score = 0.65
        notes.append(f"Noise pattern is spatially uniform across the image (variability={variability:.2f}) — real sensor noise usually varies with local texture/lighting.")
    else:
        score = 0.3
        notes.append(f"Noise pattern shows natural spatial variability (std={global_std:.2f}, variability={variability:.2f}), consistent with a real camera sensor.")

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "notes": notes,
        "details": {"global_noise_std": round(global_std, 3), "spatial_variability": round(variability, 3)}
    }


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
# Weights used when the neural classifier IS available. It gets the largest
# single weight since a trained model typically outperforms hand-built
# heuristics on modern generators - the heuristics still matter for
# explainability and as a check against the classifier being wrong/uncertain.
SIGNAL_WEIGHTS_WITH_NEURAL = {
    # A trained detector is the primary signal. Metadata only has material
    # influence when a generator explicitly declares itself; missing camera
    # data remains neutral. Heuristics are supporting evidence, not proof.
    "neural_classifier": 0.70,
    "metadata": 0.10,
    "ela": 0.07,
    "frequency": 0.07,
    "noise": 0.06,
}

# Weights used when it's not available (pure heuristic fallback) - same as before.
SIGNAL_WEIGHTS = {
    "metadata": 0.30,
    "ela": 0.20,
    "frequency": 0.25,
    "noise": 0.25,
}


def interpret_risk(
    risk_percent: float,
    trained_detector_used: bool,
    *,
    declared_ai_provenance: bool = False,
    suitable_for_decision: bool = True,
    ensemble_ready: bool = True,
) -> tuple[str, str]:
    """Return cautious, user-facing language for a forensic risk estimate."""
    if declared_ai_provenance:
        return "AI origin declared in signed provenance", "high"
    if not suitable_for_decision:
        return "Inconclusive — image resolution too low", "low"
    if not ensemble_ready:
        return "Inconclusive — detector ensemble disagrees or is incomplete", "low"
    if not trained_detector_used:
        return "Inconclusive — trained detector unavailable", "low"
    if risk_percent >= 70:
        return "High synthetic-media risk", "moderate"
    if risk_percent >= 45:
        return "Inconclusive — needs review", "low"
    return "No strong AI evidence found", "moderate"


def analyze_image(pil_image: Image.Image, raw_bytes: bytes, provenance: dict | None = None) -> dict:
    suitability = assess_image_suitability(pil_image)
    signals = {
        "metadata": analyze_metadata(pil_image, raw_bytes, provenance),
        "ela": error_level_analysis(pil_image),
        "frequency": frequency_analysis(pil_image),
        "noise": noise_residual_analysis(pil_image),
    }

    neural_result = classifier.classify(pil_image)
    if neural_result is not None:
        signals["neural_classifier"] = neural_result

    neural_used = neural_result is not None
    ensemble_ready = bool(
        neural_result
        and neural_result["details"].get("ensemble_complete")
        and neural_result["details"].get("agreement")
    )
    declared_ai_provenance = bool((provenance or {}).get("declared_ai_generation"))
    if neural_result is not None:
        # The ensemble is the primary evidence. Legacy heuristic signals are
        # only a bounded adjustment, so they cannot drag a strong model
        # prediction down toward an arbitrary neutral score.
        forensic_support = (
            signals["ela"]["score"] * 0.35
            + signals["frequency"]["score"] * 0.35
            + signals["noise"]["score"] * 0.30
        )
        fused = float(np.clip(neural_result["score"] + 0.12 * (forensic_support - 0.5), 0.0, 1.0))
    else:
        fused = sum(signals[key]["score"] * SIGNAL_WEIGHTS[key] for key in SIGNAL_WEIGHTS)
    # A valid signed declaration is stronger evidence than a statistical image
    # classifier. Keep this separate from the model's risk estimate so a
    # confirmed AI-origin asset is not misleadingly diluted by visual signals.
    fused_pct = 100.0 if declared_ai_provenance else round(fused * 100, 1)
    verdict, confidence = interpret_risk(
        fused_pct,
        neural_used,
        declared_ai_provenance=declared_ai_provenance,
        suitable_for_decision=suitability["suitable_for_decision"],
        ensemble_ready=ensemble_ready,
    )

    return {
        "ai_probability_percent": fused_pct,
        "verdict": verdict,
        "confidence": confidence,
        "trained_detector_used": ensemble_ready,
        "detector_models_loaded": neural_result["details"]["model_count"] if neural_result else 0,
        "decision_ready": declared_ai_provenance or (ensemble_ready and suitability["suitable_for_decision"]),
        "risk_basis": (
            "verified_signed_ai_provenance" if declared_ai_provenance
            else "trained_detector_risk_estimate" if ensemble_ready
            else "non_decisive_forensic_signals"
        ),
        "image_suitability": suitability,
        "analysis_limitations": [
            "This is a forensic risk estimate, not proof that media is real or AI-generated.",
            classifier.MODEL_SCOPE if neural_used else "No trained detector was available; heuristic signals cannot establish whether media is AI-generated.",
            suitability["notes"][0],
            "Missing metadata is neutral because it is commonly removed by platforms and editors.",
            "Content Credentials require cryptographic validation before they can establish provenance.",
        ],
        "provenance": provenance or {"status": "not checked", "note": "No C2PA validation was performed."},
        "signals": signals,
    }
