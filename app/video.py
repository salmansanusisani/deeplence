"""
Video analysis: samples frames across the clip, runs the same forensic
signals used for images (plus the optional neural classifier) on each
sampled frame, then aggregates them and adds a temporal-consistency signal.

Temporal signal rationale: real footage has natural frame-to-frame noise
and micro-variation from an actual sensor + real-world motion. Fully
AI-generated video frequently shows either unnaturally *smooth* frame-to-frame
transitions (over-consistent) or, at generation-boundary frames, sudden score
spikes/dips as the temporal model "reimagines" a region. Both irregular and
overly smooth patterns are used here as weak evidence.
"""

import cv2
import numpy as np
from PIL import Image

from app import classifier
from app.detector import (
    analyze_metadata,
    error_level_analysis,
    frequency_analysis,
    noise_residual_analysis,
    SIGNAL_WEIGHTS,
    SIGNAL_WEIGHTS_WITH_NEURAL,
    interpret_risk,
)


MAX_FRAMES_SAMPLED = 12


def _frame_to_pil(frame_bgr) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def analyze_video(video_path: str, raw_bytes_for_metadata: bytes, provenance: dict | None = None) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0

    sample_count = min(MAX_FRAMES_SAMPLED, max(total_frames, 1))
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, sample_count).astype(int)
    else:
        indices = np.arange(sample_count)

    neural_on = classifier.is_available()
    w = SIGNAL_WEIGHTS_WITH_NEURAL if neural_on else SIGNAL_WEIGHTS
    non_metadata_total = 1 - w["metadata"]

    per_frame_scores = []
    per_frame_breakdowns = []
    prev_gray = None
    motion_smoothness = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        pil_frame = _frame_to_pil(frame)

        ela = error_level_analysis(pil_frame)
        freq = frequency_analysis(pil_frame)
        noise = noise_residual_analysis(pil_frame)

        frame_score = (
            ela["score"] * (w["ela"] / non_metadata_total)
            + freq["score"] * (w["frequency"] / non_metadata_total)
            + noise["score"] * (w["noise"] / non_metadata_total)
        )
        frame_breakdown = {"ela": ela["score"], "frequency": freq["score"], "noise": noise["score"]}

        if neural_on:
            neural_result = classifier.classify(pil_frame)
            if neural_result is not None:
                frame_score += neural_result["score"] * (w["neural_classifier"] / non_metadata_total)
                frame_breakdown["neural_classifier"] = neural_result["score"]

        per_frame_scores.append(frame_score)
        per_frame_breakdowns.append(frame_breakdown)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, prev_gray)
            motion_smoothness.append(float(diff.std()))
        prev_gray = gray

    cap.release()

    if not per_frame_scores:
        raise ValueError("No frames could be read from video")

    mean_frame_score = float(np.mean(per_frame_scores))
    frame_variance = float(np.var(per_frame_scores))

    temporal_notes = []
    if frame_variance < 0.001:
        temporal_score = 0.7
        temporal_notes.append(f"Forensic score is unusually consistent across sampled frames (variance={frame_variance:.5f}) — can indicate synthetic generation.")
    else:
        temporal_score = 0.35
        temporal_notes.append(f"Forensic score varies naturally across frames (variance={frame_variance:.5f}), consistent with real captured footage.")

    if motion_smoothness:
        motion_std = float(np.std(motion_smoothness))
        if motion_std < 1.0:
            temporal_notes.append(f"Frame-to-frame motion is unusually smooth (std={motion_std:.2f}).")
            temporal_score = min(1.0, temporal_score + 0.1)

    first_frame_ok, first_frame = cv2.VideoCapture(video_path).read()
    metadata_pil = _frame_to_pil(first_frame) if first_frame_ok else _frame_to_pil(np.zeros((10, 10, 3), dtype=np.uint8))
    metadata_signal = analyze_metadata(metadata_pil, raw_bytes_for_metadata, provenance)

    fused = (
        metadata_signal["score"] * w["metadata"]
        + mean_frame_score * non_metadata_total * 0.75
        + temporal_score * non_metadata_total * 0.25
    )
    fused_pct = round(float(np.clip(fused, 0, 1)) * 100, 1)

    verdict, confidence = interpret_risk(fused_pct, neural_on)

    return {
        "ai_probability_percent": fused_pct,
        "verdict": verdict,
        "confidence": confidence,
        "frames_analyzed": len(per_frame_scores),
        "total_frames": total_frames,
        "fps": fps,
        "neural_classifier_used": neural_on,
        "trained_detector_used": neural_on,
        "analysis_limitations": [
            "This is a forensic risk estimate, not proof that media is real or AI-generated.",
            classifier.MODEL_SCOPE if neural_on else "No trained detector was available; heuristic signals cannot establish whether media is AI-generated.",
            "Only sampled frames were analyzed; manipulation can be missed between them.",
            "Content Credentials require cryptographic validation before they can establish provenance.",
        ],
        "provenance": provenance or {"status": "not checked", "note": "No C2PA validation was performed."},
        "signals": {
            "metadata": metadata_signal,
            "per_frame_mean_forensic_score": round(mean_frame_score, 3),
            "temporal_consistency": {"score": round(temporal_score, 3), "notes": temporal_notes, "details": {"frame_variance": frame_variance}},
        },
        "per_frame_breakdown": per_frame_breakdowns,
    }
