"""Server-side Sightengine GenAI detector integration.

Credentials are deliberately read only on the server. Never put these values
in HTML or browser JavaScript: doing so would expose the account to anyone who
opens the app.
"""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Loads the .env file created by setup_credentials.py (project root, one
# level up from this app/ package) so SIGHTENGINE_API_USER/SECRET are
# available as environment variables without the user setting them by hand.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ENDPOINT = "https://api.sightengine.com/1.0/check.json"
LEGACY_CONFIG_PATH = Path.home() / "Desktop" / "config.json"


def _generator_scores(type_result: dict) -> list[dict]:
    """Keep every numeric per-generator signal returned by Sightengine.

    The provider adds generator classes over time, so this intentionally does
    not hard-code a stale list. `ai_generated` is the overall score, not a
    generator fingerprint, and is handled separately.
    """
    scores = []
    for name, value in type_result.items():
        if name == "ai_generated" or not isinstance(value, (int, float)):
            continue
        scores.append({"generator": name, "score": round(max(0.0, min(1.0, float(value))), 4)})
    return sorted(scores, key=lambda item: item["score"], reverse=True)


def _credentials() -> tuple[str | None, str | None]:
    """Use environment variables first, then the existing desktop script config."""
    api_user = os.getenv("SIGHTENGINE_API_USER")
    api_secret = os.getenv("SIGHTENGINE_API_SECRET")
    if api_user and api_secret:
        return api_user, api_secret

    # This allows the original check_ai_image.py setup to work locally without
    # copying a secret into the repository. It is never used in a response.
    if LEGACY_CONFIG_PATH.is_file():
        try:
            config = json.loads(LEGACY_CONFIG_PATH.read_text(encoding="utf-8"))
            return config.get("api_user"), config.get("api_secret")
        except (OSError, json.JSONDecodeError):
            pass
    return None, None


def is_configured() -> bool:
    api_user, api_secret = _credentials()
    return bool(api_user and api_secret)


def analyze_image(raw_bytes: bytes, filename: str) -> dict:
    api_user, api_secret = _credentials()
    if not api_user or not api_secret:
        raise RuntimeError(
            "Sightengine is not configured. Run `python setup_credentials.py` "
            "from the project root to save your API user/secret, or set "
            "SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET manually."
        )

    try:
        response = requests.post(
            ENDPOINT,
            files={"media": (filename or "image.jpg", raw_bytes)},
            data={"models": "genai", "api_user": api_user, "api_secret": api_secret},
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise RuntimeError(f"Sightengine request failed: {error}") from error
    except ValueError as error:
        raise RuntimeError("Sightengine returned an unreadable response") from error

    if payload.get("status") != "success":
        error = payload.get("error", {})
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise RuntimeError(f"Sightengine analysis failed: {message or 'unknown API error'}")

    type_result = payload.get("type", {})
    score = type_result.get("ai_generated")
    if not isinstance(score, (int, float)):
        raise RuntimeError("Sightengine response did not include an AI-generated score")
    score = max(0.0, min(1.0, float(score)))
    percent = round(score * 100, 1)
    generator_scores = _generator_scores(type_result)
    request = payload.get("request", {})
    if score >= 0.85:
        verdict, confidence = "High synthetic-media risk", "high"
    elif score >= 0.50:
        verdict, confidence = "Likely synthetic — review recommended", "moderate"
    else:
        verdict, confidence = "No strong AI evidence found", "moderate"

    return {
        "ai_probability_percent": percent,
        "verdict": verdict,
        "confidence": confidence,
        "trained_detector_used": True,
        "detector_models_loaded": 1,
        "decision_ready": True,
        "risk_basis": "sightengine_genai_api",
        "provider_evidence": {
            "provider": "Sightengine",
            "method": "Pixel-based GenAI analysis",
            "request_id": request.get("id"),
            "timestamp": request.get("timestamp"),
            "operations": request.get("operations"),
            "generator_scores": generator_scores,
            "note": (
                "Generator scores are confidence signals returned by Sightengine. "
                "They indicate visual similarity to supported generator families, not a verified source attribution."
            ),
        },
        "analysis_limitations": [
            "This is a model risk estimate, not proof that an image is real or false.",
            "The original uploaded file is sent to Sightengine for analysis; do not use this mode for media you cannot share with that service.",
            "A result below 100% is not an error: model confidence is not cryptographic provenance.",
        ],
        "signals": {
            "sightengine_genai": {
                "score": score,
                "notes": [
                    "Sightengine analyzed pixel content with its GenAI model; metadata and EXIF are not used for this score.",
                    f"{len(generator_scores)} generator-confidence signal(s) were returned by the provider.",
                ],
                "details": {
                    "provider": "Sightengine",
                    "model": "genai",
                    "generator_scores": generator_scores,
                    "request_id": request.get("id"),
                },
            }
        },
    }
