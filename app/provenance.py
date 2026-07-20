"""Optional, evidence-first C2PA provenance inspection.

This module uses the official ``c2patool`` binary when it is installed on the
machine. It intentionally fails closed: a marker in the bytes is never called
verified provenance, and an unavailable validator never changes the forensic
risk score.
"""

import json
import os
import shutil
import subprocess
import tempfile
from typing import Any


AI_ISSUER_HINTS = (
    "openai", "chatgpt", "dall-e", "sora", "adobe", "firefly", "google",
    "gemini", "imagen", "meta ai", "midjourney", "runway", "stability",
    "stable diffusion", "comfyui", "leonardo",
)


def _find_tool() -> str | None:
    configured = os.environ.get("C2PATOOL_PATH")
    if configured and os.path.isfile(configured):
        return configured
    return shutil.which("c2patool") or shutil.which("c2patool.exe")


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _walk_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _walk_strings(child)]
    return [str(value)]


def _short_report(report: Any) -> dict:
    """Keep the useful, safe-to-display summary instead of returning all claims."""
    text = " ".join(_walk_strings(report)).lower()
    issuer_matches = sorted({hint for hint in AI_ISSUER_HINTS if hint in text})
    action_terms = ("trainedalgorithmicmedia", "generative", "ai_generated", "ai-generated")
    declared_ai = bool(issuer_matches) or any(term in text for term in action_terms)

    # c2patool supplies validation status information in its JSON report. Do
    # not infer success simply because a manifest was found.
    failure_terms = ("invalid", "mismatch", "failure", "error", "revoked")
    validation_failed = any(term in text for term in failure_terms)
    status = "validation issues reported" if validation_failed else "report available for review"

    return {
        "validator_available": True,
        "manifest_found": True,
        "status": status,
        "declared_ai_generation": declared_ai and not validation_failed,
        "issuer_hints": issuer_matches,
        "raw_report_available": True,
        "note": (
            "The C2PA report was read with c2patool. AuthentiCheck shows the "
            "issuer and declared actions as provenance evidence; it does not "
            "treat a missing credential as evidence of authenticity."
        ),
    }


def inspect_c2pa(raw_bytes: bytes, suffix: str) -> dict:
    """Read a C2PA report with c2patool, if available, without using a shell."""
    tool = _find_tool()
    if not tool:
        return {
            "validator_available": False,
            "manifest_found": False,
            "status": "validator unavailable",
            "declared_ai_generation": False,
            "issuer_hints": [],
            "raw_report_available": False,
            "note": "Install c2patool to validate Content Credentials. Byte markers alone are not trusted.",
        }

    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(raw_bytes)
            path = handle.name
        completed = subprocess.run(
            [tool, path],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        output = completed.stdout.strip()
        if not output:
            return {
                "validator_available": True,
                "manifest_found": False,
                "status": "no readable Content Credentials found",
                "declared_ai_generation": False,
                "issuer_hints": [],
                "raw_report_available": False,
                "note": "No C2PA report was returned by the validator. This does not prove the media is authentic.",
            }
        try:
            report = json.loads(output)
        except json.JSONDecodeError:
            return {
                "validator_available": True,
                "manifest_found": False,
                "status": "validator returned an unreadable report",
                "declared_ai_generation": False,
                "issuer_hints": [],
                "raw_report_available": False,
                "note": "The validator could not produce a readable C2PA report for this file.",
            }
        if not report:
            return {
                "validator_available": True,
                "manifest_found": False,
                "status": "no Content Credentials found",
                "declared_ai_generation": False,
                "issuer_hints": [],
                "raw_report_available": False,
                "note": "No signed provenance was found. This does not prove the media is authentic.",
            }
        return _short_report(report)
    except subprocess.TimeoutExpired:
        return {
            "validator_available": True,
            "manifest_found": False,
            "status": "validation timed out",
            "declared_ai_generation": False,
            "issuer_hints": [],
            "raw_report_available": False,
            "note": "C2PA validation did not finish in time; no provenance conclusion was made.",
        }
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
