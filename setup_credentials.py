"""
One-time setup script: prompts for your Sightengine API credentials and
saves them to a local .env file so DeepLence can find them automatically
on every future run (no manual environment variables needed).

Run this once after cloning the repo, from the project root:

    python setup_credentials.py

Get your API user/secret from https://sightengine.com -> Dashboard ->
API credentials. The values are written only to a .env file in this
project folder (already excluded from git via .gitignore) - nothing is
sent anywhere else by this script.
"""

import os
import sys

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _read_existing() -> dict:
    values = {}
    if os.path.isfile(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def _write(values: dict) -> None:
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    existing = _read_existing()

    current_user = existing.get("SIGHTENGINE_API_USER") or os.environ.get("SIGHTENGINE_API_USER")
    current_secret = existing.get("SIGHTENGINE_API_SECRET") or os.environ.get("SIGHTENGINE_API_SECRET")

    if current_user and current_secret:
        print(f"Sightengine credentials are already saved (user: {current_user}).")
        answer = input("Overwrite them? [y/N]: ").strip().lower()
        if answer != "y":
            print("Keeping existing credentials. Nothing changed.")
            return

    print("Enter your Sightengine API credentials (from sightengine.com dashboard).")
    api_user = input("API user: ").strip()
    api_secret = input("API secret: ").strip()

    if not api_user or not api_secret:
        print("Both fields are required. Aborting.")
        sys.exit(1)

    existing["SIGHTENGINE_API_USER"] = api_user
    existing["SIGHTENGINE_API_SECRET"] = api_secret
    _write(existing)

    print(f"\nSaved to {ENV_PATH}")
    print("DeepLence will now load these automatically on every run - no need to set environment variables by hand.")


if __name__ == "__main__":
    main()
