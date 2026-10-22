"""Vercel serverless entrypoint.

Vercel's Python runtime looks for a callable named ``app`` (or ``handler``)
in ``api/index.py``.  We simply re-export the FastAPI application from the
main package so that Vercel can serve it.
"""

import sys
import os
from pathlib import Path

# Make the repo root importable so ``from app.main import app`` works.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present (won't override real env vars set in Vercel dashboard)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from app.main import app  # noqa: F401  – Vercel picks this up as the ASGI app
