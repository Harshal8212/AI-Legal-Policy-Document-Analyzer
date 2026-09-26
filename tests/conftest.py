"""
Shared pytest fixtures for the AI Legal Analyzer test suite.

Provides mocked environment variables and an async HTTP client
wired to the FastAPI application so tests never hit external APIs.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
import httpx

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app  # noqa: E402


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Inject a fake HuggingFace token so Config.validate_api_key() passes."""
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_FAKE_TOKEN_FOR_CI_TESTING_1234567890")


@pytest_asyncio.fixture
async def async_client():
    """Yield an httpx.AsyncClient bound to the FastAPI app (no network)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
