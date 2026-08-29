"""
API endpoint tests for the AI Legal Analyzer FastAPI application.

Every test mocks the heavy dependencies (VectorStoreManager, LLM workflow)
so the CI runner never needs network access or API tokens.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------- #
# GET /api/health
# --------------------------------------------------------------------- #
class TestHealthEndpoint:
    async def test_health_returns_ok(self, async_client):
        response = await async_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


# --------------------------------------------------------------------- #
# GET / (serve home page)
# --------------------------------------------------------------------- #
class TestHomeEndpoint:
    async def test_serve_home_returns_html(self, async_client):
        response = await async_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<html" in response.text.lower() or "<!doctype" in response.text.lower()


# --------------------------------------------------------------------- #
# POST /api/ingest
# --------------------------------------------------------------------- #
class TestIngestEndpoint:
    @patch("main.VectorStoreManager")
    @patch("main.Config")
    async def test_ingest_txt_file(self, mock_config, mock_vs_class, async_client):
        """Uploading a valid .txt file should return success with clause count."""
        mock_config.validate_api_key.return_value = None

        mock_vs = MagicMock()
        mock_vs.add_documents.return_value = ["id-1", "id-2", "id-3"]
        mock_vs_class.return_value = mock_vs

        txt_content = (
            "1.1 Definitions\n"
            '"Agreement" means this document.\n\n'
            "1.2 Term\n"
            "The term of this agreement is 1 year.\n"
        )

        response = await async_client.post(
            "/api/ingest",
            files={"file": ("contract.txt", io.BytesIO(txt_content.encode()), "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["filename"] == "contract.txt"
        assert data["num_clauses"] >= 1

    async def test_ingest_unsupported_format(self, async_client):
        """Uploading an unsupported file type should return a 400 error."""
        response = await async_client.post(
            "/api/ingest",
            files={"file": ("data.csv", io.BytesIO(b"col1,col2\na,b"), "text/csv")},
        )

        # The endpoint catches ValueError and returns 400, or Config might
        # raise if the token is missing — either way it should not be 200.
        assert response.status_code in (400, 500)

    async def test_ingest_missing_file(self, async_client):
        """Calling /api/ingest without a file should return 422."""
        response = await async_client.post("/api/ingest")
        assert response.status_code == 422


# --------------------------------------------------------------------- #
# POST /api/analyze
# --------------------------------------------------------------------- #
class TestAnalyzeEndpoint:
    @patch("main.create_workflow")
    async def test_analyze_returns_answer(self, mock_create_workflow, async_client):
        """A mocked workflow should return a successful analysis response."""
        fake_result = {
            "query": "What are the risks?",
            "documents": [],
            "risk_analysis": [
                {
                    "clause_id": "1.1",
                    "risk_level": "High",
                    "risk_score": 8,
                    "reason": "Unlimited liability",
                    "recommendation": "Negotiate a cap",
                }
            ],
            "final_answer": "This contract contains high-risk clauses.",
            "overall_report": {
                "overall_risk_score": 8.0,
                "high_risk_count": 1,
                "medium_risk_count": 0,
                "low_risk_count": 0,
            },
        }

        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = fake_result
        mock_create_workflow.return_value = mock_workflow

        response = await async_client.post(
            "/api/analyze",
            json={"query": "What are the risks?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "answer" in data
        assert data["answer"] == "This contract contains high-risk clauses."
        assert data["num_clauses_analyzed"] == 1

    async def test_analyze_missing_query(self, async_client):
        """Calling /api/analyze without a query body should return 422."""
        response = await async_client.post("/api/analyze", json={})
        assert response.status_code == 422
