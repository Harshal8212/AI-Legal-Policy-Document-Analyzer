"""
Unit tests for the LegalClauseSplitter regex-based document splitter.

Validates that legal text is correctly parsed into semantic clauses
based on clause numbering patterns (1.1, ARTICLE I, SECTION 5, etc.).
"""

import pytest
from src.ingestion.legal_splitter import LegalClauseSplitter


class TestSplitNumberedClauses:
    def test_split_numbered_clauses(self):
        text = (
            '1.1 Definitions\n"Agreement" means this document.\n\n'
            "1.2 Term\nThe term of this agreement is 1 year."
        )
        splitter = LegalClauseSplitter()
        docs = splitter.create_documents([text])
        non_empty = [d for d in docs if d.page_content.strip()]
        assert len(non_empty) == 2
        assert non_empty[0].metadata["clause_id"] == "1.1"
        assert non_empty[1].metadata["clause_id"] == "1.2"

    def test_split_deeply_nested_numbers(self):
        """Patterns like 1.2.3 should also be recognised as clause boundaries."""
        text = (
            "1.2.3 Obligations\nThe vendor must deliver on time.\n\n"
            "1.2.4 Payment\nPayment is due within 30 days."
        )
        splitter = LegalClauseSplitter()
        docs = splitter.create_documents([text])
        non_empty = [d for d in docs if d.page_content.strip()]
        assert len(non_empty) == 2
        assert non_empty[0].metadata["clause_id"] == "1.2.3"


class TestSplitArticles:
    def test_split_articles(self):
        text = (
            "ARTICLE I: INTRO\nHere is the intro.\n\n"
            "ARTICLE II: TERMS\nHere are the terms."
        )
        splitter = LegalClauseSplitter()
        docs = splitter.create_documents([text])
        non_empty = [d for d in docs if d.page_content.strip()]
        assert len(non_empty) == 2
        assert "ARTICLE I" in non_empty[0].metadata["clause_id"]

    def test_split_roman_numeral_articles(self):
        """ARTICLE IV etc. should be split correctly."""
        text = (
            "ARTICLE IV: LIMITATIONS\nLimited to $1M.\n\n"
            "ARTICLE V: TERMINATION\nEither party may terminate."
        )
        splitter = LegalClauseSplitter()
        docs = splitter.create_documents([text])
        non_empty = [d for d in docs if d.page_content.strip()]
        assert len(non_empty) == 2


class TestEdgeCases:
    def test_intro_fallback(self):
        """Text without clause markers should get 'Intro' as clause_id."""
        text = "This is a preamble with no clause markers."
        splitter = LegalClauseSplitter()
        docs = splitter.create_documents([text])
        assert len(docs) == 1
        assert docs[0].metadata["clause_id"] == "Intro"

    def test_empty_text(self):
        """Empty text should produce zero documents."""
        splitter = LegalClauseSplitter()
        docs = splitter.create_documents([""])
        assert len(docs) == 0

    def test_whitespace_only_text(self):
        """Whitespace-only text should produce zero documents."""
        splitter = LegalClauseSplitter()
        docs = splitter.create_documents(["   \n\n   "])
        assert len(docs) == 0


class TestMetadata:
    def test_metadata_preserved(self):
        """Custom metadata passed to create_documents should be preserved."""
        text = "1.1 Term\nThe term is 1 year."
        splitter = LegalClauseSplitter()
        docs = splitter.create_documents([text], metadatas=[{"source": "test.txt"}])
        assert docs[0].metadata["source"] == "test.txt"
        assert docs[0].metadata["clause_id"] == "1.1"

    def test_multiple_texts_with_metadata(self):
        """Each text should receive its corresponding metadata."""
        texts = [
            "1.1 First\nContent.",
            "2.1 Second\nMore content.",
        ]
        metas = [{"source": "a.txt"}, {"source": "b.txt"}]
        splitter = LegalClauseSplitter()
        docs = splitter.create_documents(texts, metadatas=metas)
        sources = [d.metadata["source"] for d in docs]
        assert "a.txt" in sources
        assert "b.txt" in sources
