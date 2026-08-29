"""
Unit tests for the risk engine components:
- RiskRuleEngine  (deterministic keyword-based scoring)
- RiskClause      (Pydantic model validation)
"""

import pytest
from pydantic import ValidationError

from src.risk_engine.risk_rules import RiskRuleEngine
from src.risk_engine.risk_models import RiskClause, RiskReport


# --------------------------------------------------------------------- #
# RiskRuleEngine
# --------------------------------------------------------------------- #
class TestRiskRuleEngine:
    @pytest.fixture
    def engine(self):
        return RiskRuleEngine()

    def test_unlimited_indemnity_rule(self, engine):
        """Text containing 'indemnify' and 'unlimited' should trigger +7."""
        text = "The vendor shall indemnify the client for unlimited damages."
        modifier, triggered = engine.evaluate(text)
        assert "unlimited indemnity" in triggered
        assert modifier >= 7

    def test_liability_cap_rule(self, engine):
        """Text containing 'liability' and 'cap' should trigger -2."""
        text = "The total liability under this agreement is subject to a cap of $1M."
        modifier, triggered = engine.evaluate(text)
        assert "liability cap present" in triggered
        assert modifier <= -2

    def test_termination_for_convenience(self, engine):
        """Text containing 'termination' and 'convenience' should trigger +3."""
        text = "Either party may elect termination for convenience upon 30 days notice."
        modifier, triggered = engine.evaluate(text)
        assert "termination for convenience" in triggered
        assert modifier >= 3

    def test_auto_renewal_rule(self, engine):
        """Text containing 'auto-renew' should trigger +2."""
        text = "This agreement shall auto-renew for successive 1-year terms."
        modifier, triggered = engine.evaluate(text)
        assert "auto-renewal clause" in triggered
        assert modifier >= 2

    def test_governing_law_rule(self, engine):
        """Text containing 'governing law' should trigger -1."""
        text = "The governing law of this agreement shall be the State of Delaware."
        modifier, triggered = engine.evaluate(text)
        assert "governing law defined" in triggered
        assert modifier <= -1

    def test_no_rules_triggered(self, engine):
        """Benign text should trigger nothing and return zero modifier."""
        text = "The parties agree to meet quarterly to review progress."
        modifier, triggered = engine.evaluate(text)
        assert modifier == 0
        assert triggered == []

    def test_multiple_rules_triggered(self, engine):
        """Text matching several rules should accumulate all modifiers."""
        text = (
            "The vendor shall indemnify the client for unlimited damages. "
            "Either party may elect termination for convenience."
        )
        modifier, triggered = engine.evaluate(text)
        assert "unlimited indemnity" in triggered
        assert "termination for convenience" in triggered
        assert len(triggered) >= 2
        # +7 (indemnity) + 3 (termination) = 10
        assert modifier == 10

    def test_case_insensitive_matching(self, engine):
        """Rule matching should be case-insensitive."""
        text = "AUTO-RENEW is applicable to this contract."
        modifier, triggered = engine.evaluate(text)
        assert "auto-renewal clause" in triggered


# --------------------------------------------------------------------- #
# RiskClause Pydantic model
# --------------------------------------------------------------------- #
class TestRiskClauseModel:
    def test_valid_risk_clause(self):
        """A valid RiskClause should be created without errors."""
        clause = RiskClause(
            clause_id="1.1",
            clause_type="Indemnity",
            risk_level="High",
            risk_score=9,
            reason="Unlimited liability exposure.",
            recommendation="Negotiate a liability cap.",
        )
        assert clause.clause_id == "1.1"
        assert clause.risk_score == 9

    def test_risk_score_out_of_range(self):
        """Risk score outside 1-10 should raise ValidationError."""
        with pytest.raises(ValidationError):
            RiskClause(
                clause_id="1.1",
                clause_type="Indemnity",
                risk_level="High",
                risk_score=15,  # out of range
                reason="Test",
                recommendation="Test",
            )

    def test_risk_score_zero_invalid(self):
        """Risk score of 0 should raise ValidationError (minimum is 1)."""
        with pytest.raises(ValidationError):
            RiskClause(
                clause_id="1.1",
                clause_type="Term",
                risk_level="Low",
                risk_score=0,
                reason="Test",
                recommendation="Test",
            )

    def test_missing_required_field(self):
        """Omitting a required field should raise ValidationError."""
        with pytest.raises(ValidationError):
            RiskClause(
                clause_id="1.1",
                # clause_type is missing
                risk_level="Medium",
                risk_score=5,
                reason="Test",
                recommendation="Test",
            )

    def test_risk_report_model(self):
        """RiskReport should accept lists of RiskClause objects."""
        clause = RiskClause(
            clause_id="2.1",
            clause_type="Warranty",
            risk_level="Low",
            risk_score=2,
            reason="Standard warranty.",
            recommendation="No action needed.",
        )
        report = RiskReport(
            document_id="doc-001",
            overall_risk_score=2.0,
            high_risk_clauses=[],
            medium_risk_clauses=[],
            low_risk_clauses=[clause],
        )
        assert report.overall_risk_score == 2.0
        assert len(report.low_risk_clauses) == 1
