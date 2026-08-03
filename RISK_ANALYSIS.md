# Risk Analysis Component Details

The Risk Analysis component is a core subsystem of the AI Legal Policy & Document Analyzer. It operates under a **hybrid evaluation strategy** that combines the contextual understanding of a Large Language Model (LLM) with a deterministic, keyword-based rule engine. This ensures robust, consistent, and accurate risk assessments.

## 1. Subsystem Architecture

The risk engine is encapsulated within the `src/risk_engine/` directory and consists of three main modules:
- **`risk_models.py`**: Defines strictly typed data structures using Pydantic.
- **`risk_rules.py`**: A deterministic keyword-based evaluation engine.
- **`risk_scorer.py`**: The primary orchestrator handling LLM prompts, batch processing, and score combination.

## 2. Deterministic Rule Engine (`RiskRuleEngine`)

To counter potential LLM hallucinations and ensure critical legal phrases are never missed, the `RiskRuleEngine` acts as a fail-safe layer.

### How it Works
The engine scans the raw text of each clause (case-insensitively) against a predefined set of high-impact keyword combinations. If a match is found, it applies a fixed score modifier.

**Sample Rules:**
| Keywords Trigger | Modifier | Risk Context |
| :--- | :---: | :--- |
| `["indemnify", "unlimited"]` | **+7** | **High Risk:** Unlimited indemnity clauses expose the user to unbounded financial liability. |
| `["termination", "convenience"]` | **+3** | **Medium/High Risk:** The contract can be terminated without cause. |
| `["auto-renew"]` | **+2** | **Medium Risk:** Unintended subscription renewals. |
| `["no warranty", "as is"]` | **+2** | **Medium Risk:** Disclaims liability for defects. |
| `["liability", "cap"]` | **-2** | **Risk Mitigator:** Reduces legal exposure by capping damages. |
| `["governing law"]` | **-1** | **Risk Mitigator:** Establishes jurisdictional clarity. |

*Output:* The engine returns a tuple containing the `total_score_modifier` and a list of `triggered_labels` (e.g., `"unlimited indemnity"`).

## 3. LLM Scorer (`RiskScorer`)

The `RiskScorer` is responsible for semantic interpretation, utilizing the `Qwen/Qwen2.5-7B-Instruct` model via the Hugging Face Inference API.

### 3.1 Batch Processing Strategy
To conserve API quota and reduce overall latency, the `RiskScorer` prefers **Batch Processing**:
1. **Concatenation:** Retrieves all clauses relevant to the user query and concatenates them into a single, structured prompt.
2. **JSON Array Request:** Prompts the LLM to analyze all clauses simultaneously and return a strictly formatted JSON array containing the analysis for each clause.
3. **Resilience (Fallback Mechanism):** If the LLM returns invalid JSON or hallucinates formatting (e.g., Markdown blocks), the `RiskScorer` automatically falls back to analyzing each clause individually using concurrent requests (`asyncio.gather`).

### 3.2 Evaluation Criteria
For each clause, the LLM is instructed to determine:
- **Purpose:** A plain English summary of the clause.
- **Base Risk Score:** An initial score between 1 and 10 (10 being the highest risk).
- **Explanation:** Why this score was assigned.
- **Recommendation:** Actionable steps to mitigate the identified risk.

## 4. Score Combination Algorithm

Once the LLM provides its base score, the system calculates the final risk profile:

1. **Aggregation:** `Base_Score (LLM) + Modifier (Rule Engine)`
2. **Clamping:** The aggregated score is clamped to ensure it stays within valid bounds: 
   ```python
   final_score = max(1, min(10, base_score + modifier))
   ```
3. **Level Categorization:**
   - **High Risk:** Final Score $\ge$ 8
   - **Medium Risk:** Final Score 5–7
   - **Low Risk:** Final Score $\le$ 4
4. **Context Enrichment:** If the `RiskRuleEngine` triggered any rules, their descriptive labels are appended to the LLM's explanation (e.g., `"... | Rules triggered: unlimited indemnity"`).

## 5. Output Data Models (`RiskClause` & `RiskReport`)

The component guarantees structured outputs using **Pydantic**, ensuring seamless integration with the LangGraph pipeline and the frontend UI.

### `RiskClause` (Single Clause Analysis)
```python
class RiskClause(BaseModel):
    clause_id: str        # e.g., '1.1', 'Section 5'
    clause_type: str      # e.g., 'Indemnity', 'Liability Cap'
    risk_level: str       # 'High', 'Medium', or 'Low'
    risk_score: int       # Clamped value between 1 and 10
    reason: str           # LLM explanation + triggered rules
    recommendation: str   # Mitigation strategy
```

### `RiskReport` (Document-Level Aggregation)
```python
class RiskReport(BaseModel):
    document_id: str
    overall_risk_score: float
    high_risk_clauses: List[RiskClause]
    medium_risk_clauses: List[RiskClause]
    low_risk_clauses: List[RiskClause]
```
