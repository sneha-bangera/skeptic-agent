<!-- # The Skeptic's Brief

A production-grade multi-agent AI system that answers questions from a corpus of mixed-quality sources by **actively challenging its own conclusions** before finalizing them.

Most AI systems are overconfident. The Skeptic's Brief earns its confidence level by stress-testing its own reasoning through an adversarial Skeptic agent that probes every conclusion for population mismatch, commercial bias, outcome mismatch, sample size issues, recency gaps, missing counter-evidence, and ignored conflicts.

---

## Architecture Overview

```
Question + Sources
        │
        ▼
┌─────────────────────┐
│   Input Guardrail   │  Domain flag, PII check, schema validate
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Ingress Agent     │  Validate, deduplicate, assign stable IDs
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  Source Scorer Agent                     │
│  Denylist Check → Allowlist Check → Score + Classify    │
│  Credibility (0–1) · Relevance (0–1) · Bias flag        │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Evidence Extractor  │  Extract atomic falsifiable claims per source
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Contradiction Detect│  Cosine similarity → LLM judge → conflict types
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Source Count Guard  │  < 3 usable sources → cap confidence at Low
└─────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│              Adversarial Loop (max 3 rounds)              │
│                                                          │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────┐  │
│  │  Synthesis   │───▶│    Skeptic    │───▶│ Arbiter  │  │
│  │  Agent       │    │    Agent      │    │          │  │
│  └──────────────┘    └───────────────┘    └──────────┘  │
│         ▲                                      │         │
│         └──────────── revise ──────────────────┘         │
│                                              │ finalize  │
└──────────────────────────────────────────────┼───────────┘
                                               │
        ▼
┌─────────────────────┐
│ Confidence Floor    │  Force Insufficient Evidence after 3 unresolved rounds
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   HITL Review Gate  │  Triggered below Moderate confidence
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Egress Agent      │  Format output + domain disclaimer
└─────────────────────┘
        │
        ▼
┌─────────────────────┐  ┌─────────────────────┐
│   Final Response    │  │    Audit Agent       │  Async logs + replay
└─────────────────────┘  └─────────────────────┘
        │
        ▼
Async Feedback Writeback: User Rating · Source Corrections
```

**Confidence tiers:** `High` | `Moderate` | `Low` | `Insufficient Evidence`

---

## Quick Start

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager
- A [Groq API key](https://console.groq.com) (free tier available)

### Setup

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Clone the repo and enter the project directory
git clone <repo-url>
cd skeptics_brief

# 3. Initialize the project with uv
uv init

# 4. Create virtual environment
uv venv

# 5. Activate the virtual environment (Windows)
.venv\Scripts\activate

# On macOS/Linux:
# source .venv/bin/activate

# 6. Install all dependencies
uv add -r requirements.txt

# 7. Configure environment
cp .env.example .env
# Edit .env and set your GROQ_API_KEY

# 8. Run the API server
python main.py
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## API Usage

### Analyze a question with sources

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Does daily aspirin reduce cardiovascular events in healthy adults over 60?",
    "sources": [
      {
        "url": "https://www.nejm.org/doi/full/10.1056/NEJMoa1805819",
        "content": "The ASPREE trial enrolled 19,114 healthy older adults... aspirin did not significantly reduce the risk of cardiovascular disease...",
        "metadata": {
          "title": "Effect of Aspirin on Cardiovascular Events and Bleeding in the Healthy Elderly",
          "doi": "10.1056/NEJMoa1805819",
          "published_date": "2018-09-16",
          "citation_count": 1847
        }
      },
      {
        "url": "https://www.aspirin-foundation.com/benefits",
        "content": "Aspirin has been shown to dramatically reduce heart attacks in all adults...",
        "metadata": {
          "title": "Aspirin Benefits",
          "author": "Aspirin Foundation"
        }
      }
    ]
  }'
```

**Example response:**

```json
{
  "request_id": "req_abc123",
  "question": "Does daily aspirin reduce cardiovascular events in healthy adults over 60?",
  "answer": "Based on the ASPREE trial (src_001, credibility: 0.94), daily aspirin did not significantly reduce cardiovascular events in healthy adults aged 70+ and was associated with increased bleeding risk. This finding contradicts earlier general recommendations based on higher-risk populations.",
  "confidence": "Moderate",
  "citations": [
    {
      "claim_text": "Aspirin did not significantly reduce cardiovascular disease risk in healthy elderly",
      "source_id": "src_001",
      "credibility": 0.94,
      "relevance": 0.91,
      "source_url": "https://www.nejm.org/doi/full/10.1056/NEJMoa1805819"
    }
  ],
  "unresolved_contradictions": [
    "src_002 (Aspirin Foundation) claims general cardiovascular benefit for all adults, contradicting ASPREE RCT evidence — src_002 dropped due to commercial bias (credibility: 0.18)"
  ],
  "disclaimer": null,
  "sources_used": 1,
  "sources_dropped": 1,
  "loop_rounds": 1,
  "skeptic_challenges": [
    {
      "challenge_id": "c_001",
      "type": "wrong_population",
      "description": "src_002 makes claims about 'all adults' but ASPREE specifically studied healthy adults 70+. The preliminary answer conflated these populations.",
      "severity": "critical",
      "was_resolved": true,
      "arbiter_ruling": "valid_critical"
    }
  ],
  "hitl_status": "not_required",
  "status": "complete"
}
```

### Check pipeline status

```bash
curl http://localhost:8000/status/req_abc123
```

### Submit feedback

```bash
curl -X POST http://localhost:8000/feedback/req_abc123 \
  -H "Content-Type: application/json" \
  -d '{
    "rating": 4,
    "source_corrections": [
      {"source_id": "src_002", "flag": "commercial_bias"}
    ]
  }'
```

### Approve a HITL-flagged response

```bash
curl -X POST http://localhost:8000/approve/req_abc123 \
  -H "Content-Type: application/json" \
  -d '{
    "approved": true,
    "reviewer_notes": "Answer is appropriately hedged. Confidence upgrade to Moderate warranted."
  }'
```

### Replay an audit trail

```bash
curl http://localhost:8000/replay/req_abc123
```

---

## Configuration Reference

All settings are configurable via `.env` or environment variables:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | LLM model identifier |
| `MAX_LOOP_ROUNDS` | `3` | Max adversarial loop rounds |
| `MIN_CREDIBILITY_THRESHOLD` | `0.20` | Min credibility for evidence extraction |
| `MIN_RELEVANCE_THRESHOLD` | `0.30` | Min relevance for evidence extraction |
| `SOURCE_COUNT_FLOOR` | `3` | Below this, confidence capped at Low |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CONTRADICTION_SIMILARITY_THRESHOLD` | `0.70` | Cosine sim threshold for contradiction check |
| `HITL_CONFIDENCE_THRESHOLD` | `Moderate` | Confidence below which HITL triggers |
| `AUDIT_DB_PATH` | `storage/audit.db` | SQLite audit database path |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DRY_RUN` | `false` | Skip real LLM calls (for testing) |

---

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific phase
pytest tests/test_ingress.py -v
pytest tests/test_source_scorer.py -v
pytest tests/test_skeptic.py -v
pytest tests/test_arbiter.py -v
pytest tests/test_pipeline_e2e.py -v

# Run with dry-run mode (no LLM calls needed)
DRY_RUN=true pytest -v
```

---

## Project Structure

```
skeptics_brief/
├── .env.example              # API keys template
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                   # FastAPI app entrypoint
├── config/
│   └── settings.py           # Pydantic settings
├── core/
│   ├── state.py              # LangGraph PipelineState TypedDict
│   ├── graph.py              # LangGraph graph builder
│   └── exceptions.py         # Custom exception hierarchy
├── agents/
│   ├── ingress.py            # Validate, deduplicate, assign IDs
│   ├── source_scorer.py      # Credibility · Relevance · Bias scoring
│   ├── evidence_extractor.py # Extract atomic claims from sources
│   ├── contradiction_detector.py
│   ├── synthesis.py          # Build preliminary answer
│   ├── skeptic.py            # Adversarial challenger (the heart)
│   ├── arbiter.py            # Resolve challenges, control loop
│   ├── egress.py             # Format final output
│   └── audit.py              # Async post-decision logging
├── guardrails/
│   ├── input_guardrail.py    # Domain flag, PII, schema validate
│   ├── source_count_guardrail.py
│   ├── confidence_floor.py
│   └── output_disclaimer.py
├── registry/
│   ├── allowlist.py          # Trusted domains + DOI prefixes
│   └── denylist.py           # Known bad domains
├── schemas/
│   ├── request.py            # API input schema
│   ├── response.py           # API output schema
│   ├── source.py             # Source + ScoredSource schemas
│   └── claim.py              # Claim schema
├── services/
│   ├── llm.py                # Groq LLM gateway (all calls go here)
│   ├── embedder.py           # Sentence-transformer wrapper
│   └── similarity.py         # Cosine similarity helpers
├── storage/
│   └── audit_db.py           # aiosqlite audit log writer
├── api/
│   ├── routes.py             # FastAPI route definitions
│   └── dependencies.py       # FastAPI dependency injection
└── tests/
    ├── test_ingress.py
    ├── test_source_scorer.py
    ├── test_skeptic.py
    ├── test_arbiter.py
    └── test_pipeline_e2e.py
```

---

## Key Design Decisions

1. **All LLM calls go through `services/llm.py`** — never directly. Handles retry, logging, dry-run mode.
2. **Loop termination is code-level** — `loop_round >= 3` is a Python `if` statement in the graph's conditional edge, not an LLM prompt. LLMs cannot reliably count to 3.
3. **Confidence is conservative** — when between two tiers, always take the lower one.
4. **`Insufficient Evidence` is a first-class output** — not a fallback. The system outputs it clearly with specific reasons.
5. **Agent failures are isolated** — each agent catches its own exceptions, writes to `audit_trail`, and lets the pipeline continue. A failed scorer means that source is dropped, not that the request fails.
6. **State is append-only** — lists in `PipelineState` are never mutated; new records are always appended. This makes the audit trail reliable.
7. **Skeptic must be specific** — its prompt explicitly requires source IDs and concrete descriptions. Vague challenges are rejected by the Arbiter.

---

## Confidence Tier Guide

| Tier | When assigned |
|---|---|
| **High** | ≥ 2 peer-reviewed sources, credibility > 0.80, relevance > 0.80, no irreconcilable conflicts |
| **Moderate** | 1 peer-reviewed OR 2+ institutional sources, no irreconcilable conflicts |
| **Low** | Only blog/news sources, OR irreconcilable conflicts present, OR relevance < 0.50 |
| **Insufficient Evidence** | < 2 usable sources, OR all sources credibility < 0.30, OR 3 unresolved critical challenges | -->















# The Skeptic's Brief

A production-grade multi-agent AI system that answers questions from a corpus of mixed-quality sources by **actively challenging its own conclusions** before finalizing them.

Most AI systems are overconfident. The Skeptic's Brief earns its confidence level by stress-testing its own reasoning through an adversarial Skeptic agent that probes every conclusion for population mismatch, commercial bias, outcome mismatch, sample size issues, recency gaps, missing counter-evidence, and ignored conflicts.

---

## Architecture Overview

```
Question + Sources
        │
        ▼
┌─────────────────────┐
│   Input Guardrail   │  Domain flag, PII check, schema validate
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Ingress Agent     │  Validate, deduplicate, assign stable IDs
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  Source Scorer Agent                     │
│  Denylist Check → Allowlist Check → Score + Classify    │
│  Credibility (0–1) · Relevance (0–1) · Bias flag        │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Evidence Extractor  │  Extract atomic falsifiable claims per source
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Contradiction Detect│  Cosine similarity → LLM judge → conflict types
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Source Count Guard  │  < 3 usable sources → cap confidence at Low
└─────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│              Adversarial Loop (max 3 rounds)              │
│                                                          │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────┐  │
│  │  Synthesis   │───▶│    Skeptic    │───▶│ Arbiter  │  │
│  │  Agent       │    │    Agent      │    │          │  │
│  └──────────────┘    └───────────────┘    └──────────┘  │
│         ▲                                      │         │
│         └──────────── revise ──────────────────┘         │
│                                              │ finalize  │
└──────────────────────────────────────────────┼───────────┘
                                               │
        ▼
┌─────────────────────┐
│ Confidence Floor    │  Force Insufficient Evidence after 3 unresolved rounds
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   HITL Review Gate  │  Triggered below Moderate confidence
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Egress Agent      │  Format output + domain disclaimer
└─────────────────────┘
        │
        ▼
┌─────────────────────┐  ┌─────────────────────┐
│   Final Response    │  │    Audit Agent       │  Async logs + replay
└─────────────────────┘  └─────────────────────┘
        │
        ▼
Async Feedback Writeback: User Rating · Source Corrections
```

**Confidence tiers:** `High` | `Moderate` | `Low` | `Insufficient Evidence`

---

## Quick Start

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager
- A [Groq API key](https://console.groq.com) (free tier available)

### Setup

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Clone the repo and enter the project directory
git clone <repo-url>
cd skeptics_brief

# 3. Initialize the project with uv
uv init

# 4. Create virtual environment
uv venv

# 5. Activate the virtual environment (Windows)
.venv\Scripts\activate

# On macOS/Linux:
# source .venv/bin/activate

# 6. Install all dependencies
uv add -r requirements.txt

# 7. Configure environment
cp .env.example .env
# Edit .env and set your GROQ_API_KEY

# 8. Run the API server
python main.py
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## API Usage

### Analyze a question with sources

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Does daily aspirin reduce cardiovascular events in healthy adults over 60?",
    "sources": [
      {
        "url": "https://www.nejm.org/doi/full/10.1056/NEJMoa1805819",
        "content": "The ASPREE trial enrolled 19,114 healthy older adults... aspirin did not significantly reduce the risk of cardiovascular disease...",
        "metadata": {
          "title": "Effect of Aspirin on Cardiovascular Events and Bleeding in the Healthy Elderly",
          "doi": "10.1056/NEJMoa1805819",
          "published_date": "2018-09-16",
          "citation_count": 1847
        }
      },
      {
        "url": "https://www.aspirin-foundation.com/benefits",
        "content": "Aspirin has been shown to dramatically reduce heart attacks in all adults...",
        "metadata": {
          "title": "Aspirin Benefits",
          "author": "Aspirin Foundation"
        }
      }
    ]
  }'
```

**Example response:**

```json
{
  "request_id": "req_abc123",
  "question": "Does daily aspirin reduce cardiovascular events in healthy adults over 60?",
  "answer": "Based on the ASPREE trial (src_001, credibility: 0.94), daily aspirin did not significantly reduce cardiovascular events in healthy adults aged 70+ and was associated with increased bleeding risk. This finding contradicts earlier general recommendations based on higher-risk populations.",
  "confidence": "Moderate",
  "citations": [
    {
      "claim_text": "Aspirin did not significantly reduce cardiovascular disease risk in healthy elderly",
      "source_id": "src_001",
      "credibility": 0.94,
      "relevance": 0.91,
      "source_url": "https://www.nejm.org/doi/full/10.1056/NEJMoa1805819"
    }
  ],
  "unresolved_contradictions": [
    "src_002 (Aspirin Foundation) claims general cardiovascular benefit for all adults, contradicting ASPREE RCT evidence — src_002 dropped due to commercial bias (credibility: 0.18)"
  ],
  "disclaimer": null,
  "sources_used": 1,
  "sources_dropped": 1,
  "loop_rounds": 1,
  "skeptic_challenges": [
    {
      "challenge_id": "c_001",
      "type": "wrong_population",
      "description": "src_002 makes claims about 'all adults' but ASPREE specifically studied healthy adults 70+. The preliminary answer conflated these populations.",
      "severity": "critical",
      "was_resolved": true,
      "arbiter_ruling": "valid_critical"
    }
  ],
  "hitl_status": "not_required",
  "status": "complete"
}
```

### Check pipeline status

```bash
curl http://localhost:8000/status/req_abc123
```

### Submit feedback

```bash
curl -X POST http://localhost:8000/feedback/req_abc123 \
  -H "Content-Type: application/json" \
  -d '{
    "rating": 4,
    "source_corrections": [
      {"source_id": "src_002", "flag": "commercial_bias"}
    ]
  }'
```

### Approve a HITL-flagged response

```bash
curl -X POST http://localhost:8000/approve/req_abc123 \
  -H "Content-Type: application/json" \
  -d '{
    "approved": true,
    "reviewer_notes": "Answer is appropriately hedged. Confidence upgrade to Moderate warranted."
  }'
```

### Replay an audit trail

```bash
curl http://localhost:8000/replay/req_abc123
```

---

## Configuration Reference

All settings are configurable via `.env` or environment variables:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | LLM model identifier |
| `MAX_LOOP_ROUNDS` | `3` | Max adversarial loop rounds |
| `MIN_CREDIBILITY_THRESHOLD` | `0.20` | Min credibility for evidence extraction |
| `MIN_RELEVANCE_THRESHOLD` | `0.30` | Min relevance for evidence extraction |
| `SOURCE_COUNT_FLOOR` | `3` | Below this, confidence capped at Low |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CONTRADICTION_SIMILARITY_THRESHOLD` | `0.70` | Cosine sim threshold for contradiction check |
| `HITL_CONFIDENCE_THRESHOLD` | `Moderate` | Confidence below which HITL triggers |
| `AUDIT_DB_PATH` | `storage/audit.db` | SQLite audit database path |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DRY_RUN` | `false` | Skip real LLM calls (for testing) |

---

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific phase
pytest tests/test_ingress.py -v
pytest tests/test_source_scorer.py -v
pytest tests/test_skeptic.py -v
pytest tests/test_arbiter.py -v
pytest tests/test_pipeline_e2e.py -v

# Run with dry-run mode (no LLM calls needed)
DRY_RUN=true pytest -v
```

---

## Project Structure

```
skeptics_brief/
├── .env.example              # API keys template
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                   # FastAPI app entrypoint
├── config/
│   └── settings.py           # Pydantic settings
├── core/
│   ├── state.py              # LangGraph PipelineState TypedDict
│   ├── graph.py              # LangGraph graph builder
│   └── exceptions.py         # Custom exception hierarchy
├── agents/
│   ├── ingress.py            # Validate, deduplicate, assign IDs
│   ├── source_scorer.py      # Credibility · Relevance · Bias scoring
│   ├── evidence_extractor.py # Extract atomic claims from sources
│   ├── contradiction_detector.py
│   ├── synthesis.py          # Build preliminary answer
│   ├── skeptic.py            # Adversarial challenger (the heart)
│   ├── arbiter.py            # Resolve challenges, control loop
│   ├── egress.py             # Format final output
│   └── audit.py              # Async post-decision logging
├── guardrails/
│   ├── input_guardrail.py    # Domain flag, PII, schema validate
│   ├── source_count_guardrail.py
│   ├── confidence_floor.py
│   └── output_disclaimer.py
├── registry/
│   ├── allowlist.py          # Trusted domains + DOI prefixes
│   └── denylist.py           # Known bad domains
├── schemas/
│   ├── request.py            # API input schema
│   ├── response.py           # API output schema
│   ├── source.py             # Source + ScoredSource schemas
│   └── claim.py              # Claim schema
├── services/
│   ├── llm.py                # Groq LLM gateway (all calls go here)
│   ├── embedder.py           # Sentence-transformer wrapper
│   └── similarity.py         # Cosine similarity helpers
├── storage/
│   └── audit_db.py           # aiosqlite audit log writer
├── api/
│   ├── routes.py             # FastAPI route definitions
│   └── dependencies.py       # FastAPI dependency injection
└── tests/
    ├── test_ingress.py
    ├── test_source_scorer.py
    ├── test_skeptic.py
    ├── test_arbiter.py
    └── test_pipeline_e2e.py
```

---

## Key Design Decisions

1. **All LLM calls go through `services/llm.py`** — never directly. Handles retry, logging, dry-run mode.
2. **Loop termination is code-level** — `loop_round >= 3` is a Python `if` statement in the graph's conditional edge, not an LLM prompt. LLMs cannot reliably count to 3.
3. **Confidence is conservative** — when between two tiers, always take the lower one.
4. **`Insufficient Evidence` is a first-class output** — not a fallback. The system outputs it clearly with specific reasons.
5. **Agent failures are isolated** — each agent catches its own exceptions, writes to `audit_trail`, and lets the pipeline continue. A failed scorer means that source is dropped, not that the request fails.
6. **State is append-only** — lists in `PipelineState` are never mutated; new records are always appended. This makes the audit trail reliable.
7. **Skeptic must be specific** — its prompt explicitly requires source IDs and concrete descriptions. Vague challenges are rejected by the Arbiter.

---

## Confidence Tier Guide

| Tier | When assigned |
|---|---|
| **High** | ≥ 2 peer-reviewed sources, credibility > 0.80, relevance > 0.80, no irreconcilable conflicts |
| **Moderate** | 1 peer-reviewed OR 2+ institutional sources, no irreconcilable conflicts |
| **Low** | Only blog/news sources, OR irreconcilable conflicts present, OR relevance < 0.50 |
| **Insufficient Evidence** | < 2 usable sources, OR all sources credibility < 0.30, OR 3 unresolved critical challenges |