# TA-CDSS — Temporal-Aware Clinical Decision Support System

A prototype ICU decision support tool that combines a GRU temporal model, hybrid PubMed RAG, safety rule checking, and LLM-generated clinical assessment into a Streamlit web interface.

---

## Overview

TA-CDSS analyses a patient's ICU time-series data and produces a structured clinical assessment covering risk level, trajectory, evidence-backed reasoning, and recommended actions. It is designed for ICU clinicians reviewing ventilated or septic patients.

The pipeline runs in eight sequential steps:

```
eICU data
   └─▶ [1] GRU temporal engine        — risk score, trajectory, 11-feature TCSV
   └─▶ [2] Safety rule check          — 9 hard clinical rules, pre-LLM gate
   └─▶ [3] Hybrid PubMed RAG          — BM25 + cosine, top-5 articles + LLM insights
   └─▶ [4] Primary diagnosis agent    — LM Studio, structured assessment
   └─▶ [5] Confidence scoring         — data richness × sequence length
   └─▶ [6] 3C3H validation agent      — LM Studio, consistency + safety override
   └─▶ [7] Clinical explainer         — Gemini, delta-based WHY NOW narrative
   └─▶ [8] Final report (optional)    — Gemini, full structured report
```

**LLM routing:**

| Stage | Backend |
|---|---|
| GRU, safety rules, BM25/cosine retrieval | Local (no LLM) |
| RAG insight extraction, primary diagnosis, 3C3H validation | LM Studio (local) |
| Clinical explanation, final report | Gemini API |

---

## Requirements

**Python:** 3.10 or later

**Dependencies:**

```
streamlit
torch
numpy
pandas
requests
datasets
```

Install with:

```bash
pip install streamlit torch numpy pandas requests datasets
```

**LM Studio** (required for steps 4–6):
- Download from [lmstudio.ai](https://lmstudio.ai)
- Load a model (tested with Qwen 2.5 7B, DeepSeek R1 7B, Mistral 7B, LLaMA 3.2 3B)
- Go to the **Local Server** tab and click **Start Server** — must be running on `localhost:1234`

**Gemini API** (required for steps 7–8):
- Set your key in `config.py` under `GEMINI_API_KEY`
- Free tier: 15 RPM / 1500 RPD (sufficient for normal use)

---

## Data

The app expects eICU Collaborative Research Database CSVs in `data/eicu/`:

```
data/eicu/
├── patient.csv
├── vitalPeriodic.csv
├── lab.csv
├── respiratoryCharting.csv
└── treatment.csv
```

Access the eICU dataset at [physionet.org/content/eicu-crd](https://physionet.org/content/eicu-crd/). Credentialed access is required.

---

## Running the app

```bash
# from the project root
streamlit run app.py
```

Or run the pipeline directly from the command line:

```bash
python main.py
python main.py --patient_id 141168
```

---

## Project structure

```
ta_cdss/
├── app.py                    — Streamlit UI (8-step pipeline, landing page, report view)
├── main.py                   — CLI entry point
├── config.py                 — All thresholds, feature names, API keys, LLM routing
├── diagnostics.py            — Component health checks (GRU, RAG, LLM, data)
├── metrics_evaluation.py     — Offline metrics runner (BM25, trend, validation)
├── quality_control.py        — Pipeline audit trail, blocked-result builder
└── modules/
    ├── gru_temporal.py       — TimeAwareGRU model, EICUDataLoader, phase analysis
    ├── hybrid_rag.py         — PubMed search, BM25, cosine scoring, insight extraction
    ├── agentic_validation.py — Safety rules, primary agent, 3C3H validation agent
    ├── clinical_explainer.py — Delta WHY NOW, sparklines, Gemini narrative
    └── llm_client.py         — LM Studio + Gemini routing, rate limiting
```

---

## Configuration

All key settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `LM_STUDIO_URL` | `http://localhost:1234/v1/chat/completions` | LM Studio server endpoint |
| `LM_STUDIO_TIMEOUT` | 300s | Timeout for LM Studio calls |
| `GEMINI_API_KEY` | — | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for explanation + report |
| `RISK_HIGH_THRESHOLD` | 0.70 | Risk score above which HIGH label is applied |
| `RISK_MEDIUM_THRESHOLD` | 0.40 | Risk score above which MEDIUM label is applied |
| `TEMPORAL_WINDOW_HOURS` | 6 | Hours of history the GRU analyses |

Clinical thresholds for all 11 features and the 9 safety rules are also defined in `config.py` and are used by both the safety rule engine and the clinical explainer.

---

## Known limitations

**Metrics are not real evaluations.** `metrics_evaluation.py` runs against 37 hand-crafted test cases and rule-based scoring functions — it does not evaluate the GRU or LLM pipeline on actual eICU patients. The reported `validation_accuracy: 1.0` reflects self-consistency of the rules, not real performance. See `metrics_evaluation.py` for details.

**LM Studio must be started manually.** Opening the LM Studio app does not start the server. Go to the Local Server tab and click Start Server, and ensure a model is loaded there before running the pipeline.

**Gemini key is committed to config.py.** 
```python
import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
```

**No model weights are included.** `TimeAwareGRU` is initialised with random weights at runtime since no pre-trained checkpoint is saved or loaded. The model has never been trained on the eICU dataset. Risk scores are computed from the rule-based scoring functions in `estimate_risk_score`, not from learned GRU representations.

---

## Diagnostics

Run the built-in health check to verify all components are connected before a session:

```bash
python diagnostics.py
```

This checks the eICU data directory, LM Studio connectivity, Gemini API quota, GRU data loading, and RAG retrieval for a sample patient.

---

## Project context

TA-CDSS is a clinical prototype. It is not validated for clinical use and should not be used to make real patient management decisions. The system is intended for studying the integration of temporal deep learning, retrieval-augmented generation, and LLM-based clinical reasoning in a controlled clinical development setting.
