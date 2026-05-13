"""
modules/quality_control.py  --  TA-CDSS Pipeline Quality Control Layer

Addresses five systemic weaknesses:

  1. RAG Instability
     - Deterministic query hashing + in-memory result cache
       (same patient context → same PubMed results within a session)
     - Hard per-article minimum relevance threshold (ARTICLE_MIN_SCORE)
     - Clinical coherence check: article must share at least one
       clinical concept with the patient's active findings

  2. LLM Non-determinism
     - Temperature recommendations enforced per call type
     - Self-consistency check: samples the primary assessment twice
       at low temperature and cross-validates the diagnosis section
     - Structured output validator: required sections present,
       min/max length enforced, diagnosis text extracted and checked

  3. Error Propagation
     - RAGQualityGate: blocks bad evidence from reaching the LLM
       and substitutes a clearly-labelled safe-fallback block
     - Propagation audit trail: each pipeline stage records its
       quality tier so downstream stages know the upstream quality

  4. No Quality Control Layer
     - OutputQualityGate: scores LLM output on 5 dimensions
       (structure, length, diagnosis-risk alignment, specificity,
       internal consistency) and returns PASS / WARN / FAIL
     - Failed outputs are replaced with a structured fallback
       rather than silently shown as valid

  5. Data Variability
     - PatientDataQualityAssessor: measures feature missingness,
       phase coverage, outlier rate, inter-phase variance
     - Returns DATA_TIER: RICH / ADEQUATE / SPARSE / UNRELIABLE
     - Pipeline adjusts confidence floor, LLM temperature, and
       output gating based on DATA_TIER
"""

from __future__ import annotations

import re
import math
import hashlib
import json
from collections import Counter
from datetime import datetime
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum per-article relevance score to pass into the LLM context.
# Articles below this are stripped even if they survive BM25+semantic ranking.
ARTICLE_MIN_SCORE = 0.15

# Minimum average score across top-K articles for RAG to be considered usable.
RAG_USABLE_THRESHOLD = 0.20

# Minimum number of required LLM output sections present for PASS.
REQUIRED_SECTIONS = ["DIAGNOSIS", "KEY FINDINGS", "INTERVENTIONS", "RISK FLAGS"]

# Minimum/maximum response length (chars) for a valid primary assessment.
MIN_RESPONSE_CHARS = 200
MAX_RESPONSE_CHARS = 8000

# Self-consistency: fraction of diagnosis tokens that must overlap between
# two independent samples for them to be considered consistent.
SELF_CONSISTENCY_OVERLAP_THRESHOLD = 0.35

# ── Hard data gate — absolute minimums before ANY pipeline runs ────────────────
# If either condition is not met the pipeline is blocked entirely.
# No LLM calls, no RAG, no risk score — only a structured refusal.
HARD_GATE_MIN_TIMESTEPS    = 10    # fewer than 10 observations → blocked
HARD_GATE_MIN_FEAT_COVERAGE = 0.50 # fewer than 50 % of 11 features populated → blocked

# Confidence caps per tier — prevents inflated confidence when data is poor
CONFIDENCE_CAP = {
    "RICH":       1.00,   # no cap
    "ADEQUATE":   0.75,   # genuine ceiling given incomplete data
    "SPARSE":     0.40,   # data is unreliable — hard cap at 40 %
    "UNRELIABLE": 0.20,   # almost no usable data
    "BLOCKED":    0.00,   # pipeline never ran
    "UNKNOWN":    0.40,   # conservative default
}

# Data quality tier thresholds
DATA_TIER_THRESHOLDS = {
    "RICH":       {"min_seq": 20, "min_phases": 3, "min_features": 7, "max_missing": 0.30},
    "ADEQUATE":   {"min_seq": 10, "min_phases": 2, "min_features": 4, "max_missing": 0.55},
    "SPARSE":     {"min_seq":  4, "min_phases": 1, "min_features": 2, "max_missing": 0.80},
    # Below SPARSE = UNRELIABLE
}

# Per-tier pipeline behaviour
DATA_TIER_BEHAVIOUR = {
    "RICH":       {"llm_temp": 0.25, "confidence_floor": 0.0,  "gate_threshold": 0.50, "warn": False},
    "ADEQUATE":   {"llm_temp": 0.20, "confidence_floor": 0.0,  "gate_threshold": 0.40, "warn": False},
    "SPARSE":     {"llm_temp": 0.15, "confidence_floor": 0.0,  "gate_threshold": 0.30, "warn": True},
    "UNRELIABLE": {"llm_temp": 0.10, "confidence_floor": 0.0,  "gate_threshold": 0.20, "warn": True},
    "BLOCKED":    {"llm_temp": 0.00, "confidence_floor": 0.0,  "gate_threshold": 0.00, "warn": True},
}

# Clinical concept groups used for coherence checking
CLINICAL_CONCEPT_GROUPS = {
    "respiratory": ["ventilat", "respiratory", "ards", "oxygen", "spo2", "hypoxia",
                    "extubat", "fio2", "peep", "tidal"],
    "cardiac":     ["cardiac", "heart", "arrhythmia", "tachycardia", "bradycardia",
                    "atrial", "ventricular", "ecg"],
    "sepsis":      ["sepsis", "septic", "antibiotic", "infection", "bacteremia",
                    "vasopressor", "norepinephrine"],
    "shock":       ["shock", "hypotension", "fluid", "resuscitat", "lactate", "perfusion"],
    "renal":       ["renal", "kidney", "creatinine", "dialysis", "rrt", "aki", "urine"],
    "metabolic":   ["lactate", "acidosis", "glucose", "electrolyte", "bicarbonate"],
    "neurological":["consciousness", "gcs", "sedation", "delirium", "neuro"],
    "coagulation": ["coagulat", "platelet", "fibrinogen", "dic", "bleeding"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. RAG STABILITY
# ══════════════════════════════════════════════════════════════════════════════

class RAGCache:
    """
    In-memory cache keyed by a deterministic hash of the query + patient
    context fingerprint.  Identical inputs return identical article lists
    regardless of PubMed API non-determinism.

    Cache is session-scoped (cleared on process restart).
    Entries expire after CACHE_TTL_MINUTES minutes to allow fresh data
    while maintaining intra-session stability.
    """

    CACHE_TTL_MINUTES = 60
    _store: dict = {}

    @classmethod
    def _key(cls, query: str, patient_fingerprint: str) -> str:
        raw = f"{query.strip().lower()}|{patient_fingerprint}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @classmethod
    def get(cls, query: str, patient_fingerprint: str) -> Optional[dict]:
        key   = cls._key(query, patient_fingerprint)
        entry = cls._store.get(key)
        if entry is None:
            return None
        age_min = (datetime.now() - entry["cached_at"]).total_seconds() / 60
        if age_min > cls.CACHE_TTL_MINUTES:
            del cls._store[key]
            return None
        print(f"   [RAGCache] HIT  key={key}  age={age_min:.0f}min")
        return entry["result"]

    @classmethod
    def put(cls, query: str, patient_fingerprint: str, result: dict):
        key = cls._key(query, patient_fingerprint)
        cls._store[key] = {"result": result, "cached_at": datetime.now()}
        print(f"   [RAGCache] STORE  key={key}  articles={len(result.get('evidence', []))}")

    @classmethod
    def clear(cls):
        cls._store.clear()


def patient_fingerprint(patient_info: dict, temporal_phases: dict) -> str:
    """
    Build a short stable fingerprint for a patient's current clinical context.
    Used as the cache key component alongside the query.
    """
    recent = temporal_phases.get("Recent", {}) if temporal_phases else {}
    parts  = [
        str(patient_info.get("age", "?")),
        str(patient_info.get("ventilated", False)),
        str(patient_info.get("diagnosis_str", ""))[:30],
        # Round vital values to 1 d.p. so minor sensor noise doesn't bust the cache
        "|".join(f"{k}={round(v, 1)}" for k, v in sorted(recent.items()) if v and v != 0),
    ]
    raw = "~".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _extract_patient_concepts(patient_info: dict, temporal_phases: dict) -> set:
    """
    Identify which clinical concept groups are active for this patient,
    based on vital sign abnormalities and diagnosis string.
    """
    active  = set()
    recent  = (temporal_phases or {}).get("Recent", {})
    diag    = str(patient_info.get("diagnosis_str", "")).lower()

    # Vital-sign based activation
    if recent.get("SpO2", 100) < 94 or recent.get("Respiration Rate", 0) > 25:
        active.add("respiratory")
    if recent.get("Lactate", 0) > 2.0 or recent.get("Systolic BP", 120) < 90:
        active.add("sepsis")
        active.add("shock")
    if recent.get("Heart Rate", 80) > 100 or recent.get("Heart Rate", 80) < 50:
        active.add("cardiac")
    if recent.get("PaO2_FiO2", 500) < 200:
        active.add("respiratory")
    if patient_info.get("ventilated"):
        active.add("respiratory")

    # Diagnosis string based activation
    for group, keywords in CLINICAL_CONCEPT_GROUPS.items():
        if any(kw in diag for kw in keywords):
            active.add(group)

    # Always include ICU-general
    if not active:
        active.add("respiratory")  # default safe assumption

    return active


def check_article_coherence(article: dict, patient_concepts: set) -> tuple[bool, float]:
    """
    Check whether an article is clinically coherent with the patient's active
    findings.  Returns (passes: bool, overlap_score: float).

    An article passes if its text shares at least one keyword from at least
    one of the patient's active clinical concept groups.
    """
    text         = (article.get("title", "") + " " + article.get("abstract", "")).lower()
    matched_kwds = 0
    total_kwds   = 0

    for group in patient_concepts:
        keywords = CLINICAL_CONCEPT_GROUPS.get(group, [])
        total_kwds += len(keywords)
        for kw in keywords:
            if kw in text:
                matched_kwds += 1

    overlap = matched_kwds / max(total_kwds, 1)
    passes  = matched_kwds >= 1   # at least one keyword from any active group

    return passes, round(overlap, 3)


def filter_articles_by_quality(
    articles: list,
    patient_concepts: set,
    min_score: float = ARTICLE_MIN_SCORE,
) -> tuple[list, dict]:
    """
    Apply two-stage quality filter to a ranked article list:
      1. Hard score threshold (removes low-relevance articles)
      2. Clinical coherence check (removes off-topic articles)

    Returns (filtered_articles, filter_report).
    """
    passed           = []
    low_score_count  = 0
    incoherent_count = 0

    for art in articles:
        score = art.get("score", 0.0)

        # Stage 1: score threshold
        if score < min_score:
            low_score_count += 1
            continue

        # Stage 2: clinical coherence
        coherent, overlap = check_article_coherence(art, patient_concepts)
        if not coherent:
            incoherent_count += 1
            continue

        passed.append({**art, "coherence_overlap": overlap})

    report = {
        "total_input":       len(articles),
        "passed":            len(passed),
        "rejected_score":    low_score_count,
        "rejected_coherence":incoherent_count,
        "pass_rate":         round(len(passed) / max(len(articles), 1), 2),
    }
    return passed, report


# ══════════════════════════════════════════════════════════════════════════════
# 2 & 3.  RAG QUALITY GATE + ERROR PROPAGATION
# ══════════════════════════════════════════════════════════════════════════════

SAFE_FALLBACK_EVIDENCE = """
[FALLBACK EVIDENCE — RAG quality below usable threshold]

The automated PubMed retrieval did not return sufficiently relevant articles
for this patient's specific clinical profile. The following general principles
apply to ICU management and should be interpreted alongside direct clinical
assessment:

[1] For mechanically ventilated patients with respiratory deterioration:
    Use lung-protective ventilation (tidal volume 6 ml/kg IBW, PEEP titration).
    Consider prone positioning if PaO2/FiO2 < 150 mmHg (moderate-severe ARDS).
    (Source: ARDS Network / ARDSnet protocol — established standard of care)

[2] For patients with elevated lactate (> 2 mmol/L) and haemodynamic instability:
    Initiate sepsis bundle within 1 hour: blood cultures, broad-spectrum
    antibiotics, 30 ml/kg crystalloid, vasopressors if MAP < 65 mmHg.
    (Source: Surviving Sepsis Campaign guidelines)

[3] For patients requiring high FiO2 (> 0.60):
    Monitor for oxygen toxicity; target SpO2 92-96% rather than 100%.
    Consider PEEP optimisation before increasing FiO2 further.
    (Source: ICU oxygen therapy consensus guidelines)

NOTE: These fallback principles replace retrieved evidence due to low RAG
confidence. They represent well-established clinical consensus, not
patient-specific evidence. Clinical judgment must be applied.
""".strip()


class RAGQualityGate:
    """
    Evaluates RAG output quality and either passes it through, warns,
    or substitutes a safe fallback block.

    Quality is assessed on:
      - Average article score across top-K results
      - Pass rate from article quality filter
      - Minimum number of usable articles (>= 2 required)
      - Whether any article has a meaningful clinical insight
    """

    @staticmethod
    def evaluate(
        rag_result: dict,
        patient_concepts: set,
        min_score: float = RAG_USABLE_THRESHOLD,
    ) -> dict:
        """
        Returns an augmented rag_result with quality_gate fields added.
        If the gate fails, top_evidence is replaced with SAFE_FALLBACK_EVIDENCE.
        """
        evidence    = rag_result.get("evidence", [])
        top_conf    = rag_result.get("confidence", 0.0)

        if not evidence:
            return RAGQualityGate._fail(rag_result, "No articles retrieved")

        # Apply article-level filtering
        filtered, filter_report = filter_articles_by_quality(evidence, patient_concepts, min_score)

        avg_score = sum(a.get("score", 0) for a in filtered) / max(len(filtered), 1)
        n_usable  = len(filtered)
        has_insights = any(a.get("clinical_insight", "") for a in filtered[:3])

        # Gate decision
        if n_usable < 2:
            return RAGQualityGate._fail(
                rag_result, f"Only {n_usable} article(s) passed quality filter",
                filter_report, filtered,
            )
        if avg_score < min_score:
            return RAGQualityGate._fail(
                rag_result, f"Average article score {avg_score:.3f} below threshold {min_score}",
                filter_report, filtered,
            )

        # Gate PASS
        status = "WARN" if (not has_insights or n_usable < 3) else "PASS"
        print(f"   [RAGGate] {status}  usable={n_usable}  avg_score={avg_score:.3f}  "
              f"pass_rate={filter_report['pass_rate']:.0%}")

        return {
            **rag_result,
            "evidence":            filtered,
            "quality_gate":        status,
            "quality_gate_reason": f"{n_usable} articles passed, avg_score={avg_score:.3f}",
            "filter_report":       filter_report,
            "fallback_used":       False,
        }

    @staticmethod
    def _fail(rag_result, reason, filter_report=None, filtered=None) -> dict:
        print(f"   [RAGGate] FAIL  reason={reason}")
        return {
            **rag_result,
            "evidence":            filtered or [],
            "top_evidence":        SAFE_FALLBACK_EVIDENCE,
            "confidence":          0.05,
            "quality_gate":        "FAIL",
            "quality_gate_reason": reason,
            "filter_report":       filter_report or {},
            "fallback_used":       True,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. LLM OUTPUT QUALITY GATE
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize_simple(text: str) -> set:
    return set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))


def _diagnosis_tokens(text: str) -> set:
    """Extract tokens from the DIAGNOSIS section of structured LLM output."""
    match = re.search(r'DIAGNOSIS\s*:\s*(.+?)(?:\n|KEY FINDINGS|$)', text, re.IGNORECASE | re.DOTALL)
    if match:
        return _tokenize_simple(match.group(1)[:200])
    # Fallback: first non-empty line
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) > 10 and not stripped.startswith("#"):
            return _tokenize_simple(stripped[:200])
    return set()


def _check_diagnosis_risk_alignment(text: str, risk_score: float) -> tuple[bool, str]:
    """
    Verify that the diagnosis section is consistent with the risk score.
    Returns (aligned: bool, reason: str).
    """
    diag_lower = text.lower()[:600]

    HIGH_RISK_PHRASES = [
        "septic shock", "ards", "respiratory failure", "cardiac arrest",
        "multi-organ", "multiorgan", "critical", "emergent", "ecmo",
        "immediate", "urgent intervention",
    ]
    LOW_RISK_PHRASES = [
        "stable", "routine monitoring", "no acute", "within normal",
        "mild", "post-operative monitoring", "conservative management",
    ]

    has_high = any(p in diag_lower for p in HIGH_RISK_PHRASES)
    has_low  = any(p in diag_lower for p in LOW_RISK_PHRASES)

    if risk_score >= 0.70 and has_low and not has_high:
        return False, f"HIGH risk ({risk_score:.2f}) but diagnosis uses low-acuity language"
    if risk_score < 0.35 and has_high and not has_low:
        return False, f"LOW risk ({risk_score:.2f}) but diagnosis uses high-acuity language"

    return True, "aligned"


def _check_specificity(text: str) -> tuple[bool, str]:
    """
    Check that the response contains at least some specific clinical values
    rather than being entirely generic.
    """
    # Look for numbers (vital values, doses, thresholds)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:mmHg|bpm|mmol|ml|mg|%|cm|hr|h\b)', text, re.IGNORECASE)
    # Look for specific clinical terms
    specific_terms = [
        "vasopressor", "norepinephrine", "tidal volume", "peep", "fio2",
        "lactate", "spo2", "pao2", "creatinine", "intubat", "extubat",
        "crystalloid", "antibiotic", "prone",
    ]
    has_specific = any(t in text.lower() for t in specific_terms)

    if not numbers and not has_specific:
        return False, "Response lacks specific clinical values or interventions"
    return True, f"{len(numbers)} clinical values found"


class OutputQualityGate:
    """
    Scores LLM primary assessment output on 5 dimensions:
      1. Structure   — required section headers present
      2. Length      — within acceptable character range
      3. Alignment   — diagnosis consistent with risk score
      4. Specificity — contains actual clinical values / terms
      5. Coherence   — no obvious self-contradictions
    
    Returns a QualityReport with tier PASS / WARN / FAIL and per-dimension scores.
    """

    @staticmethod
    def evaluate(
        text: str,
        risk_score: float,
        patient_summary: str = "",
    ) -> dict:
        scores    = {}
        reasons   = {}
        warnings  = []

        # Dimension 1: Structure
        found_sections = [s for s in REQUIRED_SECTIONS if s.upper() in text.upper()]
        struct_score   = len(found_sections) / len(REQUIRED_SECTIONS)
        scores["structure"]  = round(struct_score, 2)
        reasons["structure"] = f"{len(found_sections)}/{len(REQUIRED_SECTIONS)} required sections present"
        if struct_score < 1.0:
            missing = [s for s in REQUIRED_SECTIONS if s.upper() not in text.upper()]
            warnings.append(f"Missing sections: {', '.join(missing)}")

        # Dimension 2: Length
        n_chars = len(text)
        if n_chars < MIN_RESPONSE_CHARS:
            len_score = n_chars / MIN_RESPONSE_CHARS
            warnings.append(f"Response too short ({n_chars} chars, min {MIN_RESPONSE_CHARS})")
        elif n_chars > MAX_RESPONSE_CHARS:
            len_score = 0.7  # penalise but don't fail
            warnings.append(f"Response unusually long ({n_chars} chars)")
        else:
            len_score = 1.0
        scores["length"]  = round(len_score, 2)
        reasons["length"] = f"{n_chars} chars"

        # Dimension 3: Risk alignment
        aligned, align_reason = _check_diagnosis_risk_alignment(text, risk_score)
        scores["alignment"]  = 1.0 if aligned else 0.0
        reasons["alignment"] = align_reason
        if not aligned:
            warnings.append(f"Risk alignment issue: {align_reason}")

        # Dimension 4: Specificity
        specific, spec_reason = _check_specificity(text)
        scores["specificity"]  = 1.0 if specific else 0.5
        reasons["specificity"] = spec_reason
        if not specific:
            warnings.append("Response is too generic — lacks specific clinical values")

        # Dimension 5: Coherence (basic contradiction check)
        coherence_score, coherence_reason = OutputQualityGate._check_coherence(text, risk_score)
        scores["coherence"]  = coherence_score
        reasons["coherence"] = coherence_reason
        if coherence_score < 0.7:
            warnings.append(f"Coherence issue: {coherence_reason}")

        # Overall score (weighted)
        overall = (
            0.25 * scores["structure"]  +
            0.15 * scores["length"]     +
            0.25 * scores["alignment"]  +
            0.20 * scores["specificity"]+
            0.15 * scores["coherence"]
        )
        overall = round(overall, 3)

        # Tier
        if overall >= 0.75 and scores["alignment"] >= 0.5:
            tier = "PASS"
        elif overall >= 0.50:
            tier = "WARN"
        else:
            tier = "FAIL"

        print(f"   [OutputQC] {tier}  overall={overall:.3f}  "
              f"struct={scores['structure']:.2f}  align={scores['alignment']:.2f}  "
              f"spec={scores['specificity']:.2f}")
        if warnings:
            for w in warnings:
                print(f"             WARN: {w}")

        return {
            "tier":     tier,
            "overall":  overall,
            "scores":   scores,
            "reasons":  reasons,
            "warnings": warnings,
        }

    @staticmethod
    def _check_coherence(text: str, risk_score: float) -> tuple[float, str]:
        """
        Simple contradiction detector:
        - 'no intervention needed' + high risk = contradiction
        - 'immediate action' + low risk = contradiction
        - 'stable' + critical safety language = contradiction
        """
        lower = text.lower()
        contradictions = []

        if risk_score >= 0.70:
            if "no intervention" in lower or "no action needed" in lower:
                contradictions.append("'no intervention' contradicts high risk score")
            if "stable" in lower and "critical" not in lower:
                contradictions.append("'stable' language contradicts high risk score")

        if risk_score < 0.35:
            if "immediate" in lower and "transfer" in lower:
                contradictions.append("Urgency language contradicts low risk score")

        # Self-referential or meta-commentary
        if "as an ai" in lower or "i cannot provide" in lower or "i am unable" in lower:
            contradictions.append("LLM meta-commentary in clinical response")

        if contradictions:
            return 0.4, "; ".join(contradictions)
        return 1.0, "no contradictions detected"


# ══════════════════════════════════════════════════════════════════════════════
# 2. LLM SELF-CONSISTENCY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def self_consistency_check(
    response_a: str,
    response_b: str,
    threshold: float = SELF_CONSISTENCY_OVERLAP_THRESHOLD,
) -> dict:
    """
    Compare two independently-generated responses for diagnosis consistency.
    Uses Jaccard similarity on diagnosis-section tokens.

    Returns:
      consistent: bool
      overlap:    float  (Jaccard similarity of diagnosis tokens)
      tokens_a:   set
      tokens_b:   set
    """
    tokens_a = _diagnosis_tokens(response_a)
    tokens_b = _diagnosis_tokens(response_b)

    if not tokens_a or not tokens_b:
        # Cannot compare — treat as consistent to avoid false negatives
        return {"consistent": True, "overlap": 0.0, "reason": "Could not extract diagnosis tokens"}

    intersection = tokens_a & tokens_b
    union        = tokens_a | tokens_b
    jaccard      = len(intersection) / max(len(union), 1)

    consistent = jaccard >= threshold
    reason     = (
        f"Diagnosis token overlap: {jaccard:.2f} "
        f"({'consistent' if consistent else 'inconsistent'}), "
        f"shared: {', '.join(list(intersection)[:8])}"
    )

    print(f"   [SelfConsistency] overlap={jaccard:.3f}  consistent={consistent}")
    return {
        "consistent": consistent,
        "overlap":    round(jaccard, 3),
        "reason":     reason,
        "tokens_a":   list(tokens_a)[:15],
        "tokens_b":   list(tokens_b)[:15],
    }


def select_better_response(
    response_a: str,
    response_b: str,
    risk_score: float,
    consistency_result: dict,
) -> tuple[str, str]:
    """
    When two responses are inconsistent, select the one with better QC score.
    Returns (selected_response, selection_reason).
    """
    if consistency_result["consistent"]:
        return response_a, "responses consistent — using primary"

    qc_a = OutputQualityGate.evaluate(response_a, risk_score)
    qc_b = OutputQualityGate.evaluate(response_b, risk_score)

    if qc_a["overall"] >= qc_b["overall"]:
        return response_a, f"Primary response selected (QC: {qc_a['overall']:.3f} vs {qc_b['overall']:.3f})"
    else:
        return response_b, f"Secondary response selected — higher QC score ({qc_b['overall']:.3f} vs {qc_a['overall']:.3f})"


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATA QUALITY ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

class PatientDataQualityAssessor:
    """
    Evaluates the quality and completeness of a patient's ICU data
    before it enters the pipeline.

    Metrics:
      - seq_len:           number of timesteps processed
      - n_phases:          how many temporal phases have data
      - feature_coverage:  fraction of 11 features with non-zero recent values
      - missing_rate:      fraction of feature-timestep cells that are zero
      - outlier_rate:      fraction of cells that were clamped/replaced (proxy)
      - phase_variance:    inter-phase variability (low = flat/uninformative)

    Returns DATA_TIER: RICH / ADEQUATE / SPARSE / UNRELIABLE
    """

    @staticmethod
    def assess(gru_result: dict) -> dict:
        import numpy as np

        seq_len  = gru_result.get("seq_len", 0)
        phases   = gru_result.get("temporal_phases", {})
        features = gru_result.get("features", [])

        n_phases = len(phases)

        # Feature coverage from most recent phase
        recent        = phases.get("Recent", phases.get("Mid", phases.get("Early", {})))
        n_populated   = sum(1 for v in recent.values() if v and v != 0)
        n_features    = 11  # ICU_INPUT_SIZE
        feat_coverage = n_populated / n_features

        # Missing rate across full feature matrix
        if features:
            arr          = np.array(features, dtype=float)
            total_cells  = arr.size
            zero_cells   = int((arr == 0).sum())
            missing_rate = zero_cells / max(total_cells, 1)
        else:
            missing_rate = 1.0

        # ── Hard gate — absolute minimums before any pipeline stage runs ──────
        gate_blocked = False
        gate_reasons = []
        if seq_len < HARD_GATE_MIN_TIMESTEPS:
            gate_blocked = True
            gate_reasons.append(
                f"Only {seq_len} timestep(s) recorded "
                f"(minimum required: {HARD_GATE_MIN_TIMESTEPS}). "
                f"Add vitalPeriodic.csv with more observations."
            )
        if feat_coverage < HARD_GATE_MIN_FEAT_COVERAGE:
            gate_blocked = True
            gate_reasons.append(
                f"Only {feat_coverage:.0%} of features populated "
                f"(minimum required: {HARD_GATE_MIN_FEAT_COVERAGE:.0%}). "
                f"Check that vitalPeriodic.csv and lab.csv are present."
            )

        if gate_blocked:
            print(f"   [DataQC] BLOCKED  seq={seq_len}  feat_cov={feat_coverage:.0%}")
            for r in gate_reasons:
                print(f"           BLOCK: {r}")
            return {
                "tier":                 "BLOCKED",
                "gate_blocked":         True,
                "gate_reasons":         gate_reasons,
                "seq_len":              seq_len,
                "n_phases":             n_phases,
                "feature_coverage":     round(feat_coverage, 2),
                "n_populated_features": n_populated,
                "missing_rate":         round(missing_rate, 2),
                "phase_variance":       0.0,
                "llm_temperature":      0.0,
                "gate_threshold":       0.0,
                "should_warn":          True,
                "warnings":             gate_reasons,
                "confidence_cap":       CONFIDENCE_CAP["BLOCKED"],
            }

        # ── Tier classification (gate passed) ─────────────────────────────────
        phase_variance = PatientDataQualityAssessor._inter_phase_variance(phases)

        tier = "UNRELIABLE"
        for t, thresholds in DATA_TIER_THRESHOLDS.items():
            if (seq_len >= thresholds["min_seq"]
                    and n_phases >= thresholds["min_phases"]
                    and n_populated >= thresholds["min_features"]
                    and missing_rate <= thresholds["max_missing"]):
                tier = t
                break

        behaviour = DATA_TIER_BEHAVIOUR[tier]
        warnings  = PatientDataQualityAssessor._tier_warnings(
            tier, seq_len, n_phases, n_populated, missing_rate, phase_variance
        )

        result = {
            "tier":                 tier,
            "gate_blocked":         False,
            "gate_reasons":         [],
            "seq_len":              seq_len,
            "n_phases":             n_phases,
            "feature_coverage":     round(feat_coverage, 2),
            "n_populated_features": n_populated,
            "missing_rate":         round(missing_rate, 2),
            "phase_variance":       round(phase_variance, 4),
            "llm_temperature":      behaviour["llm_temp"],
            "gate_threshold":       behaviour["gate_threshold"],
            "should_warn":          behaviour["warn"],
            "warnings":             warnings,
            "confidence_cap":       CONFIDENCE_CAP.get(tier, 0.40),
        }

        print(f"   [DataQC] tier={tier}  seq={seq_len}  phases={n_phases}  "
              f"feat_cov={feat_coverage:.0%}  missing={missing_rate:.0%}  "
              f"phase_var={phase_variance:.4f}  conf_cap={result['confidence_cap']:.0%}")
        for w in warnings:
            print(f"           WARN: {w}")

        return result

    @staticmethod
    def _inter_phase_variance(phases: dict) -> float:
        """
        Compute mean across-phase variance for shared vital signs.
        Very low variance (< 0.001) means phases are nearly identical —
        the temporal model has nothing to differentiate.
        """
        import numpy as np
        if len(phases) < 2:
            return 0.0

        # Collect per-feature values across phases
        all_features: dict = {}
        for label, vals in phases.items():
            for feat, val in vals.items():
                if val and val != 0:
                    all_features.setdefault(feat, []).append(val)

        variances = []
        for feat, vals in all_features.items():
            if len(vals) >= 2:
                variances.append(float(np.var(vals)))

        return float(np.mean(variances)) if variances else 0.0

    @staticmethod
    def _tier_warnings(
        tier: str, seq_len: int, n_phases: int, n_populated: int,
        missing_rate: float, phase_variance: float,
    ) -> list:
        warnings = []
        if tier == "UNRELIABLE":
            warnings.append(
                f"Data quality is UNRELIABLE (seq_len={seq_len}, phases={n_phases}, "
                f"features={n_populated}). Outputs should not be used clinically."
            )
        elif tier == "SPARSE":
            warnings.append(
                f"Data is SPARSE — only {seq_len} timesteps and {n_populated} features. "
                f"Assessment confidence will be low."
            )
        if missing_rate > 0.60:
            warnings.append(
                f"{missing_rate:.0%} of feature-timestep cells are missing. "
                f"Add vitalPeriodic.csv and respiratoryCharting.csv for better coverage."
            )
        if phase_variance < 0.001 and n_phases >= 2:
            warnings.append(
                "Phase variance is very low — vitals appear flat across the observation window. "
                "This may indicate a patient with genuinely stable vitals, "
                "or that the data is insufficient to detect trends."
            )
        return warnings


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE AUDIT TRAIL
# ══════════════════════════════════════════════════════════════════════════════

class PipelineAuditTrail:
    """
    Records quality tiers and gate results at each pipeline stage.
    Passed downstream so each stage knows upstream quality.

    Used by:
      - Primary agent: adjusts temperature based on data tier
      - Validation agent: includes audit in its context
      - Report: surfaces QC information to the clinician
    """

    def __init__(self):
        self.stages: dict = {}
        self.created_at = datetime.now().isoformat()

    def record(self, stage: str, result: dict):
        self.stages[stage] = {**result, "recorded_at": datetime.now().isoformat()}

    def get(self, stage: str) -> dict:
        return self.stages.get(stage, {})

    def overall_quality(self) -> str:
        """
        Aggregate quality across all recorded stages.
        Returns GOOD / DEGRADED / POOR.
        """
        tiers = []
        for stage_data in self.stages.values():
            t = stage_data.get("tier") or stage_data.get("quality_gate") or ""
            tiers.append(t)

        if "FAIL" in tiers or "UNRELIABLE" in tiers:
            return "POOR"
        if "WARN" in tiers or "SPARSE" in tiers:
            return "DEGRADED"
        return "GOOD"

    def to_dict(self) -> dict:
        return {
            "created_at":     self.created_at,
            "overall_quality":self.overall_quality(),
            "stages":         self.stages,
        }

    def format_for_llm(self) -> str:
        """Format audit trail as a compact block for inclusion in LLM prompts."""
        lines = [f"Pipeline quality: {self.overall_quality()}"]
        for stage, data in self.stages.items():
            tier = data.get("tier") or data.get("quality_gate") or "?"
            warn = "; ".join(data.get("warnings", [])[:2])
            lines.append(f"  {stage}: {tier}" + (f" — {warn}" if warn else ""))
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: GENERATE FALLBACK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

def generate_fallback_assessment(
    patient_summary: str,
    risk_score: float,
    safety_check: dict,
    data_tier: str,
    qc_report: dict,
) -> str:
    """
    Generate a safe structured fallback assessment when the LLM output
    fails quality gates.

    Uses only rule-based logic from safety_check + risk_score, so it is
    fully deterministic and does not depend on LLM output quality.
    """
    risk_label = "HIGH" if risk_score >= 0.70 else ("MEDIUM" if risk_score >= 0.40 else "LOW")
    triggered  = safety_check.get("triggered", [])
    severity   = safety_check.get("severity", "LOW")

    # Build diagnosis from triggered rules
    if triggered:
        primary_issues = "; ".join(r["meaning"] for r in triggered[:3])
        diag = f"Clinical deterioration with safety alerts: {primary_issues}"
    elif risk_label == "HIGH":
        diag = "High-risk ICU patient — multiple physiological abnormalities detected"
    elif risk_label == "MEDIUM":
        diag = "Moderate-risk ICU patient — some abnormal findings present"
    else:
        diag = "Stable ICU patient — routine monitoring indicated"

    # Build interventions from triggered rules
    interventions = []
    rule_interventions = {
        "SpO2":             "1. Review oxygen therapy and ventilator settings",
        "Respiration Rate": "2. Assess respiratory status — consider ABG and CXR",
        "Heart Rate":       "3. Review haemodynamics — 12-lead ECG",
        "Systolic BP":      "4. Initiate vasopressor review — fluid challenge if appropriate",
        "Lactate":          "5. Sepsis workup — blood cultures, antibiotics within 1h",
        "PaO2_FiO2":        "6. Lung-protective ventilation — consider prone positioning",
        "FiO2":             "7. Optimise PEEP before increasing FiO2 further",
    }
    for rule in triggered:
        feat = rule.get("feature", "")
        if feat in rule_interventions:
            interventions.append(rule_interventions[feat])
    if not interventions:
        interventions = ["1. Continue routine ICU monitoring", "2. Re-assess in 2 hours"]

    risk_flags = [f"- {r['feature']} = {r['value']} (threshold: {r['operator']} {r['threshold']})"
                  for r in triggered[:3]] or ["- Continue monitoring all vital signs"]

    qc_warnings = "; ".join(qc_report.get("warnings", [])[:2])
    fallback_note = (
        f"\n\n[FALLBACK ASSESSMENT — generated from rule-based logic because LLM output "
        f"failed quality checks: {qc_warnings or 'quality score below threshold'}. "
        f"Data quality tier: {data_tier}. "
        f"This assessment reflects only objective threshold violations, not full clinical reasoning. "
        f"Senior physician review required.]"
    )

    return (
        f"DIAGNOSIS: {diag}\n\n"
        f"KEY FINDINGS:\n"
        + "\n".join(f"- {r['meaning']} ({r['feature']}={r['value']})" for r in triggered[:5])
        + ("\n- No critical safety thresholds breached" if not triggered else "")
        + f"\n\nINTERVENTIONS:\n" + "\n".join(interventions)
        + f"\n\nRISK FLAGS:\n" + "\n".join(risk_flags)
        + f"\n\nDIFFERENTIALS:\n- See full clinical assessment after data quality improves"
        + fallback_note
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE CAPPING
# ══════════════════════════════════════════════════════════════════════════════

def apply_confidence_cap(confidence: dict, data_quality: dict) -> dict:
    """
    Apply a hard ceiling to the confidence score based on data quality tier.

    Prevents inflated confidence scores when the underlying data is poor.
    For example, a patient with only 3 timesteps and 2 features should never
    show 78% confidence regardless of what the formula produces.

    Returns an updated confidence dict with capped score, pct, label,
    and a new field 'cap_applied' indicating whether capping changed the value.
    """
    from config import CONFIDENCE_BANDS

    tier = data_quality.get("tier", "UNKNOWN")
    cap  = CONFIDENCE_CAP.get(tier, 0.40)

    original_score = confidence.get("score", 0.0)
    capped_score   = min(original_score, cap)
    cap_applied    = capped_score < original_score

    if cap_applied:
        print(f"   [ConfidenceCap] tier={tier}  "
              f"original={original_score:.3f}  cap={cap:.2f}  "
              f"capped={capped_score:.3f}")

    # Re-derive label from capped score
    label = "very low"
    for band_label, (lo, hi) in CONFIDENCE_BANDS.items():
        if lo <= capped_score < hi:
            label = band_label.replace("_", " ")
            break

    return {
        **confidence,
        "score":       round(capped_score, 3),
        "pct":         int(capped_score * 100),
        "label":       label,
        "cap_applied": cap_applied,
        "cap_reason":  f"Data tier {tier} — confidence capped at {int(cap*100)}%" if cap_applied else "",
        "original_score": original_score,
        "original_pct":   int(original_score * 100),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HARD GATE RESULT  —  returned when pipeline is blocked entirely
# ══════════════════════════════════════════════════════════════════════════════

BLOCKED_ASSESSMENT = (
    "DIAGNOSIS: Insufficient data — clinical assessment cannot be generated.\n\n"
    "KEY FINDINGS:\n"
    "- The hard data gate was triggered before any AI analysis ran.\n"
    "- This is not a clinical finding — it is a data quality failure.\n\n"
    "INTERVENTIONS:\n"
    "1. Ensure vitalPeriodic.csv is present and contains at least 10 timesteps for this patient.\n"
    "2. Ensure that at least 50% of the 11 monitored features have recorded values.\n"
    "3. Re-run the pipeline once the data requirements are met.\n\n"
    "RISK FLAGS:\n"
    "- No risk score computed — pipeline did not run.\n"
    "- Do not use any output from this run for clinical decision-making.\n\n"
    "DIFFERENTIALS:\n"
    "- Not applicable — data gate blocked assessment generation."
)


def build_blocked_result(patient_id: int, data_quality: dict) -> dict:
    """
    Build a complete pipeline result dict for a BLOCKED patient.
    All downstream fields are set to safe sentinel values so the
    Streamlit UI and report writer never crash on missing keys.
    """
    reasons = data_quality.get("gate_reasons", ["Insufficient data"])

    return {
        "timestamp":  __import__("datetime").datetime.now().isoformat(),
        "patient_id": patient_id,
        "blocked":    True,
        "block_reasons": reasons,
        "data_quality": data_quality,
        "overall_quality": "BLOCKED",
        "gru": {
            "patient_id":       patient_id,
            "seq_len":          data_quality.get("seq_len", 0),
            "risk_score":       0.0,
            "clinical_summary": f"Patient {patient_id}: pipeline blocked — insufficient data.",
            "patient_info":     {},
            "temporal_phases":  {},
            "data_quality":     data_quality,
        },
        "safety": {
            "triggered":   [],
            "all_checks":  [],
            "n_triggered": 0,
            "severity":    "LOW",
        },
        "rag": {
            "query":         "",
            "evidence":      [],
            "top_evidence":  "",
            "confidence":    0.0,
            "quality_gate":  "SKIP",
            "fallback_used": False,
        },
        "confidence": {
            "score":       0.0,
            "pct":         0,
            "label":       "blocked",
            "data_richness": 0.0,
            "temporal_cov":  0.0,
            "rag_quality":   0.0,
            "cap_applied":   True,
            "cap_reason":    "Pipeline blocked — no data",
        },
        "primary":     BLOCKED_ASSESSMENT,
        "explanation": {
            "risk_label":    "BLOCKED",
            "risk_level":    "BLOCKED",
            "risk_score":    0.0,
            "risk_pct":      0,
            "top_line":      "Insufficient data — assessment blocked",
            "why_now":       "; ".join(reasons),
            "delta_why_now": "; ".join(reasons),
            "watch_for":     [
                "Obtain complete vital signs data before re-running",
                "Ensure vitalPeriodic.csv is present with at least 10 timesteps",
                "Verify that lab.csv and respiratoryCharting.csv are available",
            ],
            "sparklines":          {},
            "threshold_crossings": [],
            "evidence":            [],
            "primary_assessment":  BLOCKED_ASSESSMENT,
            "temporal_phases":     {},
            "phase_trajectory":    "",
            "safety_check":        {},
            "confidence":          {},
        },
        "validation": {
            "verdict":          "BLOCKED",
            "score":            "N/A",
            "concern":          "Pipeline did not run — data gate blocked.",
            "validation_text":  "No validation performed — pipeline blocked by hard data gate.",
            "actions": [
                "Do not use any output from this run for clinical decisions.",
                "Add complete patient data (vitalPeriodic.csv, lab.csv) and re-run.",
                "Contact the data engineering team if eICU files are missing.",
            ],
        },
        "audit_trail":  {"overall_quality": "BLOCKED", "stages": {}},
        "output_qc":    {"tier": "BLOCKED", "overall": 0.0, "warnings": reasons},
    }
