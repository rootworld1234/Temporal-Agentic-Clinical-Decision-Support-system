"""
modules/agentic_validation.py -- Agentic Validation Layer (Research-Grade)

Key improvements over prototype:
  1. Rule-based safety check: evaluates CLINICAL_THRESHOLDS before
     calling the LLM — surfaces violated rules explicitly in output
     (prototype had no pre-LLM safety gate)
  2. Structured primary agent output: prompts the LLM to return
     clearly labelled sections that map to the report format
  3. Validation prompt hardened: includes safety rule violations
     so the validator can weigh them against the LLM assessment
  4. Verdict parsing unchanged (robust) + actions unchanged

LLM Routing (Stable Architecture):
  Assessment   -> call_llm_local  (LM Studio ONLY — no Gemini fallback)
  Validation   -> call_llm_local  (LM Studio ONLY — no Gemini fallback)
  Benchmarks   -> call_llm_local  (LM Studio ONLY — no Gemini fallback)

  Gemini is reserved exclusively for Final Explanation (clinical_explainer.py)
  and the optional Final Report. It is never called from this module.
"""

import re
import random
from datasets import load_dataset
from modules.llm_client import call_llm_local, build_messages, get_active_backend

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SAFETY_RULES, CLINICAL_THRESHOLDS

# Quality control imports (lazy)
def _qc():
    try:
        from modules.quality_control import (
            OutputQualityGate, self_consistency_check,
            select_better_response, generate_fallback_assessment,
            PatientDataQualityAssessor,
        )
        return OutputQualityGate, self_consistency_check, select_better_response, generate_fallback_assessment, PatientDataQualityAssessor
    except ImportError:
        return None, None, None, None, None


# ── Model selection (unchanged) ────────────────────────────────────────────────

def select_model() -> str:
    from modules.llm_client import _get_key, _get_lmstudio_model, gemini_budget_status
    import config

    print("\nLLM Backend Configuration (Stable Architecture)")
    print("   LM Studio  -> RAG Insight Extraction, Primary Diagnosis, Validation, Benchmarks")
    print("   Gemini     -> Final Explanation, Final Report (ONLY)\n")

    key = _get_key()
    if key:
        budget = gemini_budget_status()
        remaining_today = config.FREE_TIER_RPD - budget["calls_today"]
        print(f"   Gemini key found  (model: {config.GEMINI_MODEL})")
        print(f"   Free tier budget: {remaining_today}/{config.FREE_TIER_RPD} calls remaining today")
    else:
        print("   Gemini key NOT set -> all calls will use LM Studio")

    lm_model = _get_lmstudio_model()
    if lm_model:
        print(f"   LM Studio running (model: {lm_model})")
    else:
        if not key:
            print("   LM Studio not detected AND no Gemini key -- no LLM available")
        else:
            print("   LM Studio not detected -- will use Gemini")

    return None


# ── Rule-Based Safety Check (NEW) ─────────────────────────────────────────────

def run_safety_rule_check(temporal_phases: dict) -> dict:
    """
    Evaluate SAFETY_RULES against the most recent phase of vital measurements.
    Returns list of triggered rules with clinical meanings.

    This runs BEFORE the LLM assessment, so violations are surfaced
    regardless of what the LLM says.
    """
    recent   = temporal_phases.get("Recent", {})
    if not recent:
        # Fall back to any available phase
        for label in ["Mid", "Early"]:
            if label in temporal_phases:
                recent = temporal_phases[label]
                break

    triggered = []
    all_checks = []

    for (feature, operator, threshold, meaning) in SAFETY_RULES:
        val = recent.get(feature, None)
        if val is None or val == 0:
            all_checks.append({
                "feature":  feature,
                "operator": operator,
                "threshold":threshold,
                "meaning":  meaning,
                "value":    None,
                "triggered":False,
                "note":     "no data",
            })
            continue

        fired = False
        if operator == "<"  and val < threshold:  fired = True
        elif operator == ">" and val > threshold: fired = True
        elif operator == "<=" and val <= threshold: fired = True
        elif operator == ">=" and val >= threshold: fired = True

        check = {
            "feature":  feature,
            "operator": operator,
            "threshold":threshold,
            "meaning":  meaning,
            "value":    round(val, 2),
            "triggered":fired,
        }
        all_checks.append(check)
        if fired:
            triggered.append(check)

    severity = "HIGH" if len(triggered) >= 3 else "MODERATE" if triggered else "LOW"

    print(f"\n   Safety Rule Check: {len(triggered)}/{len(SAFETY_RULES)} rules triggered (severity: {severity})")
    for r in triggered:
        print(f"      !! {r['meaning']} ({r['feature']}={r['value']} {r['operator']} {r['threshold']})")

    return {
        "triggered":     triggered,
        "all_checks":    all_checks,
        "n_triggered":   len(triggered),
        "severity":      severity,
    }


def format_safety_rules_for_prompt(safety_check: dict) -> str:
    """Format safety rule violations as a compact block for LLM prompts."""
    triggered = safety_check.get("triggered", [])
    if not triggered:
        return "No safety rules triggered."
    lines = ["Safety rules triggered:"]
    for r in triggered:
        lines.append(f"  - Rule: {r['feature']} {r['operator']} {r['threshold']} "
                     f"(actual: {r['value']}) → {r['meaning']}")
    return "\n".join(lines)


# ── Primary Diagnostic Agent ───────────────────────────────────────────────────

# FIX 2: Risk-diagnosis alignment constraints injected into system prompt.
# The LLM must not diagnose severe life-threatening conditions when the
# risk model says LOW, and must not say LOW when critical vitals are present.
_RISK_DIAGNOSIS_CONSTRAINT = """
CRITICAL CONSISTENCY RULE — you MUST follow this:
- If risk level is LOW (<30%): do NOT diagnose ARDS, septic shock, cardiac arrest,
  or other immediately life-threatening conditions. Use terms like "stable",
  "mild", "monitoring recommended".
- If risk level is MEDIUM (30–70%): diagnoses may include moderate conditions.
  Avoid catastrophic language unless safety alerts are present.
- If risk level is HIGH (>70%) OR safety alerts are triggered: diagnoses should
  reflect the severity. Do not downplay findings.
Your diagnosis MUST be consistent with the risk level provided."""


PRIMARY_SYSTEM = """You are an expert ICU physician and clinical decision support system.
Given a patient's temporal clinical state and supporting biomedical evidence, provide a structured assessment.

Use EXACTLY these section headers:

DIAGNOSIS: (primary diagnosis or clinical impression in one line)

KEY FINDINGS:
- (finding 1 with specific value if known)
- (finding 2)
- (finding 3)

INTERVENTIONS:
1. (highest priority — specific and actionable)
2. (second priority)
3. (third priority)

RISK FLAGS:
- (specific parameter to monitor, with threshold)
- (second risk flag)

DIFFERENTIALS:
- (differential 1)
- (differential 2)

Be specific. Name actual values where you have them. Do not use vague language.""" + _RISK_DIAGNOSIS_CONSTRAINT


def _risk_label_for_prompt(risk_score: float) -> str:
    """Convert risk score to the label passed into the LLM prompt."""
    if risk_score >= 0.70:   return f"HIGH ({int(risk_score*100)}%) — severe findings expected"
    elif risk_score >= 0.40: return f"MEDIUM ({int(risk_score*100)}%) — moderate findings expected"
    else:                    return f"LOW ({int(risk_score*100)}%) — stable / mild findings expected"


def run_primary_agent(
    patient_summary:  str,
    tcsv_description: str,
    evidence_text:    str,
    safety_check:     dict  = None,
    risk_score:       float = None,
    model:            str   = None,
    data_quality:     dict  = None,
    audit_trail       = None,
) -> str:
    """
    Primary diagnostic reasoning agent with QC layer.

    QC additions:
      - Temperature is lowered based on data quality tier (less hallucination
        on sparse data)
      - Self-consistency check: samples twice and cross-validates diagnosis
        sections; selects the more coherent response
      - OutputQualityGate: scores the chosen response; falls back to
        deterministic rule-based assessment if the LLM output fails
      - Audit trail records QC results for downstream stages
    """
    print("\nPrimary Diagnostic Agent reasoning...")

    # Determine temperature from data quality tier
    if data_quality:
        temperature = data_quality.get("llm_temperature", 0.25)
    else:
        temperature = 0.25

    safety_block = ""
    if safety_check and safety_check.get("triggered"):
        safety_block = (
            f"\nSAFETY ALERTS (pre-computed rule violations):\n"
            f"{format_safety_rules_for_prompt(safety_check)}\n"
        )

    risk_block = ""
    if risk_score is not None:
        risk_block = f"\nMODEL RISK LEVEL: {_risk_label_for_prompt(risk_score)}\n"

    # Include pipeline quality context if available
    audit_block = ""
    if audit_trail is not None:
        audit_block = f"\nPIPELINE QUALITY CONTEXT:\n{audit_trail.format_for_llm()}\n"

    messages = build_messages(
        system_prompt=PRIMARY_SYSTEM,
        user_content=(
            f"PATIENT CLINICAL SUMMARY:\n{patient_summary}\n\n"
            f"TEMPORAL TRAJECTORY ANALYSIS:\n{tcsv_description}\n"
            f"{risk_block}"
            f"{safety_block}"
            f"{audit_block}\n"
            f"SUPPORTING BIOMEDICAL EVIDENCE (PubMed, top 3):\n{evidence_text[:1500]}\n\n"
            f"Provide your structured clinical assessment. Your DIAGNOSIS must match the MODEL RISK LEVEL above."
        ),
    )

    # ── Sample 1 (primary) ────────────────────────────────────────────────────
    response_a = call_llm_local(messages, temperature=temperature, max_tokens=2048)
    print(f"   Primary assessment generated ({len(response_a)} chars) via {get_active_backend()}")

    # ── Sample 2 (consistency check) — only if data is sparse or output is short ──
    OutputQualityGate, self_consistency_check, select_better_response, generate_fallback_assessment, _ = _qc()

    run_second_sample = (
        data_quality is not None and data_quality.get("tier") in ("SPARSE", "UNRELIABLE")
    ) or len(response_a) < 300

    if run_second_sample and self_consistency_check and select_better_response:
        print("   Running second sample for self-consistency check...")
        response_b = call_llm_local(messages, temperature=max(temperature - 0.05, 0.05), max_tokens=2048)
        consistency = self_consistency_check(response_a, response_b)
        chosen, reason = select_better_response(response_a, response_b, risk_score or 0.5, consistency)
        print(f"   Self-consistency: overlap={consistency['overlap']:.3f}  selected: {reason}")
        if audit_trail:
            audit_trail.record("self_consistency", consistency)
    else:
        chosen = response_a

    # ── Output quality gate ───────────────────────────────────────────────────
    if OutputQualityGate:
        qc_report = OutputQualityGate.evaluate(chosen, risk_score or 0.5, patient_summary)
        if audit_trail:
            audit_trail.record("output_quality", qc_report)

        if qc_report["tier"] == "FAIL" and generate_fallback_assessment:
            print(f"   [OutputQC] FAIL — substituting rule-based fallback assessment")
            data_tier = (data_quality or {}).get("tier", "UNKNOWN")
            chosen = generate_fallback_assessment(
                patient_summary, risk_score or 0.5, safety_check or {},
                data_tier, qc_report,
            )
        elif qc_report["tier"] == "WARN":
            print(f"   [OutputQC] WARN — response passed with warnings, proceeding")

    return chosen


# ── 3C3H Validation Framework (hardened prompt) ───────────────────────────────

VALIDATOR_SYSTEM = """You are an ICU physician reviewing a clinical assessment. Be brief and direct.

Write 2-3 sentences reviewing the assessment, then end with your verdict.

You MUST end your response with exactly one of these lines:
MY VERDICT: APPROVE
MY VERDICT: REVISE - [reason]
MY VERDICT: REJECT - [reason]

Example of a complete response:
The assessment correctly identifies the clinical picture and interventions are appropriate.
MY VERDICT: APPROVE"""


def run_validation_agent(
    primary_output:   str,
    evidence_text:    str,
    patient_summary:  str,
    safety_check:     dict = None,
    model:            str  = None,
) -> dict:
    """
    Supervisory validation agent with safety rule context.
    FIX 4: Retries up to 3 times if response is too short (<50 chars)
    or missing the MY VERDICT line. Uses a simplified prompt on retry
    to maximise chance of getting a parseable verdict.
    """
    print("\nComponent C: 3C3H Agentic Validation")

    assessment_summary = primary_output[:400] if primary_output else "No assessment available."
    safety_block       = format_safety_rules_for_prompt(safety_check) if safety_check else ""

    def _build_messages(simplified: bool = False) -> list:
        if simplified:
            # Minimal prompt on retry — just ask for the verdict line
            return build_messages(
                "You are an ICU physician. Review the assessment below and respond with "
                "EXACTLY one of these three lines and nothing else:\n"
                "MY VERDICT: APPROVE\n"
                "MY VERDICT: REVISE - [one sentence]\n"
                "MY VERDICT: REJECT - [one sentence]",
                f"Assessment: {assessment_summary[:300]}\n"
                f"Patient: {patient_summary[:200]}\n"
                f"Respond with MY VERDICT only.",
            )
        return build_messages(
            system_prompt=VALIDATOR_SYSTEM,
            user_content=(
                f"Patient context: {patient_summary[:300]}\n\n"
                f"{safety_block}\n\n"
                f"Assessment to review:\n{assessment_summary}\n\n"
                f"Relevant evidence summary: {evidence_text[:300]}\n\n"
                f"Give your clinical opinion and end with MY VERDICT: APPROVE / REVISE / REJECT."
            ),
        )

    # FIX 4: Retry loop — up to 3 attempts
    validation_text = ""
    for attempt in range(3):
        simplified   = attempt >= 1        # simplify prompt from 2nd attempt onward
        msgs         = _build_messages(simplified)
        raw          = call_llm_local(msgs, temperature=0.3 if attempt == 0 else 0.1, max_tokens=1500)
        print(f"   Validation attempt {attempt+1}: {len(raw)} chars via {get_active_backend()}")

        if len(raw) < 50:
            print(f"   Response too short ({len(raw)} chars) — retrying...")
            continue

        validation_text = raw
        if "MY VERDICT:" in raw.upper():
            print(f"   MY VERDICT line found — proceeding")
            break
        print(f"   No MY VERDICT line found — retrying with simplified prompt...")

    if not validation_text:
        validation_text = "Validation failed after 3 attempts — treating as NEEDS_REVISION."

    print(f"   Final response:\n{validation_text[:500]}")

    # Parse verdict (unchanged logic — robust)
    verdict    = "UNKNOWN"
    upper_full = validation_text.upper()

    if "MY VERDICT:" in upper_full:
        idx     = upper_full.index("MY VERDICT:")
        segment = upper_full[idx: idx + 80]
        if   "APPROVE" in segment and "REVISE" not in segment:
            verdict = "APPROVED"
        elif "REVISE" in segment:
            verdict = "NEEDS_REVISION"
        elif "REJECT" in segment:
            verdict = "REJECTED"

    # Fallback inference
    if verdict == "UNKNOWN" and len(validation_text) > 30:
        lower    = validation_text.lower()
        positive = ["sound", "appropriate", "reasonable", "safe to act", "good reasoning", "well-reasoned", "correct", "accurate", "adequate", "sufficient", "consistent", "acceptable", "addresses", "well-structured"]
        negative = ["concern", "missing", "incorrect", "wrong", "not appropriate", "insufficient"]
        revision = ["revise", "revision", "needs work", "before acting", "incomplete"]
        pos_hits = sum(1 for w in positive if w in lower)
        neg_hits = sum(1 for w in negative if w in lower)
        rev_hits = sum(1 for w in revision if w in lower)
        if rev_hits > 0 or (neg_hits > 0 and pos_hits > 0):
            verdict = "NEEDS_REVISION"
        elif pos_hits >= 2 and neg_hits == 0:
            verdict = "APPROVED"
        elif neg_hits >= 2:
            verdict = "REJECTED"

    # Extract concern
    concern = ""
    for marker in ["MY VERDICT: REVISE -", "MY VERDICT: REJECT -"]:
        if marker in upper_full:
            idx     = upper_full.index(marker) + len(marker)
            concern = validation_text[idx: idx + 120].strip().split("\n")[0]
            break

    verdict_labels = {
        "APPROVED":       "APPROVED",
        "NEEDS_REVISION": "NEEDS REVISION",
        "REJECTED":       "REJECTED",
    }
    print(f"   Verdict: {verdict_labels.get(verdict, 'UNKNOWN')} | via {get_active_backend()}")
    if concern:
        print(f"   Concern: {concern}")

    # Safety rule override: force NEEDS_REVISION if critical rules fired
    # and validation returned APPROVED without acknowledging them
    if safety_check and safety_check.get("n_triggered", 0) >= 3 and verdict == "APPROVED":
        override_reason = f"{safety_check['n_triggered']} critical safety rules triggered"
        if not any(r["feature"].lower() in validation_text.lower()
                   for r in safety_check.get("triggered", [])):
            verdict = "NEEDS_REVISION"
            concern = f"Safety override: {override_reason} not addressed in assessment"
            print(f"   !! Safety override applied: {override_reason}")

    actions = get_verdict_actions(verdict, concern)
    print("\n   Recommended Actions:")
    for action in actions:
        print(f"      {action}")

    return {
        "validation_text": validation_text,
        "verdict":         verdict,
        "score":           "N/A",
        "concern":         concern,
        "actions":         actions,
        "safety_check":    safety_check,
    }


def get_verdict_actions(verdict: str, concern: str = "") -> list:
    concern_note = f" ({concern})" if concern else ""
    if verdict == "APPROVED":
        return [
            "Assessment APPROVED by peer review — safe to act on.",
            "Initiate the prioritised interventions per protocol.",
            "Monitor flagged parameters at the specified intervals.",
            "Document your clinical decision with this report as evidence.",
        ]
    elif verdict == "NEEDS_REVISION":
        return [
            f"Assessment needs REVISION before clinical action{concern_note}.",
            "Senior physician review required — do not act unilaterally.",
            "Address the concern highlighted in the clinician opinion above.",
            "Manually verify recommendations against the cited evidence.",
            "Re-run pipeline after resolving the flagged issue.",
        ]
    elif verdict == "REJECTED":
        return [
            f"Assessment REJECTED{concern_note} — DO NOT act on these recommendations.",
            "Escalate to attending physician immediately.",
            "Discard this assessment — do not use as clinical guidance.",
            "Check input data quality: missing vitals, corrupt labs, or sparse history.",
            "Re-run pipeline with corrected / enriched patient data.",
            "Log this rejection for system audit and quality review.",
        ]
    else:
        return [
            "Verdict undetermined — treat as NEEDS_REVISION.",
            "Do not act without senior physician review.",
            "Read the full clinician opinion above for manual interpretation.",
            "Re-run if the response appeared truncated or incomplete.",
        ]


# ── MedAgentsBench Evaluation (FIX 7) ─────────────────────────────────────────

def _safe_get(row, key: str, default=""):
    """Safely get a value from a HuggingFace dataset row (dict or dataclass)."""
    try:
        if isinstance(row, dict):
            return row.get(key, default)
        return getattr(row, key, default)
    except Exception:
        return default


def _normalise_row(row, subset: str) -> dict:
    """
    FIX 7: Each MedAgentsBench subset has a slightly different schema.
    This normaliser handles the known variants to avoid the
    'must be called with a dataclass' error and missing field crashes.

    Subset schemas:
      MedQA, MedMCQA, MMLU, MedBullets — have 'options' dict + 'answer_idx'
      PubMedQA                          — has 'final_decision' as answer
      MedExQA                           — has 'answer' directly
    """
    question = _safe_get(row, "question", "") or _safe_get(row, "input", "")
    options  = _safe_get(row, "options",  {})
    answer   = _safe_get(row, "answer",   "")
    answer_idx = _safe_get(row, "answer_idx", "")

    # PubMedQA: answer is in 'final_decision' (yes/no/maybe)
    if subset == "PubMedQA":
        answer = _safe_get(row, "final_decision", answer) or answer
        options = {}
        answer_idx = ""

    # MedExQA: answer is a plain string (no options dict)
    if subset == "MedExQA":
        options = {}
        answer_idx = ""

    # Resolve answer text from options dict if needed
    if isinstance(options, dict) and answer_idx and answer_idx in options:
        answer_text = options[answer_idx]
    elif answer:
        answer_text = str(answer)
    else:
        answer_text = str(answer_idx)

    # Format options as readable string
    if isinstance(options, dict) and options:
        options_str = " | ".join(
            f"{k}: {v}" for k, v in options.items()
            if v and str(v).lower() != "n/a"
        )
    else:
        options_str = ""

    return {
        "question":   str(question)[:500],
        "options":    options_str,
        "answer":     str(answer_text)[:300],
        "answer_idx": str(answer_idx),
        "reason":     _safe_get(row, "reason", ""),
        "subset":     subset,
    }


def load_medagentsbench(n_samples: int = 5) -> list:
    print(f"\nLoading MedAgentsBench from HuggingFace...")
    # FIX 7: Try subsets in order of reliability; skip on any per-subset error
    SUBSETS     = ["PubMedQA", "MedMCQA", "MMLU", "MedBullets", "MedExQA", "MedQA"]
    all_samples = []

    for subset in SUBSETS:
        if len(all_samples) >= n_samples:
            break
        try:
            dataset = load_dataset(
                "super-dainiu/medagents-benchmark",
                name=subset,
                split="test",
                trust_remote_code=True,
            )
            if len(dataset) == 0:
                print(f"   {subset}: empty — skipping")
                continue

            needed  = n_samples - len(all_samples)
            take    = min(needed, len(dataset), 2)
            indices = random.sample(range(len(dataset)), take)

            loaded_this_subset = 0
            for idx in indices:
                try:
                    row = dataset[idx]
                    # FIX 7: convert dataclass-style rows to dict via __dict__ if needed
                    if not isinstance(row, dict):
                        try:
                            row = dict(row)
                        except Exception:
                            import dataclasses
                            row = dataclasses.asdict(row) if dataclasses.is_dataclass(row) else vars(row)

                    sample = _normalise_row(row, subset)
                    if sample["question"]:
                        all_samples.append(sample)
                        loaded_this_subset += 1
                except Exception as row_err:
                    print(f"   {subset}[{idx}] row error: {row_err}")

            print(f"   {subset}: {loaded_this_subset} cases loaded")

        except Exception as e:
            print(f"   {subset} failed: {e}")
            # Continue to next subset — do NOT abort entire benchmark

    if not all_samples:
        print("   HuggingFace unavailable or all subsets failed — using built-in ICU cases...")
        return _fallback_cases(n_samples)

    print(f"   Total: {len(all_samples)} cases loaded from HuggingFace")
    return all_samples[:n_samples]


def _fallback_cases(n: int) -> list:
    cases = [
        {
            "question": "A 65-year-old ICU patient has fever, hypotension, tachycardia, and lactate 4.2. Blood cultures pending. Best immediate management?",
            "answer":   "Sepsis bundle: cultures, broad-spectrum antibiotics within 1 hour, 30ml/kg crystalloid, vasopressors if MAP <65.",
        },
        {
            "question": "Ventilated ICU patient develops bilateral infiltrates, PaO2/FiO2=180. Diagnosis and ventilator strategy?",
            "answer":   "Moderate ARDS. Low tidal volume 6ml/kg IBW, optimise PEEP, prone if PaO2/FiO2 <150.",
        },
        {
            "question": "Post-cardiac surgery patient: oliguria, rising creatinine, metabolic acidosis at 24h. Approach?",
            "answer":   "AKI: optimise haemodynamics, avoid nephrotoxins, monitor for RRT need.",
        },
    ]
    return cases[:n]


def run_medagentsbench_eval(model: str = None, n_samples: int = 3) -> dict:
    print(f"\nMedAgentsBench Evaluation ({n_samples} cases)")
    cases   = load_medagentsbench(n_samples)
    results = []
    correct = 0

    for i, case in enumerate(cases):
        question    = case.get("question") or case.get("input") or "Unknown question"
        options_str = case.get("options", "")
        reference   = case.get("answer", "No reference")
        subset      = case.get("subset", "Unknown")
        print(f"\n   Case {i+1}/{len(cases)} [{subset}]: {question[:75]}...")

        user_content = question
        if options_str:
            user_content += f"\n\nOptions: {options_str}\n\nSelect the best answer and explain briefly."

        ans_messages = build_messages(
            "You are an expert clinical decision support system. Answer the medical question accurately. "
            "For multiple choice, state the correct option letter and answer, then give a brief explanation.",
            user_content,
        )
        agent_answer = call_llm_local(ans_messages, temperature=0.2, max_tokens=512)

        judge_messages = build_messages(
            "You are a medical examiner. Judge if the agent's answer matches the correct answer. "
            "Respond with CORRECT or INCORRECT followed by one sentence explanation.",
            f"Question: {question}\n\nCorrect Answer: {reference}\n\nAgent Answer: {agent_answer}\n\nIs the agent correct?",
        )
        judgment = call_llm_local(judge_messages, temperature=0.1, max_tokens=150)

        upper      = judgment.upper()
        is_correct = (
            "CORRECT" in upper
            and not (upper.startswith("INCORRECT") or "\nINCORRECT" in upper or " INCORRECT" in upper[:30])
        )
        if is_correct:
            correct += 1

        status = "CORRECT" if is_correct else "INCORRECT"
        print(f"   {status}: {judgment[:100]}")
        results.append({
            "case":         i + 1,
            "question":     question[:100],
            "agent_answer": agent_answer[:200],
            "judgment":     judgment[:150],
            "correct":      is_correct,
        })

    accuracy = correct / max(len(cases), 1) * 100
    print(f"\n   MedAgentsBench: {correct}/{len(cases)} = {accuracy:.1f}%")
    return {"cases": results, "accuracy": accuracy, "correct": correct, "total": len(cases)}
