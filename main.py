"""
TA-CDSS -- Temporal-Aware Clinical Decision Support System (Research-Grade)

Architecture:
  GRU Temporal Engine (11 features, phase analysis)
  -> Hybrid PubMed RAG (clinical query, insight extraction)
  -> Safety Rule Check (pre-LLM gate)
  -> Primary Diagnostic Agent (safety-aware prompt)
  -> Confidence Score Estimation
  -> Clinical Explainer (delta-based WHY NOW, temporal phases)
  -> 3C3H Agentic Validation (safety-override)

Key improvements over prototype:
  - 11 ICU features (was 6): adds FiO2, PEEP, Tidal Volume, PaO2/FiO2, Lactate
  - Calibrated risk score (clinical abnormality count + trend signal)
  - Phase-based temporal analysis (Hour 0–2 / 2–4 / 4–6)
  - Clinically-specific PubMed queries (vs generic "ICU management")
  - Pre-LLM safety rule check (9 clinical rules)
  - Delta-based WHY NOW: "SpO2 96→91%, RR 18→27 br/min"
  - Confidence score displayed to clinician
  - Evidence clinical_insight sentence (vs raw abstract)

Stable Architecture LLM Routing:
  [GRU Temporal Model]      -> Local (PyTorch, no LLM)
  [Safety Rules]            -> Local (rule engine, no LLM)
  [RAG Retrieval]           -> Local (BM25 + scoring, no LLM)
  -----------------------------------------------------------
  [RAG Insight Extraction]  -> LM Studio  (call_llm_local)
  [Primary Diagnosis]       -> LM Studio  (call_llm_local)
  [Validation 3C3H]         -> LM Studio  (call_llm_local)
  -----------------------------------------------------------
  [Final Explanation]       -> Gemini     (call_llm_gemini_only)
  [Final Report — optional] -> Gemini     (call_llm_gemini_only)
  -----------------------------------------------------------
  Gemini is called ONLY for patient-facing output (steps 7–8).
  All reasoning and validation stages run on LM Studio.

Usage:
  python main.py
  python main.py --patient_id 141168
"""

import sys
import json
import random
import time
import argparse
from pathlib import Path
from datetime import datetime

random.seed(int(time.time() * 1000) % 2**32)
sys.path.insert(0, str(Path(__file__).parent))

from modules.gru_temporal import (
    EICUDataLoader, run_gru_module, estimate_confidence,
)
from modules.hybrid_rag import run_hybrid_rag, build_clinical_query
from modules.clinical_explainer import (
    build_clinical_explanation,
    print_clinical_explanation,
    expand_evidence,
    expand_assessment,
)
from modules.agentic_validation import (
    select_model,
    run_primary_agent,
    run_validation_agent,
    run_safety_rule_check,
    run_medagentsbench_eval,
)
from config import ICU_INPUT_SIZE

# Quality control layer
try:
    from modules.quality_control import PipelineAuditTrail, build_blocked_result
    _QC_AVAILABLE = True
except ImportError:
    _QC_AVAILABLE = False
    build_blocked_result = None
    class PipelineAuditTrail:           # minimal stub when QC not installed
        def __init__(self): self.stages = {}
        def record(self, *a, **kw): pass
        def format_for_llm(self): return ""
        def overall_quality(self): return "UNKNOWN"
        def to_dict(self): return {}
        def get(self, k): return {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_tcsv_description(gru_result: dict) -> str:
    """
    Describe TCSV output in clinical language, now including:
      - Phase trajectory (Hour 0–2 / 2–4 / 4–6 breakdown)
      - Ventilation status
    """
    import numpy as np
    info      = gru_result.get("patient_info", {})
    seq_len   = gru_result.get("seq_len", 0)
    tcsv      = gru_result.get("tcsv", [])
    mag       = float(abs(tcsv).mean()) if len(tcsv) > 0 else 0.0
    trend     = "escalating" if mag > 0.3 else "stable" if mag > 0.1 else "minimal"
    vent_str  = "mechanically ventilated" if info.get("ventilated") else "not ventilated"
    vent_h    = info.get("vent_hours", 0)
    icu_h     = info.get("icu_hours", "?")
    diag      = info.get("diagnosis_str", info.get("apache_score", "unspecified"))
    trajectory = gru_result.get("phase_trajectory", "")

    desc = (
        f"Temporal trajectory encoded from {seq_len} ICU observations "
        f"(11-feature vector including ventilator parameters). "
        f"Physiological trend: {trend} (TCSV mean magnitude: {mag:.4f}). "
        f"Patient: Age={info.get('age','?')}, Unit={info.get('unit_type','?')}, "
        f"ICU stay={icu_h}h, {vent_str} for {vent_h}h, "
        f"Diagnosis: {diag}. "
        f"GRU detected {'complex multi-phase evolution' if seq_len > 20 else 'limited temporal data'}."
    )
    if trajectory:
        desc += f"\n\nTemporal phase analysis:\n{trajectory}"
    return desc


def save_report(report: dict, output_dir: str = "outputs") -> str:
    Path(output_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_fname = f"{output_dir}/ta_cdss_report_{ts}.json"
    with open(json_fname, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report saved: {json_fname}")

    txt_fname = f"{output_dir}/ta_cdss_report_{ts}.txt"
    lines = []
    W = 65

    lines.append("=" * W)
    lines.append("  TA-CDSS -- Clinical Decision Support Report (Research-Grade)")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * W)

    p = report.get("pipeline", {})
    if isinstance(p.get("pipeline"), dict):
        p = p["pipeline"]

    gru  = p.get("gru",  {})
    exp  = p.get("explanation", {})
    val  = p.get("validation",  {})
    rag  = p.get("rag",  {})
    full = p.get("primary_assessment", "")
    conf = p.get("confidence", {})
    safety = p.get("safety_check", {})

    lines.append("")
    lines.append(f"PATIENT:     {gru.get('patient_id', 'N/A')}")
    lines.append(f"ICU STAY:    {gru.get('icu_hours','?')}h  |  Ventilated: {gru.get('ventilated','?')}  ({gru.get('vent_hours','?')}h)")
    lines.append(f"DIAGNOSIS:   {gru.get('diagnosis_str','unknown')}")
    lines.append(f"RISK LEVEL:  {exp.get('risk_label','?')}  —  {exp.get('risk_pct','?')}% predicted deterioration risk")
    lines.append(f"CONFIDENCE:  {conf.get('pct','?')}%  ({conf.get('label','unknown')})")
    lines.append(f"SUMMARY:     {exp.get('top_line','N/A')}")
    lines.append("")

    # Safety rules
    if safety and safety.get("triggered"):
        lines.append("-" * W)
        lines.append(f"SAFETY ALERTS  ({safety['n_triggered']} rules triggered  —  severity: {safety['severity']})")
        lines.append("-" * W)
        for rule in safety["triggered"]:
            lines.append(f"  !! {rule['meaning']}")
            lines.append(f"     {rule['feature']}={rule['value']} {rule['operator']} threshold {rule['threshold']}")
        lines.append("")

    lines.append("-" * W)
    lines.append("WHY NOW  (measured vital sign changes)")
    lines.append("-" * W)
    lines.append(exp.get("delta_why_now") or exp.get("why_now", "N/A"))
    lines.append("")

    # Temporal phases
    phases = p.get("temporal_phases", {})
    if phases:
        lines.append("-" * W)
        lines.append("TEMPORAL ANALYSIS  (Hour 0–2 / 2–4 / 4–6)")
        lines.append("-" * W)
        window_map = {"Early": "Hour 0–2", "Mid": "Hour 2–4", "Recent": "Hour 4–6"}
        for label, vals in phases.items():
            tag   = window_map.get(label, label)
            items = [f"{k}={v:.1f}" for k, v in vals.items() if v and v != 0]
            if items:
                lines.append(f"  {tag}: {', '.join(items)}")
        traj = p.get("phase_trajectory", "")
        if traj:
            lines.append("")
            lines.append(f"  Pattern: {traj.split(chr(10))[-1]}")
        lines.append("")

    lines.append("-" * W)
    lines.append("WHAT TO WATCH NEXT")
    lines.append("-" * W)
    for item in exp.get("watch_for", []):
        lines.append(f"  - {item}")
    lines.append("")

    lines.append("-" * W)
    lines.append("FULL CLINICAL ASSESSMENT")
    lines.append("-" * W)
    assessment = full if full else val.get("details", "N/A")
    assessment = assessment.replace("**", "").replace("__", "")
    lines.append(assessment)
    lines.append("")

    lines.append("-" * W)
    lines.append(f"EVIDENCE  ({rag.get('num_articles',0)} articles)")
    lines.append("-" * W)
    for ev in p.get("rag_evidence", []):
        lines.append(f"  [{ev.get('year','?')}] {ev.get('title','?')}")
        insight = ev.get("clinical_insight", "")
        if insight:
            lines.append(f"  Clinical insight: {insight}")
        abstract = ev.get("abstract", "")[:300]
        if abstract:
            lines.append(f"  {abstract}...")
        lines.append("")

    lines.append("-" * W)
    lines.append("VALIDATION")
    lines.append("-" * W)
    verdict = val.get("verdict", "?")
    concern = val.get("concern", "")
    lines.append(f"  Verdict: {verdict}")
    if concern:
        lines.append(f"  Concern: {concern}")
    val_text = val.get("details", "")
    if val_text:
        lines.append("")
        lines.append(val_text)
    lines.append("")

    actions = val.get("actions", [])
    if actions:
        lines.append("-" * W)
        lines.append("RECOMMENDED ACTIONS")
        lines.append("-" * W)
        for action in actions:
            lines.append(f"  {action}")
        lines.append("")

    if "medagentsbench" in report:
        b = report["medagentsbench"]
        lines.append("-" * W)
        lines.append("MEDAGENTSBENCH RESULTS")
        lines.append("-" * W)
        lines.append(f"  Score: {b['correct']}/{b['total']} = {b['accuracy']:.1f}%")
        lines.append("")
        for case in b.get("cases", []):
            status = "CORRECT" if case["correct"] else "INCORRECT"
            lines.append(f"  Case {case['case']}: {status}")
            lines.append(f"  Q: {case['question'][:100]}")
            lines.append(f"  A: {case['agent_answer'][:200]}")
            lines.append("")

    lines.append("=" * W)
    lines.append("  END OF REPORT")
    lines.append("=" * W)

    with open(txt_fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"TXT report saved: {txt_fname}")
    return json_fname


def print_banner():
    print("=" * 65)
    print("  TA-CDSS -- Temporal-Aware Clinical Decision Support System")
    print("  Research-Grade: 11 features | Phase analysis | Safety rules")
    print("  GRU Engine + Hybrid RAG + Gemini + Agentic Validation")
    print("=" * 65)


def print_dashboard(explanation: dict, gru_result: dict, safety_check: dict = None):
    """
    FIX 8: ASCII patient risk dashboard with vital sign mini-graphs
    and a timeline chart showing the observation window phases.

    Replaces: console output only
    Adds:     risk gauge, vital sparkline table, phase timeline
    """
    W      = 65
    phases = gru_result.get("temporal_phases", {})
    risk   = explanation["risk_score"]
    conf   = explanation.get("confidence", {})

    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    YELLOW= "\033[93m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    CYAN  = "\033[96m"

    risk_colour = RED if risk >= 0.70 else YELLOW if risk >= 0.40 else GREEN

    print(f"\n{CYAN}{'━'*W}{RESET}")
    print(f"{BOLD}  PATIENT RISK DASHBOARD{RESET}")
    print(f"{CYAN}{'━'*W}{RESET}")

    # ── Risk gauge ──────────────────────────────────────────────────────────
    gauge_width = 40
    filled      = int(risk * gauge_width)
    bar         = "█" * filled + "░" * (gauge_width - filled)
    print(f"\n  Risk   [{risk_colour}{bar}{RESET}] {int(risk*100)}%  {explanation['risk_label']}")

    # Confidence bar
    c_pct   = conf.get("pct", 0)
    c_col   = GREEN if c_pct >= 60 else YELLOW if c_pct >= 35 else RED
    c_fill  = int(c_pct / 100 * gauge_width)
    c_bar   = "█" * c_fill + "░" * (gauge_width - c_fill)
    print(f"  Conf.  [{c_col}{c_bar}{RESET}] {c_pct}%   {conf.get('label','unknown')}")

    # ── Safety alerts summary ───────────────────────────────────────────────
    if safety_check and safety_check.get("triggered"):
        n   = safety_check["n_triggered"]
        sev = safety_check["severity"]
        sev_col = RED if sev == "HIGH" else YELLOW if sev == "MODERATE" else GREEN
        print(f"\n  {RED}!! {n} safety rule(s) triggered  —  severity: {sev_col}{sev}{RESET}")
        for rule in safety_check["triggered"][:3]:
            print(f"     - {rule['meaning']}")

    # ── Vital sign mini-table from phases ──────────────────────────────────
    VITALS_TO_SHOW = [
        ("SpO2",             "%",     50,  100, 94,  92),
        ("Respiration Rate", "br/min", 0,   60, 25,  30),
        ("Heart Rate",       "bpm",   20,  200, 100, 120),
        ("Systolic BP",      "mmHg",  40,  200, 90,  80),
        ("Lactate",          "mmol/L", 0,   15, 2.0, 4.0),
    ]

    early  = phases.get("Early",  {})
    recent = phases.get("Recent", {})

    if early or recent:
        print(f"\n  {'Vital':<20} {'Early':>8} {'Recent':>8}  {'Trend':>6}  Status")
        print(f"  {'─'*20} {'─'*8} {'─'*8}  {'─'*6}  {'─'*16}")
        for (vname, unit, vmin, vmax, warn_thr, crit_thr) in VITALS_TO_SHOW:
            e_val = early.get(vname,  0)
            r_val = recent.get(vname, 0)
            if e_val == 0 and r_val == 0:
                continue
            e_str = f"{e_val:.1f}" if e_val else "  —  "
            r_str = f"{r_val:.1f}" if r_val else "  —  "

            # Trend arrow
            if e_val and r_val:
                delta = r_val - e_val
                if abs(delta) < 1:   arrow = "  →  "
                elif delta > 0:      arrow = f" ↑{delta:+.1f}"
                else:                arrow = f" ↓{delta:+.1f}"
            else:
                arrow = "     "

            # Status colour based on critical threshold
            # For SpO2/BP: lower is worse; for RR/HR/Lactate: higher is worse
            check_val = r_val if r_val else e_val
            if vname in ("SpO2", "Systolic BP"):
                if check_val and check_val < crit_thr:   status = f"{RED}CRITICAL{RESET}"
                elif check_val and check_val < warn_thr:  status = f"{YELLOW}WARNING{RESET} "
                else:                                      status = f"{GREEN}OK{RESET}      "
            else:
                if check_val and check_val > crit_thr:   status = f"{RED}CRITICAL{RESET}"
                elif check_val and check_val > warn_thr:  status = f"{YELLOW}WARNING{RESET} "
                else:                                      status = f"{GREEN}OK{RESET}      "

            print(f"  {vname:<20} {e_str:>8} {r_str:>8}  {arrow}  {status}")

    # ── Phase timeline ──────────────────────────────────────────────────────
    if phases:
        print(f"\n  TIMELINE  (last 6 hours)")
        print(f"  {'─'*W}")
        labels  = {"Early": "0–2h", "Mid": "2–4h", "Recent": "4–6h"}
        colours = {"Early": GREEN,   "Mid": YELLOW,  "Recent": RED}
        has_deterioration = False

        for label, ph_vals in phases.items():
            tag = labels.get(label, label)
            col = colours.get(label, RESET)
            spo2 = ph_vals.get("SpO2", 0)
            rr   = ph_vals.get("Respiration Rate", 0)

            flag = ""
            if spo2 and spo2 < 92:   flag += " [SpO2 LOW]"
            if rr   and rr   > 28:   flag += " [RR HIGH]"
            if flag:
                has_deterioration = True

            val_items = []
            for vn in ["SpO2", "Respiration Rate", "Heart Rate", "Lactate"]:
                v = ph_vals.get(vn, 0)
                if v:
                    val_items.append(f"{vn}={v:.0f}")

            vals_str = ", ".join(val_items) if val_items else "no data"
            print(f"  {col}[{tag}]{RESET}  {vals_str}{RED}{flag}{RESET}")

        if has_deterioration:
            print(f"\n  {RED}!! Deterioration detected in recent phase — review urgently{RESET}")

    print(f"\n{CYAN}{'━'*W}{RESET}\n")


def print_section(title: str):
    print(f"\n{'-' * 65}")
    print(f"  {title}")
    print(f"{'-' * 65}")


def interactive_expand(explanation: dict):
    while True:
        try:
            choice = input("\n  [E] Expand evidence  [A] Full assessment  [Q] Continue: ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print("\n  (Skipping interactive expand)")
            break
        if choice == "E":
            expand_evidence(explanation)
        elif choice == "A":
            expand_assessment(explanation)
        elif choice in ("Q", ""):
            break
        else:
            print("  Type E, A, or Q.")


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(patient_id: int = None) -> dict:
    """Run the full research-grade TA-CDSS pipeline for a single patient."""
    report = {"timestamp": datetime.now().isoformat(), "pipeline": {}}

    # Initialise audit trail — threads through all 8 steps
    audit = PipelineAuditTrail()

    # ── Step 1: Load eICU Data ─────────────────────────────────────────────────
    print_section("STEP 1 -- Loading eICU Data")
    loader  = EICUDataLoader(data_dir="data/eicu")
    success = loader.load()

    if not success:
        print("\nNo eICU data found.")
        print("   Place eICU CSV files in: data/eicu/")
        print("   Required: patient.csv, vitalPeriodic.csv")
        print("   Optional: lab.csv, treatment.csv, diagnosis.csv, respiratoryCharting.csv")
        sys.exit(1)

    if patient_id is None and loader.patients is not None:
        pid_col    = "patientunitstayid" if "patientunitstayid" in loader.patients.columns else loader.patients.columns[0]
        all_ids    = loader.patients[pid_col].dropna().unique().tolist()
        candidates = all_ids

        if loader.vitals is not None:
            vpid_col   = "patientunitstayid" if "patientunitstayid" in loader.vitals.columns else loader.vitals.columns[0]
            rich_ids   = loader.vitals[vpid_col].value_counts()
            rich_ids   = rich_ids[rich_ids >= 20].index.tolist()
            candidates = [p for p in all_ids if p in rich_ids] or all_ids

        patient_id = int(random.choice(candidates))
        print(f"\n  Auto-selected patient ID: {patient_id} (from {len(candidates)} rich-data patients)")

    # ── Step 2: GRU Temporal Reasoning ────────────────────────────────────────
    print_section("STEP 2 -- GRU Temporal Reasoning (11 features)")
    gru_result = run_gru_module(patient_id, loader)
    risk_score = gru_result["risk_score"]

    # Record data quality from GRU module
    data_quality = gru_result.get("data_quality", {"tier": "UNKNOWN", "warnings": []})
    audit.record("data_quality", data_quality)

    # ── HARD DATA GATE ─────────────────────────────────────────────────────────
    # If the gate is triggered the entire pipeline is aborted here.
    # No LLM calls, no RAG, no risk score — a structured refusal is returned.
    if data_quality.get("gate_blocked", False):
        reasons = data_quality.get("gate_reasons", ["Insufficient data"])
        print(f"\n   !! HARD DATA GATE TRIGGERED — pipeline blocked")
        for r in reasons:
            print(f"      {r}")
        print("   !! No clinical assessment will be generated.")
        print("   !! Add complete patient data and re-run.\n")
        if build_blocked_result:
            blocked = build_blocked_result(patient_id, data_quality)
            report["pipeline"] = blocked
            report["blocked"]  = True
            return report
        else:
            sys.exit(
                f"\nPipeline blocked — insufficient data:\n"
                + "\n".join(f"  - {r}" for r in reasons)
            )

    if data_quality.get("warnings"):
        print(f"\n   [DataQC] tier={data_quality['tier']}")
        for w in data_quality["warnings"]:
            print(f"   [DataQC] WARN: {w}")

    report["pipeline"]["gru"] = {
        "patient_id":     patient_id,
        "seq_len":        gru_result["seq_len"],
        "risk_score":     risk_score,
        "clinical_summary": gru_result["clinical_summary"],
        "icu_hours":      gru_result["patient_info"].get("icu_hours"),
        "ventilated":     gru_result["patient_info"].get("ventilated"),
        "vent_hours":     gru_result["patient_info"].get("vent_hours"),
        "diagnosis_str":  gru_result["patient_info"].get("diagnosis_str"),
        "data_quality":   data_quality,
    }
    report["pipeline"]["temporal_phases"]  = gru_result.get("temporal_phases", {})
    report["pipeline"]["phase_trajectory"] = gru_result.get("phase_trajectory", "")

    # ── Step 3: Safety Rule Check (pre-LLM gate) ───────────────────────────────
    print_section("STEP 3 -- Safety Rule Check")
    safety_check = run_safety_rule_check(gru_result.get("temporal_phases", {}))
    audit.record("safety", {
        "tier": "FAIL" if safety_check["severity"] == "HIGH" else
                "WARN" if safety_check["n_triggered"] > 0 else "PASS",
        "n_triggered": safety_check["n_triggered"],
        "severity":    safety_check["severity"],
    })
    report["pipeline"]["safety_check"] = {
        "n_triggered": safety_check["n_triggered"],
        "severity":    safety_check["severity"],
        "triggered":   safety_check["triggered"],
    }

    # ── Step 4: Hybrid RAG (clinically-specific query) ─────────────────────────
    print_section("STEP 4 -- Hybrid PubMed RAG Retrieval")
    rag_result = run_hybrid_rag(
        clinical_query   = "",
        patient_summary  = gru_result["clinical_summary"],
        patient_info     = gru_result["patient_info"],
        temporal_phases  = gru_result["temporal_phases"],
        top_k            = 8,
        use_cache        = True,
        apply_quality_gate = True,
    )

    # Log RAG quality gate result
    rag_gate = rag_result.get("quality_gate", "UNKNOWN")
    audit.record("rag", {
        "tier":          rag_gate,
        "confidence":    rag_result["confidence"],
        "fallback_used": rag_result.get("fallback_used", False),
        "n_articles":    len(rag_result.get("evidence", [])),
        "warnings":      [rag_result.get("quality_gate_reason", "")] if rag_gate == "FAIL" else [],
    })
    if rag_result.get("fallback_used"):
        print(f"\n   [RAGGate] Fallback evidence block in use — RAG quality below threshold")
    report["pipeline"]["rag"] = {
        "query":         rag_result["query"],
        "confidence":    rag_result["confidence"],
        "num_articles":  len(rag_result["evidence"]),
        "quality_gate":  rag_gate,
        "fallback_used": rag_result.get("fallback_used", False),
    }
    report["pipeline"]["rag_evidence"] = rag_result["evidence"]

    # ── Step 5: Confidence Score ───────────────────────────────────────────────
    print_section("STEP 5 -- Confidence Score Estimation")
    confidence = estimate_confidence(
        gru_result,
        rag_confidence=rag_result["confidence"],
        data_quality=data_quality,
    )
    report["pipeline"]["confidence"] = confidence
    print(f"   Overall confidence: {confidence['pct']}%  ({confidence['label']})")
    print(f"   Data richness: {int(confidence['data_richness']*100)}%  |  "
          f"Temporal coverage: {int(confidence['temporal_cov']*100)}%  |  "
          f"RAG quality: {int(confidence['rag_quality']*100)}%")

    if confidence["data_richness"] < 0.5:
        print(f"\n   NOTE: Data richness is {int(confidence['data_richness']*100)}% — this is expected when using")
        print(f"      the eICU demo subset (~2,500 patients). The full eICU Collaborative")
        print(f"      Research Database (~200k admissions, 300+ hospitals) would give")
        print(f"      richer records and higher confidence scores.")

    # ── Step 6: Primary Diagnostic Agent ──────────────────────────────────────
    print_section("STEP 6 -- Primary Diagnostic Agent")
    tcsv_desc      = build_tcsv_description(gru_result)
    primary_output = run_primary_agent(
        patient_summary  = gru_result["clinical_summary"],
        tcsv_description = tcsv_desc,
        evidence_text    = rag_result["top_evidence"],
        safety_check     = safety_check,
        risk_score       = risk_score,
        data_quality     = data_quality,
        audit_trail      = audit,
    )
    report["pipeline"]["primary_assessment"] = primary_output

    # Log output quality from audit (was recorded inside run_primary_agent)
    oq = audit.get("output_quality")
    if oq:
        print(f"\n   [OutputQC] tier={oq.get('tier')}  overall={oq.get('overall', 0):.3f}")

    # ── Step 7: Clinical Explanation ──────────────────────────────────────────
    print_section("STEP 7 -- Building Clinician Explanation")
    explanation = build_clinical_explanation(
        gru_result         = gru_result,
        risk_score         = risk_score,
        primary_assessment = primary_output,
        evidence           = rag_result["evidence"],
        safety_check       = safety_check,
        confidence         = confidence,
        window_hours       = 6,
    )
    report["pipeline"]["explanation"] = {
        "risk_label":    explanation["risk_label"],
        "risk_pct":      explanation["risk_pct"],
        "top_line":      explanation["top_line"],
        "why_now":       explanation["why_now"],
        "delta_why_now": explanation["delta_why_now"],
        "watch_for":     explanation["watch_for"],
    }

    print_clinical_explanation(explanation)
    print_dashboard(explanation, gru_result, safety_check)
    interactive_expand(explanation)

    # ── Step 8: Agentic Validation ────────────────────────────────────────────
    print_section("STEP 8 -- 3C3H Agentic Validation")
    validation = run_validation_agent(
        primary_output  = primary_output,
        evidence_text   = rag_result["top_evidence"],
        patient_summary = gru_result["clinical_summary"],
        safety_check    = safety_check,
    )
    report["pipeline"]["validation"] = {
        "verdict": validation["verdict"],
        "score":   validation["score"],
        "concern": validation.get("concern", ""),
        "details": validation["validation_text"],
        "actions": validation.get("actions", []),
    }

    # ── Audit trail summary ────────────────────────────────────────────────────
    report["pipeline"]["audit_trail"]    = audit.to_dict()
    report["pipeline"]["overall_quality"]= audit.overall_quality()

    overall_q = audit.overall_quality()
    print(f"\n   Pipeline overall quality: {overall_q}")
    if overall_q == "POOR":
        print("   WARNING: One or more stages produced low-quality output.")
        print("   This report should NOT be used for clinical decisions without senior review.")
    elif overall_q == "DEGRADED":
        print("   NOTE: Some stages produced degraded output (sparse data or low RAG confidence).")
        print("   Treat confidence scores conservatively.")

    return report


def run_benchmark() -> dict:
    print_section("MEDAGENTSBENCH EVALUATION")
    try:
        n = input("\n   How many benchmark cases? (1-10, default=3): ").strip()
    except (KeyboardInterrupt, EOFError):
        n = "3"
    n = int(n) if n.isdigit() else 3
    return run_medagentsbench_eval(n_samples=n)


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TA-CDSS Research-Grade Pipeline")
    parser.add_argument("--patient_id", type=int, default=None)
    args = parser.parse_args()

    print_banner()
    select_model()

    print("\n\nWhat would you like to run?")
    print("   [1] Full pipeline (single patient)")
    print("   [2] MedAgentsBench evaluation only")
    print("   [3] Both")

    try:
        choice = input("\n   Enter choice (1/2/3): ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "1"

    report = {"run_time": datetime.now().isoformat()}

    if choice in ("1", "3"):
        pid = args.patient_id
        if pid is None:
            try:
                inp = input("\n   Enter patient ID (or Enter to auto-select): ").strip()
            except (KeyboardInterrupt, EOFError):
                inp = ""
            pid = int(inp) if inp.isdigit() else None
        pipeline_report = run_pipeline(patient_id=pid)
        report["pipeline"] = pipeline_report

    if choice in ("2", "3"):
        report["medagentsbench"] = run_benchmark()

    # ── Final Summary ─────────────────────────────────────────────────────────
    print_section("FINAL SUMMARY")

    if "pipeline" in report:
        p        = report["pipeline"].get("pipeline", report["pipeline"])
        gru_info = p.get("gru",        {})
        val_info = p.get("validation", {})
        exp_info = p.get("explanation",{})
        conf     = p.get("confidence", {})
        safety   = p.get("safety_check", {})

        print(f"   Patient:      {gru_info.get('patient_id','N/A')}")
        print(f"   ICU stay:     {gru_info.get('icu_hours','?')}h  |  "
              f"Ventilated: {gru_info.get('ventilated','?')} ({gru_info.get('vent_hours','?')}h)")
        print(f"   Diagnosis:    {gru_info.get('diagnosis_str','unknown')}")
        print(f"   Risk:         {exp_info.get('risk_label','?')}  —  "
              f"{exp_info.get('risk_pct','?')}% predicted deterioration risk")
        print(f"   Confidence:   {conf.get('pct','?')}%  ({conf.get('label','unknown')})")

        if safety and safety.get("n_triggered", 0) > 0:
            print(f"   Safety:       {safety['n_triggered']} rule(s) triggered  "
                  f"(severity: {safety['severity']})")

        print(f"   Summary:      {exp_info.get('top_line','N/A')[:80]}")
        print(f"   Evidence:     {p.get('rag',{}).get('num_articles',0)} articles")

        verdict = val_info.get("verdict", "?")
        concern = val_info.get("concern", "")
        print(f"   Validation:   {verdict}")
        if concern:
            print(f"   Concern:      {concern}")

        actions = val_info.get("actions", [])
        if actions:
            print("\n   WHAT TO DO NEXT:")
            for action in actions:
                print(f"      {action}")

    if "medagentsbench" in report:
        b = report["medagentsbench"]
        print(f"   MedAgentsBench: {b['correct']}/{b['total']} = {b['accuracy']:.1f}%")

    save_report(report)
    print("\nTA-CDSS run complete.\n")


if __name__ == "__main__":
    main()
