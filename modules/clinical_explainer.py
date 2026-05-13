"""
modules/clinical_explainer.py -- Clinician-Friendly Explanation Engine (Research-Grade)

Key improvements over prototype:
  1. WHY NOW uses actual delta values: "SpO2 declined 96→91%,
     RR increased 18→27 br/min" (was: generic threshold list)
  2. Phase trajectory shown in console: Hour 0–2 / 2–4 / 4–6 breakdown
  3. Confidence score displayed prominently (new field from GRU module)
  4. Risk score presented as percentage with calibrated band description
  5. Evidence now shows clinical_insight sentence instead of raw abstract
  6. Safety rule violations surfaced in the WHY NOW section

LLM Routing (Stable Architecture):
  generate_narrative (Final Explanation) -> call_llm_gemini_only
  This is one of ONLY TWO Gemini call sites in the entire pipeline.
  All upstream stages (RAG insight, diagnosis, validation) use LM Studio.
"""

from __future__ import annotations

import re
import math
from datetime import datetime
from modules.llm_client import call_llm_gemini_only, build_messages


# ── Sparkline (unchanged) ──────────────────────────────────────────────────────

SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

def _sparkline(values: list, width: int = 20) -> str:
    if not values or len(values) < 2:
        return "-- (insufficient data)"
    if len(values) > width:
        step   = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    mn, mx = min(values), max(values)
    rng    = mx - mn
    if rng == 0:
        return SPARK_CHARS[3] * len(values) + "  (flat)"
    def _char(v):
        idx = int((v - mn) / rng * (len(SPARK_CHARS) - 1))
        return SPARK_CHARS[min(idx, len(SPARK_CHARS) - 1)]
    return "".join(_char(v) for v in values) + f"  min={mn:.1f} max={mx:.1f}"


def build_trend_sparklines(gru_result: dict) -> dict:
    features_raw  = gru_result.get("features", [])
    if not features_raw:
        return {}
    feature_names = [
        "Heart Rate", "Respiration Rate", "SpO2", "Temperature",
        "Systolic BP", "Diastolic BP", "FiO2", "PEEP",
        "Tidal Volume", "PaO2/FiO2", "Lactate",
    ]
    sparklines = {}
    for fi, name in enumerate(feature_names):
        series = []
        for timestep in features_raw:
            if fi < len(timestep) and timestep[fi] != 0.0:
                series.append(float(timestep[fi]))
        if len(series) >= 3:
            sparklines[name] = _sparkline(series)
    return sparklines


# ── Threshold detection (unchanged logic, wider feature set) ──────────────────

from config import CLINICAL_THRESHOLDS

def detect_threshold_crossings(gru_result: dict, window_hours: int = 6) -> list:
    features_raw = gru_result.get("features",   [])
    timestamps   = gru_result.get("timestamps", [])
    if not features_raw or not timestamps:
        return []

    feature_names = [
        "Heart Rate", "Respiration Rate", "SpO2", "Temperature",
        "Systolic BP", "Diastolic BP", "FiO2", "PEEP",
        "Tidal Volume", "PaO2_FiO2", "Lactate",
    ]
    events     = []
    max_offset = max(timestamps)
    cutoff     = max_offset - (window_hours * 60)

    for fi, name in enumerate(feature_names):
        thresh_name = name.replace("/", "_")
        thresh = CLINICAL_THRESHOLDS.get(thresh_name) or CLINICAL_THRESHOLDS.get(name)
        if not thresh:
            continue

        series = []
        for ti, timestep in enumerate(features_raw):
            if fi < len(timestep) and timestep[fi] != 0.0:
                series.append((timestamps[ti] if ti < len(timestamps) else ti * 60, timestep[fi]))

        if len(series) < 2:
            continue

        for i in range(1, len(series)):
            t_prev, v_prev = series[i - 1]
            t_curr, v_curr = series[i]
            if t_curr < cutoff:
                continue
            mins_ago = int((max_offset - t_curr) / 60 * 60)

            if "low" in thresh and v_prev >= thresh["low"] and v_curr < thresh["low"]:
                events.append({
                    "feature":   name, "direction": "dropped below",
                    "threshold": thresh["low"], "value": round(v_curr, 1),
                    "unit":      thresh.get("unit", ""), "offset_min": int(t_curr),
                    "mins_ago":  mins_ago, "prev_value": round(v_prev, 1),
                })
            elif "high" in thresh and v_prev <= thresh["high"] and v_curr > thresh["high"]:
                events.append({
                    "feature":   name, "direction": "rose above",
                    "threshold": thresh["high"], "value": round(v_curr, 1),
                    "unit":      thresh.get("unit", ""), "offset_min": int(t_curr),
                    "mins_ago":  mins_ago, "prev_value": round(v_prev, 1),
                })

    return events


def _format_crossing_time(event: dict) -> str:
    offset_min = event.get("offset_min", 0)
    hrs  = offset_min // 60
    mins = offset_min % 60
    return f"ICU hour {hrs}:{mins:02d}"


# ── Risk label ─────────────────────────────────────────────────────────────────

def _risk_label(risk_score: float) -> tuple:
    from config import RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD
    if risk_score >= RISK_HIGH_THRESHOLD:
        return "HIGH", "HIGH"
    elif risk_score >= RISK_MEDIUM_THRESHOLD:
        return "MEDIUM", "MEDIUM"
    else:
        return "LOW", "LOW"


# ── Delta-based WHY NOW builder (NEW) ─────────────────────────────────────────

def build_delta_why_now(gru_result: dict, safety_check: dict = None) -> str:
    """
    Build a specific WHY NOW string using actual value deltas between
    early and recent temporal phases.

    Example output:
      "SpO2 declined 96→91% (−5%), RR increased 18→27 br/min (+9),
       Lactate rose 1.2→2.8 mmol/L. Pattern suggests respiratory
       distress during ventilator weaning."

    Prototype: only listed which threshold was crossed, no delta.
    """
    phases = gru_result.get("temporal_phases", {})
    early  = phases.get("Early",  {})
    recent = phases.get("Recent", {})

    if not early or not recent:
        return gru_result.get("phase_trajectory", "Recent trend change detected.")

    delta_items = []
    checks = [
        ("SpO2",             "declined",  -3,  "%"),
        ("Respiration Rate", "increased",  4,  "br/min"),
        ("Heart Rate",       "increased", 15,  "bpm"),
        ("Systolic BP",      "declined", -15,  "mmHg"),
        ("Lactate",          "increased",  0.5,"mmol/L"),
        ("PaO2_FiO2",        "declined", -50,  "mmHg"),
        ("FiO2",             "increased",  0.1,"fraction"),
    ]

    for fname, direction, min_delta, unit in checks:
        e = early.get(fname, 0)
        r = recent.get(fname, 0)
        if e == 0 or r == 0:
            continue
        delta = r - e
        threshold_met = (delta < min_delta) if direction == "declined" else (delta > min_delta)
        if threshold_met:
            sign   = "−" if delta < 0 else "+"
            d_str  = f"{sign}{abs(delta):.1f}"
            delta_items.append(f"{fname} {direction} {e:.1f}→{r:.1f} {unit} ({d_str})")

    # Add safety rule violations
    safety_items = []
    if safety_check:
        for rule in safety_check.get("triggered", [])[:2]:
            safety_items.append(f"{rule['meaning']}")

    if not delta_items and not safety_items:
        return gru_result.get("phase_trajectory", "Recent trend change detected.")

    parts = []
    if delta_items:
        parts.append(", ".join(delta_items))
    if safety_items:
        parts.append(". Safety alerts: " + "; ".join(safety_items))

    # Classify pattern
    combined = " ".join(delta_items).lower()
    if "spo2" in combined and "respiration" in combined:
        parts.append(". Pattern suggests progressive respiratory distress.")
    elif "lactate" in combined and "systolic" in combined:
        parts.append(". Pattern suggests haemodynamic deterioration.")
    elif "heart rate" in combined:
        parts.append(". Pattern suggests increasing physiological stress.")

    return "".join(parts)


# ── LLM narrative generation ───────────────────────────────────────────────────

EXPLAINER_SYSTEM = """You are a clinical decision support assistant for ICU physicians.
Write exactly three labeled sections. Be direct and specific -- name actual vitals and values.

TOP_LINE: (one sentence naming the specific vital signs driving this alert and the time window)
WHY_NOW: (one or two sentences on what threshold was crossed or what rate of change triggered this, with timing)
WATCH_FOR:
- (specific next signal 1, e.g. SpO2 dropping below 90%)
- (specific next signal 2)
- (specific next signal 3)
- (specific next signal 4)
- (specific next signal 5)

Do not include evidence citations. Do not add any other sections."""


def generate_narrative(
    patient_summary:    str,
    risk_score:         float,
    crossings:          list,
    primary_assessment: str,
    delta_why_now:      str = "",
    window_hours:       int = 6,
) -> dict:
    risk_label, _ = _risk_label(risk_score)

    if crossings:
        lines = []
        for ev in crossings[:4]:
            lines.append(
                f"- {ev['feature']} {ev['direction']} {ev['threshold']} {ev.get('unit','')} "
                f"(was {ev.get('prev_value','?')}, now {ev['value']}) at {_format_crossing_time(ev)}"
            )
        crossing_text = "Threshold crossings detected:\n" + "\n".join(lines)
    else:
        crossing_text = "No clear threshold crossings detected in this window."

    delta_context = f"\nMeasured vital sign changes:\n{delta_why_now}\n" if delta_why_now else ""

    user_content = (
        f"Patient context: {patient_summary}\n\n"
        f"Risk level: {risk_label} (score: {risk_score:.2f})\n"
        f"Observation window: last {window_hours} hours\n\n"
        f"{crossing_text}\n"
        f"{delta_context}\n"
        f"Clinical assessment summary:\n{primary_assessment[:600]}\n\n"
        f"Write the three labeled sections now."
    )

    messages = build_messages(EXPLAINER_SYSTEM, user_content)
    raw      = call_llm_gemini_only(messages, temperature=0.2, max_tokens=1200)  # Gemini ONLY — Final Explanation
    return _parse_labeled_response(raw)


def _parse_labeled_response(raw: str) -> dict:
    normalised = re.sub(r'(?i)top[\s_-]?line\s*:', 'TOP_LINE:', raw)
    normalised = re.sub(r'(?i)why[\s_-]?now\s*:',  'WHY_NOW:',  normalised)
    normalised = re.sub(r'(?i)watch[\s_-]?for\s*:', 'WATCH_FOR:', normalised)
    normalised = re.sub(r'(?i)\bwatch\s*:',          'WATCH_FOR:', normalised)

    top_line  = ""
    why_now   = ""
    watch_for = []

    section_pattern = re.compile(r'(TOP_LINE:|WHY_NOW:|WATCH_FOR:)', re.IGNORECASE)
    parts = section_pattern.split(normalised)

    current_label = None
    for part in parts:
        part_upper = part.strip().upper()
        if part_upper in ('TOP_LINE:', 'WHY_NOW:', 'WATCH_FOR:'):
            current_label = part_upper.rstrip(':')
        elif current_label == 'TOP_LINE':
            lines = [l.strip() for l in part.splitlines() if l.strip()]
            if lines:
                top_line = lines[0]
            current_label = None
        elif current_label == 'WHY_NOW':
            lines = [
                l.strip() for l in part.splitlines()
                if l.strip() and not l.strip().startswith(('-', '*', '•'))
                and not re.match(r'(?i)(top_line|why_now|watch_for):', l)
            ]
            why_now = ' '.join(lines).strip()
            current_label = None
        elif current_label == 'WATCH_FOR':
            for line in part.splitlines():
                stripped = line.strip()
                if stripped.startswith(('-', '*', '•')):
                    item = stripped.lstrip('-*•').strip()
                    if len(item) > 5:
                        watch_for.append(item)
            current_label = None

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    if not top_line:
        top_line = next(
            (l for l in lines if len(l) > 20 and not l.startswith(('-', '*', '•'))),
            lines[0] if lines else "Risk level updated based on recent vitals.",
        )

    if not why_now:
        candidates = [
            l for l in lines
            if len(l) > 20 and l != top_line and not l.startswith(('-', '*', '•'))
        ]
        why_now = candidates[0] if candidates else ""

    if not watch_for:
        watch_for = [
            l.lstrip('-*•').strip()
            for l in raw.splitlines()
            if l.strip().startswith(('-', '*', '•')) and len(l.strip()) > 10
        ][:5]

    if not watch_for:
        sentences = [s.strip() for s in re.split(r'[.;]', raw) if len(s.strip()) > 15]
        watch_for = sentences[2:5]

    # FIX 6: If LLM still returned nothing, generate rule-based monitoring items
    # so WHAT TO WATCH NEXT is never empty in the report.
    if not watch_for:
        watch_for = _rule_based_watch_for(raw)

    return {
        "top_line":  top_line.strip(),
        "why_now":   why_now.strip(),
        "watch_for": watch_for[:5],
    }


def _rule_based_watch_for(context_hint: str = "") -> list:
    """
    FIX 6: Generate monitoring items from clinical thresholds when the LLM
    fails to return a WATCH_FOR list. Covers the most common ICU deterioration
    signals so the section is never blank.
    """
    ctx = context_hint.lower()
    items = []

    # Respiratory
    if any(w in ctx for w in ["spo2", "respir", "oxygen", "ards", "ventilat", "breath"]):
        items.append("SpO2 dropping below 92% — indicates worsening hypoxaemia")
        items.append("Respiratory rate rising above 30 br/min — respiratory distress marker")
        items.append("FiO2 requirement increasing above 60% — oxygen toxicity threshold")
    else:
        items.append("SpO2 < 94% — early hypoxaemia requiring oxygen review")
        items.append("Respiratory rate > 25 br/min — early respiratory stress")

    # Haemodynamic
    if any(w in ctx for w in ["lactate", "sepsis", "shock", "hypotens", "pressure"]):
        items.append("Lactate rising above 4.0 mmol/L — severe tissue hypoperfusion")
        items.append("Systolic BP falling below 80 mmHg — vasopressor threshold")
    else:
        items.append("Heart rate > 120 bpm or < 50 bpm — haemodynamic instability")

    # Catch-all if nothing matched
    if not items:
        items = [
            "SpO2 < 92% — critical hypoxaemia",
            "Respiratory rate > 30 br/min — respiratory failure threshold",
            "Lactate > 4.0 mmol/L — shock / severe hypoperfusion",
            "Systolic BP < 80 mmHg — vasopressor review required",
            "Heart rate > 130 bpm — haemodynamic stress",
        ]

    return items[:5]


# ── Master explainer ───────────────────────────────────────────────────────────

def build_clinical_explanation(
    gru_result:         dict,
    risk_score:         float,
    primary_assessment: str,
    evidence:           list,
    safety_check:       dict = None,
    confidence:         dict = None,
    window_hours:       int  = 6,
) -> dict:
    risk_label, risk_level = _risk_label(risk_score)
    crossings   = detect_threshold_crossings(gru_result, window_hours)
    sparklines  = build_trend_sparklines(gru_result)
    delta_why   = build_delta_why_now(gru_result, safety_check)

    print(f"\n   Generating clinician explanation (risk={risk_label})...")
    narrative = generate_narrative(
        patient_summary    = gru_result.get("clinical_summary", ""),
        risk_score         = risk_score,
        crossings          = crossings,
        primary_assessment = primary_assessment,
        delta_why_now      = delta_why,
        window_hours       = window_hours,
    )

    # Prefer delta-based why_now over generic LLM output when available
    why_now = delta_why if len(delta_why) > 30 else narrative.get("why_now", "Recent trend change detected.")

    return {
        "risk_label":          risk_label,
        "risk_level":          risk_level,
        "risk_score":          round(risk_score, 3),
        "risk_pct":            int(risk_score * 100),
        "top_line":            narrative.get("top_line", "Risk level updated based on recent vitals."),
        "why_now":             why_now,
        "delta_why_now":       delta_why,
        "sparklines":          sparklines,
        "threshold_crossings": crossings,
        "watch_for":           narrative.get("watch_for", []),
        "evidence":            evidence,
        "primary_assessment":  primary_assessment,
        "temporal_phases":     gru_result.get("temporal_phases", {}),
        "phase_trajectory":    gru_result.get("phase_trajectory", ""),
        "safety_check":        safety_check,
        "confidence":          confidence or {},
    }


# ── Console renderer (improved) ────────────────────────────────────────────────

def print_clinical_explanation(exp: dict):
    W   = 65
    sep = "-" * W

    risk_pct  = exp.get("risk_pct", int(exp["risk_score"] * 100))
    conf      = exp.get("confidence", {})
    conf_pct  = conf.get("pct", "?")
    conf_lbl  = conf.get("label", "unknown")

    print(f"\n{'=' * W}")
    print(f"  RISK: {exp['risk_label']}  —  {risk_pct}% predicted deterioration risk")
    print(f"  MODEL CONFIDENCE: {conf_pct}%  ({conf_lbl})")
    if conf:
        print(f"  Data richness: {int(conf.get('data_richness',0)*100)}%  |  "
              f"Temporal coverage: {int(conf.get('temporal_cov',0)*100)}%  |  "
              f"Evidence quality: {int(conf.get('rag_quality',0)*100)}%")
    print(f"{'=' * W}")
    print(f"\n  {exp['top_line']}")

    # Safety alerts block (NEW)
    safety = exp.get("safety_check", {})
    if safety and safety.get("triggered"):
        print(f"\n{sep}")
        print(f"  !! SAFETY ALERTS  ({safety['n_triggered']} rules triggered)")
        print(sep)
        for rule in safety["triggered"]:
            print(f"  !! {rule['meaning']}")
            print(f"     {rule['feature']}={rule['value']} {rule['operator']} threshold {rule['threshold']}")

    print(f"\n{sep}")
    print("  WHY NOW")
    print(sep)
    print(f"  {exp['why_now']}")

    if exp.get("threshold_crossings"):
        for ev in exp["threshold_crossings"][:3]:
            print(
                f"    - {ev['feature']} {ev['direction']} {ev['threshold']} {ev.get('unit','')} "
                f"→ current {ev['value']} ({_format_crossing_time(ev)})"
            )

    # Phase trajectory (NEW)
    phases = exp.get("temporal_phases", {})
    if phases:
        print(f"\n{sep}")
        print("  TEMPORAL ANALYSIS")
        print(sep)
        window_map = {"Early": "Hour 0–2", "Mid": "Hour 2–4", "Recent": "Hour 4–6"}
        for label, vals in phases.items():
            tag   = window_map.get(label, label)
            items = []
            for fname in ["Heart Rate", "Respiration Rate", "SpO2", "Systolic BP", "Lactate", "PaO2_FiO2"]:
                if fname in vals and vals[fname] != 0:
                    items.append(f"{fname}={vals[fname]:.1f}")
            if items:
                print(f"  {tag}: {', '.join(items)}")
        traj = exp.get("phase_trajectory", "")
        if traj:
            summary_line = traj.split("\n")[-1] if "\n" in traj else traj
            print(f"\n  Pattern: {summary_line}")

    print(f"\n{sep}")
    n_readings = len(next(iter(exp["sparklines"].values()), "")) if exp["sparklines"] else 0
    print(f"  TREND (last {n_readings} readings)")
    print(sep)
    if exp["sparklines"]:
        for feat, spark in exp["sparklines"].items():
            print(f"  {feat:<20} {spark}")
    else:
        print("  (No vital sign trend data available)")

    print(f"\n{sep}")
    print("  WHAT TO WATCH NEXT")
    print(sep)
    for signal in exp.get("watch_for", []):
        print(f"  - {signal}")

    print(f"\n{sep}")
    n_ev = len(exp.get("evidence", []))
    print(f"  EVIDENCE  [{n_ev} articles — press E to expand]")
    print(sep)
    if n_ev > 0:
        for i, ev in enumerate(exp["evidence"][:3], 1):
            year   = ev.get("year", "?")
            age    = datetime.now().year - int(year) if str(year).isdigit() else None
            fresh  = "recent" if age is not None and age <= 3 else "older"
            insight = ev.get("clinical_insight", "")
            print(f"  [{i}] ({fresh}) {ev.get('title','?')[:55]}... ({year})")
            if insight:
                print(f"       ↳ {insight[:100]}")
    else:
        print("  No evidence retrieved.")

    print(f"\n{sep}")
    print("  FULL ASSESSMENT  [press A to expand]")
    print(sep)
    preview = exp.get("primary_assessment", "")[:200].replace("\n", " ")
    print(f"  {preview}...")

    print(f"\n{'=' * W}\n")


def expand_evidence(exp: dict):
    print(f"\n{'-' * 65}")
    print("  FULL EVIDENCE")
    print(f"{'-' * 65}")
    for i, ev in enumerate(exp.get("evidence", []), 1):
        print(f"\n  [{i}] {ev.get('title', '?')} ({ev.get('year', '?')})")
        print(f"       Score: {ev.get('score', 0):.3f}")
        insight = ev.get("clinical_insight", "")
        if insight:
            print(f"       Clinical Insight: {insight}")
        abstract = ev.get("abstract", "No abstract available.")[:500]
        print(f"       Abstract: {abstract}...")


def expand_assessment(exp: dict):
    print(f"\n{'-' * 65}")
    print("  FULL CLINICAL ASSESSMENT")
    print(f"{'-' * 65}")
    print(exp.get("primary_assessment", "No assessment available."))
