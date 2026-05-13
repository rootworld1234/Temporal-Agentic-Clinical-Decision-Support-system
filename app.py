"""
app.py — TA-CDSS Streamlit Application (Redesigned)
Temporal-Aware Clinical Decision Support System (Research-Grade)

Architecture: Click → Pipeline Executes → Report Displayed
UI controls NOTHING. Backend controls everything.

Run: streamlit run app.py
"""

import sys
import io
import json
import random
import time
import contextlib
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="TA-CDSS",
    page_icon="⬤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg:        #09090b;
  --bg-1:      #111116;
  --bg-2:      #18181f;
  --bg-3:      #1f1f2a;
  --line:      #2a2a38;
  --line-2:    #353548;
  --cyan:      #67e8f9;
  --cyan-dim:  #164e63;
  --green:     #4ade80;
  --green-dim: #14532d;
  --amber:     #fbbf24;
  --amber-dim: #78350f;
  --red:       #f87171;
  --red-dim:   #7f1d1d;
  --muted:     #6b7280;
  --body:      #d1d5db;
  --head:      #f9fafb;
  --mono:      'DM Mono', monospace;
  --sans:      'DM Sans', sans-serif;
  --display:   'Syne', sans-serif;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: var(--sans);
  color: var(--body);
}

/* Kill all Streamlit chrome */
header[data-testid="stHeader"],
.stDeployButton,
footer { display: none !important; }

section[data-testid="stSidebar"] { display: none !important; }

/* Main block */
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* All text inputs */
.stTextInput input, .stNumberInput input, .stSelectbox select,
.stTextInput > div > div > input {
  background: var(--bg-2) !important;
  border: 1px solid var(--line-2) !important;
  color: var(--head) !important;
  font-family: var(--mono) !important;
  font-size: 0.85rem !important;
  border-radius: 6px !important;
}

.stTextInput > div > div > input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px rgba(103,232,249,0.12) !important;
}

/* Selectbox */
.stSelectbox > div > div {
  background: var(--bg-2) !important;
  border: 1px solid var(--line-2) !important;
  color: var(--head) !important;
  font-family: var(--mono) !important;
  font-size: 0.8rem !important;
}

/* Primary CTA */
button[kind="primary"] {
  background: var(--cyan) !important;
  color: #000 !important;
  border: none !important;
  font-family: var(--display) !important;
  font-weight: 700 !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.04em !important;
  border-radius: 6px !important;
  transition: opacity .15s !important;
}
button[kind="primary"]:hover { opacity: 0.85 !important; }

/* Secondary buttons */
.stButton > button {
  background: var(--bg-3) !important;
  color: var(--body) !important;
  border: 1px solid var(--line-2) !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  border-radius: 6px !important;
  transition: border-color .15s, color .15s !important;
}
.stButton > button:hover {
  border-color: var(--cyan) !important;
  color: var(--cyan) !important;
}

/* Progress bar */
.stProgress > div > div { background: var(--cyan) !important; }
.stProgress > div { background: var(--bg-3) !important; }

/* Metrics */
[data-testid="metric-container"] {
  background: var(--bg-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  padding: 0.9rem 1.1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.68rem !important; text-transform: uppercase; letter-spacing: .09em; }
[data-testid="stMetricValue"] { color: var(--head) !important; font-family: var(--mono) !important; font-size: 1.5rem !important; }

/* Expanders */
details { background: var(--bg-2) !important; border: 1px solid var(--line) !important; border-radius: 8px !important; }
summary { color: var(--body) !important; font-size: 0.82rem !important; }

/* Download buttons */
.stDownloadButton > button {
  background: var(--bg-3) !important;
  color: var(--cyan) !important;
  border: 1px solid var(--cyan-dim) !important;
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
}

/* Captions */
.stCaption { color: var(--muted) !important; font-size: 0.75rem !important; }

/* Alert boxes */
.stAlert { border-radius: 8px !important; }
.stWarning { background: #2d1d00 !important; border-color: var(--amber) !important; }
.stError   { background: #2d0000 !important; border-color: var(--red) !important; }
.stSuccess { background: #001f10 !important; border-color: var(--green) !important; }
.stInfo    { background: var(--bg-2) !important; border-color: var(--line-2) !important; }

/* Dataframe */
.stDataFrame { border: 1px solid var(--line) !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Sys path ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


# ── Lazy module loader ────────────────────────────────────────────────────────
@st.cache_resource
def load_modules():
    try:
        from modules.gru_temporal       import EICUDataLoader, run_gru_module, estimate_confidence
        from modules.hybrid_rag         import run_hybrid_rag
        from modules.clinical_explainer import build_clinical_explanation
        from modules.agentic_validation import (
            run_primary_agent, run_validation_agent,
            run_safety_rule_check, run_medagentsbench_eval,
        )
        from modules.llm_client import gemini_budget_status
        from config import ICU_INPUT_SIZE, ICU_FEATURE_NAMES
        return dict(
            ok=True,
            EICUDataLoader=EICUDataLoader,
            run_gru_module=run_gru_module,
            estimate_confidence=estimate_confidence,
            run_hybrid_rag=run_hybrid_rag,
            build_clinical_explanation=build_clinical_explanation,
            run_primary_agent=run_primary_agent,
            run_validation_agent=run_validation_agent,
            run_safety_rule_check=run_safety_rule_check,
            run_medagentsbench_eval=run_medagentsbench_eval,
            gemini_budget_status=gemini_budget_status,
            ICU_INPUT_SIZE=ICU_INPUT_SIZE,
            ICU_FEATURE_NAMES=ICU_FEATURE_NAMES,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}


def capture(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def strip_ansi(text):
    import re
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def build_tcsv_description(gru_result):
    import numpy as np
    info     = gru_result.get("patient_info", {})
    seq_len  = gru_result.get("seq_len", 0)
    tcsv     = gru_result.get("tcsv", [])
    mag      = float(abs(tcsv).mean()) if len(tcsv) > 0 else 0.0
    trend    = "escalating" if mag > 0.3 else "stable" if mag > 0.1 else "minimal"
    vent_str = "mechanically ventilated" if info.get("ventilated") else "not ventilated"
    trajectory = gru_result.get("phase_trajectory", "")
    desc = (
        f"Temporal trajectory encoded from {seq_len} ICU observations "
        f"(11-feature vector including ventilator parameters). "
        f"Physiological trend: {trend} (TCSV mean magnitude: {mag:.4f}). "
        f"Patient: Age={info.get('age','?')}, Unit={info.get('unit_type','?')}, "
        f"ICU stay={info.get('icu_hours','?')}h, {vent_str} for {info.get('vent_hours',0)}h, "
        f"Diagnosis: {info.get('diagnosis_str', info.get('apache_score', 'unspecified'))}. "
    )
    if trajectory:
        desc += f"\n\nTemporal phase analysis:\n{trajectory}"
    return desc


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "stage":         "landing",   # landing | running | report
    "loader":        None,
    "data_dir":      "data/eicu",
    "patient_ids":   [],
    "data_loaded":   False,
    "run_steps":     [],          # list of (label, done)
    "report":        {},
    "pipeline_log":  "",
    "forced_pid":    None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def get_loader():
    if st.session_state.loader is not None:
        return st.session_state.loader
    if st.session_state.data_loaded:
        mods = load_modules()
        if mods["ok"]:
            try:
                loader = mods["EICUDataLoader"](data_dir=st.session_state.data_dir)
                loader.load()
                st.session_state.loader = loader
                return loader
            except Exception:
                pass
    return None


# ── Color helpers ─────────────────────────────────────────────────────────────
def risk_color(label):
    l = (label or "").upper()
    if "HIGH"   in l: return "#f87171", "#7f1d1d"
    if "MEDIUM" in l: return "#fbbf24", "#78350f"
    return "#4ade80", "#14532d"


def verdict_color(verdict):
    v = (verdict or "").upper()
    if "APPROVED" in v: return "#4ade80", "#14532d"
    if "REVISION" in v: return "#fbbf24", "#78350f"
    if "REJECTED" in v: return "#f87171", "#7f1d1d"
    return "#6b7280", "#1f1f2a"


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT SHELL
# ─────────────────────────────────────────────────────────────────────────────
def topbar():
    st.markdown("""
    <div style="
      display:flex; align-items:center; justify-content:space-between;
      padding:0.9rem 2.5rem;
      border-bottom:1px solid var(--line);
      background:var(--bg-1);
      position:sticky; top:0; z-index:100;
    ">
      <div style="display:flex;align-items:center;gap:1rem;">
        <div style="
          width:28px;height:28px;border-radius:50%;
          background:var(--cyan);display:flex;align-items:center;justify-content:center;
        ">
          <span style="font-size:12px;font-weight:800;color:#000;font-family:var(--display);">T</span>
        </div>
        <span style="font-family:var(--display);font-size:1rem;font-weight:700;
                     letter-spacing:0.05em;color:var(--head);">TA-CDSS</span>
        <span style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);
                     border:1px solid var(--line-2);padding:2px 8px;border-radius:20px;">
          Research · v2.0
        </span>
      </div>
      <div style="font-family:var(--mono);font-size:0.7rem;color:var(--muted);">
        Temporal-Aware Clinical Decision Support System
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LANDING
# ═════════════════════════════════════════════════════════════════════════════
def stage_landing():
    # Hero
    st.markdown("""
    <div style="
      min-height:30vh;
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      padding:4rem 2rem 2rem;
      text-align:center;
    ">
      <div style="font-family:var(--mono);font-size:0.7rem;color:var(--cyan);
                  letter-spacing:0.2em;text-transform:uppercase;margin-bottom:1.2rem;">
        Clinical Decision Support
      </div>
      <h1 style="
        font-family:var(--display);font-size:clamp(2.2rem,5vw,3.8rem);
        font-weight:800;color:var(--head);line-height:1.05;
        letter-spacing:-0.03em;margin:0 0 1rem;
      ">
        Temporal-Aware<br>
        <span style="color:var(--cyan);">ICU Analysis</span>
      </h1>
      <p style="font-family:var(--sans);font-size:1rem;color:var(--muted);
                max-width:520px;line-height:1.7;margin:0 0 2.5rem;">
        GRU temporal reasoning · Hybrid RAG retrieval · Agentic diagnosis · 3C3H validation.
        One click. Full pipeline.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline architecture preview
    _pipe_steps = [
        ("LOCAL",     "GRU Temporal",      "#4ade80"),
        ("LOCAL",     "Safety Check",      "#4ade80"),
        ("LOCAL",     "RAG Retrieval",     "#4ade80"),
        ("LOCAL",     "Confidence",        "#4ade80"),
        ("LM Studio", "Primary Diagnosis", "#fbbf24"),
        ("Gemini",    "Explanation",       "#67e8f9"),
        ("LM Studio", "Validation",        "#fbbf24"),
    ]
    _pipe_html = ""
    for _i, (_backend, _label, _col) in enumerate(_pipe_steps):
        _arrow = f"<span style='color:#2a2a38;padding:0 3px;'>&#8594;</span>" if _i < len(_pipe_steps) - 1 else ""
        _pipe_html += (
            f"<div style='display:inline-flex;align-items:center;'>"
            f"<div style='background:#18181f;border:1px solid #2a2a38;border-radius:6px;"
            f"padding:0.5rem 0.85rem;text-align:center;min-width:100px;'>"
            f"<div style='font-size:0.55rem;color:{_col};text-transform:uppercase;"
            f"letter-spacing:.1em;margin-bottom:3px;font-family:monospace;'>{_backend}</div>"
            f"<div style='font-size:0.73rem;color:#d1d5db;'>{_label}</div>"
            f"</div>{_arrow}</div>"
        )
    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:center;"
        f"flex-wrap:wrap;padding:0 1rem 2rem;'>{_pipe_html}</div>",
        unsafe_allow_html=True,
    )

    mods = load_modules()
    if not mods["ok"]:
        st.error(f"Module import failed: {mods['error']}\n\nRun `streamlit run app.py` from the project root.")
        return

    # ── LM Studio status indicator ────────────────────────────────────────────
    import requests as _req
    try:
        _lm_resp = _req.get("http://localhost:1234/v1/models", timeout=4)
        _lm_models = _lm_resp.json().get("data", []) if _lm_resp.ok else []
        if _lm_models:
            _lm_name = _lm_models[0].get("id", "unknown")
            st.markdown(
                f"<div style='background:#14532d;border:1px solid #4ade80;border-radius:8px;"
                f"padding:0.6rem 1rem;font-family:monospace;font-size:0.75rem;color:#4ade80;"
                f"margin-bottom:0.8rem;'>✓ LM Studio connected · model: {_lm_name}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#78350f;border:1px solid #fbbf24;border-radius:8px;"
                "padding:0.6rem 1rem;font-family:monospace;font-size:0.75rem;color:#fbbf24;"
                "margin-bottom:0.8rem;'>⚠ LM Studio server reachable but <b>no model loaded</b> — "
                "load a model in the Local Server tab before running analysis.</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            "<div style='background:#7f1d1d;border:1px solid #f87171;border-radius:8px;"
            "padding:0.6rem 1rem;font-family:monospace;font-size:0.75rem;color:#f87171;"
            "margin-bottom:0.8rem;'>✗ LM Studio not reachable at localhost:1234 — "
            "open LM Studio → Local Server tab → click <b>Start Server</b>, then reload this page.</div>",
            unsafe_allow_html=True,
        )

    # Data directory
    data_dir = st.text_input(
        "eICU data directory",
        value=st.session_state.data_dir,
        placeholder="data/eicu",
        help="Folder with patient.csv, vitalPeriodic.csv, etc."
    )

    # Load data button
    if not st.session_state.data_loaded:
        if st.button("Load Data →", use_container_width=True):
            with st.spinner("Scanning eICU files…"):
                EICUDataLoader = mods["EICUDataLoader"]
                loader = EICUDataLoader(data_dir=data_dir)
                _, _ = capture(loader.load)
                st.session_state.loader   = loader
                st.session_state.data_dir = data_dir
                st.session_state.data_loaded = True

                pid_col = "patientunitstayid"
                if loader.patients is not None and pid_col in loader.patients.columns:
                    all_ids = loader.patients[pid_col].dropna().unique().tolist()
                    if loader.vitals is not None and pid_col in loader.vitals.columns:
                        rich = loader.vitals[pid_col].value_counts()
                        rich = rich[rich >= 5].index.tolist()
                        candidates = [p for p in all_ids if p in rich] or all_ids
                    else:
                        candidates = all_ids
                    st.session_state.patient_ids = sorted([int(p) for p in candidates])
                st.rerun()

    # Patient selection (only after data loaded)
    if st.session_state.data_loaded:
        loader = get_loader()

        if loader is None:
            st.warning("Data connection lost. Reload the page and click **Load Data** again.")
        else:
            # Show data loaded confirmation
            n_patients = len(st.session_state.patient_ids)
            tables = {
                "patient.csv": loader.patients,
                "vitalPeriodic.csv": loader.vitals,
                "lab.csv": loader.labs,
                "diagnosis.csv": loader.diagnoses,
            }
            loaded_tables = [(k, len(v)) for k, v in tables.items() if v is not None]

            tbl_html = " &nbsp;·&nbsp; ".join(
                f'<span style="color:var(--green);">✓</span> {k.replace(".csv","")} <span style="color:var(--muted);">({n:,})</span>'
                for k, n in loaded_tables
            )
            st.markdown(f"""
            <div style="
              background:var(--green-dim);border:1px solid var(--green);
              border-radius:6px;padding:0.6rem 1rem;margin-bottom:1rem;
              font-family:var(--mono);font-size:0.72rem;
            ">{tbl_html}
              &nbsp;&nbsp;·&nbsp;&nbsp;
              <span style="color:var(--green);">{n_patients:,} patients</span>
            </div>
            """, unsafe_allow_html=True)

            # Patient ID
            col_sel, col_rnd = st.columns([4, 1])
            with col_sel:
                pid_options = st.session_state.patient_ids[:500]
                sel_pid = st.selectbox(
                    "Patient ID",
                    options=["Auto-select"] + pid_options,
                    key="landing_pid",
                    label_visibility="collapsed",
                )
            with col_rnd:
                if st.button("⟳ Random", use_container_width=True):
                    if st.session_state.patient_ids:
                        st.session_state.forced_pid = random.choice(st.session_state.patient_ids)
                        st.rerun()

            # Resolve patient
            resolved_pid = None
            if sel_pid != "Auto-select":
                resolved_pid = int(sel_pid)
            elif st.session_state.forced_pid:
                resolved_pid = st.session_state.forced_pid

            # Patient preview
            if resolved_pid and loader.patients is not None:
                pid_col = "patientunitstayid"
                row = loader.patients[loader.patients[pid_col] == resolved_pid]
                if not row.empty:
                    r = row.iloc[0]
                    st.markdown(f"""
                    <div style="
                      background:var(--bg-2);border:1px solid var(--line);
                      border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem;
                      font-family:var(--mono);font-size:0.75rem;
                      display:flex;gap:1.5rem;flex-wrap:wrap;
                    ">
                      <span><span style="color:var(--muted);">age </span>{r.get('age','?')}</span>
                      <span><span style="color:var(--muted);">sex </span>{r.get('gender','?')}</span>
                      <span><span style="color:var(--muted);">unit </span>{str(r.get('unittype','?'))[:16]}</span>
                      <span><span style="color:var(--muted);">discharge </span>{str(r.get('unitdischargestatus','?'))[:12]}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # RUN BUTTON
            run_clicked = st.button("▶ Run Full Analysis", type="primary", use_container_width=True)

            if run_clicked:
                final_pid = resolved_pid
                if final_pid is None and st.session_state.patient_ids:
                    final_pid = random.choice(st.session_state.patient_ids)

                if final_pid is None:
                    st.error("No patient IDs found. Check that patient.csv exists in the data directory.")
                else:
                    st.session_state.forced_pid = final_pid
                    st.session_state.stage = "running"
                    st.rerun()



# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — RUNNING
# Stores intermediate results in session_state, reruns after each step so
# st.markdown() (which supports unsafe_allow_html) renders the step list —
# avoiding st.empty().markdown() which does NOT support unsafe_allow_html.
# ═════════════════════════════════════════════════════════════════════════════
PIPELINE_STEPS = [
    ("GRU Temporal Reasoning",   "LOCAL",     "Encoding 6-hour vital trajectory"),
    ("Safety Rule Check",        "LOCAL",     "Evaluating 7 clinical safety rules"),
    ("Hybrid RAG Retrieval",     "LOCAL",     "Searching PubMed evidence base"),
    ("Confidence Estimation",    "LOCAL",     "Scoring data richness and quality"),
    ("Primary Diagnostic Agent", "LM Studio", "Generating clinical assessment"),
    ("Clinical Explanation",     "Gemini",    "Building patient-facing report"),
    ("3C3H Validation",          "LM Studio", "Verifying diagnosis consistency"),
]

# Init running state keys
for _k, _v in {
    "run_step":       0,       # which step to execute next (0-6), 7 = done
    "run_gru":        None,
    "run_safety":     None,
    "run_rag":        None,
    "run_conf":       None,
    "run_primary":    None,
    "run_explanation":None,
    "run_validation": None,
    "run_logs":       [],
    "run_error":      None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _draw_steps(n_done, active_idx):
    """Render the step checklist as st.markdown — always has unsafe_allow_html."""
    rows = ""
    for i, (name, backend, desc) in enumerate(PIPELINE_STEPS):
        if i < n_done:
            icon, icol, tcol, opacity = "✓", "var(--green)", "var(--body)", "1"
            bcol = {"LOCAL":"var(--green)","LM Studio":"var(--amber)","Gemini":"var(--cyan)"}.get(backend,"var(--muted)")
        elif i == active_idx:
            icon, icol, tcol, opacity = "◌", "var(--cyan)", "var(--head)", "1"
            bcol = {"LOCAL":"var(--green)","LM Studio":"var(--amber)","Gemini":"var(--cyan)"}.get(backend,"var(--muted)")
        else:
            icon, icol, tcol, opacity = "·", "var(--line-2)", "var(--muted)", "0.4"
            bcol = "var(--muted)"

        rows += f"""<div style="display:flex;align-items:center;gap:1.2rem;
          padding:0.75rem 0;border-bottom:1px solid var(--line);opacity:{opacity};">
          <span style="font-family:var(--mono);font-size:1rem;color:{icol};min-width:22px;">{icon}</span>
          <div style="flex:1;">
            <div style="font-size:0.88rem;font-weight:500;color:{tcol};">{name}</div>
            <div style="font-size:0.72rem;color:var(--muted);margin-top:1px;">{desc}</div>
          </div>
          <span style="font-family:var(--mono);font-size:0.62rem;color:{bcol};
            border:1px solid currentColor;padding:2px 8px;border-radius:20px;">{backend}</span>
        </div>"""

    st.markdown(f"""
    <div style="max-width:680px;margin:0 auto;padding:0 2.5rem;">
      <div style="background:var(--bg-1);border:1px solid var(--line);
                  border-radius:12px;padding:1.5rem 2rem;">
        {rows}
      </div>
    </div>
    """, unsafe_allow_html=True)


def stage_running():
    pid   = st.session_state.forced_pid
    mods  = load_modules()
    loader = get_loader()

    if not mods["ok"] or loader is None:
        st.error("Cannot run pipeline — modules or data not available.")
        if st.button("← Back"):
            st.session_state.stage = "landing"
            st.rerun()
        return

    # If a previous error was stored, show it
    if st.session_state.run_error:
        st.error(st.session_state.run_error)
        if st.button("← Start Over"):
            _reset_run_state()
            st.session_state.stage = "landing"
            st.rerun()
        return

    step = st.session_state.run_step

    # Header
    st.markdown(f"""
    <div style="max-width:680px;margin:3rem auto 1.5rem;padding:0 2.5rem;">
      <div style="font-family:var(--mono);font-size:0.68rem;color:var(--cyan);
                  letter-spacing:.15em;text-transform:uppercase;margin-bottom:0.5rem;">
        Analysis Running
      </div>
      <div style="font-family:var(--display);font-size:1.8rem;font-weight:800;
                  color:var(--head);letter-spacing:-0.02em;margin-bottom:0.3rem;">
        Patient {pid}
      </div>
      <div style="color:var(--muted);font-size:0.85rem;">
        Full 7-stage clinical pipeline · Please wait
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Draw the step list (plain st.markdown — works fine)
    _draw_steps(n_done=step, active_idx=step if step < len(PIPELINE_STEPS) else len(PIPELINE_STEPS))

    # Progress bar
    st.progress(step / len(PIPELINE_STEPS))

    # ── Execute exactly ONE step per rerun ────────────────────────────────────
    try:
        if step == 0:
            result, log = capture(mods["run_gru_module"], pid, loader)
            st.session_state.run_gru  = result
            st.session_state.run_logs = [f"=== GRU ===\n{log}"]
            st.session_state.run_step = 1
            st.rerun()

        elif step == 1:
            phases = st.session_state.run_gru.get("temporal_phases", {})
            result, log = capture(mods["run_safety_rule_check"], phases)
            st.session_state.run_safety = result
            st.session_state.run_logs.append(f"=== SAFETY ===\n{log}")
            st.session_state.run_step = 2
            st.rerun()

        elif step == 2:
            gru = st.session_state.run_gru
            result, log = capture(
                mods["run_hybrid_rag"],
                clinical_query  = "",
                patient_summary = gru["clinical_summary"],
                patient_info    = gru["patient_info"],
                temporal_phases = gru["temporal_phases"],
                top_k           = 8,
            )
            st.session_state.run_rag  = result
            st.session_state.run_logs.append(f"=== RAG ===\n{log}")
            st.session_state.run_step = 3
            st.rerun()

        elif step == 3:
            result, log = capture(
                mods["estimate_confidence"],
                st.session_state.run_gru,
                st.session_state.run_rag["confidence"],
            )
            st.session_state.run_conf = result
            st.session_state.run_logs.append(f"=== CONFIDENCE ===\n{log}")
            st.session_state.run_step = 4
            st.rerun()

        elif step == 4:
            # ── Pre-flight: verify LM Studio is reachable before calling it ──
            import requests as _req
            _lm_ok = False
            _lm_error = ""
            try:
                _r = _req.get("http://localhost:1234/v1/models", timeout=8)
                _models = _r.json().get("data", []) if _r.ok else []
                if _models:
                    _lm_ok = True
                else:
                    _lm_error = "LM Studio server is running but no model is loaded. Load a model in LM Studio → Local Server tab, then re-run."
            except _req.exceptions.ConnectionError:
                _lm_error = (
                    "Cannot reach LM Studio at localhost:1234. "
                    "Make sure you have started the server inside LM Studio "
                    "(Local Server tab → Start Server), not just opened the app."
                )
            except Exception as _e:
                _lm_error = f"LM Studio check failed: {_e}"

            if not _lm_ok:
                st.session_state.run_error = f"⚠️ LM Studio not reachable\n\n{_lm_error}"
                st.rerun()

            gru      = st.session_state.run_gru
            rag      = st.session_state.run_rag
            safety   = st.session_state.run_safety
            tcsv_desc = build_tcsv_description(gru)
            result, log = capture(
                mods["run_primary_agent"],
                patient_summary  = gru["clinical_summary"],
                tcsv_description = tcsv_desc,
                evidence_text    = rag["top_evidence"],
                safety_check     = safety,
                risk_score       = gru["risk_score"],
            )
            st.session_state.run_primary = result
            st.session_state.run_logs.append(f"=== PRIMARY AGENT ===\n{log}")
            st.session_state.run_step = 5
            st.rerun()

        elif step == 5:
            gru    = st.session_state.run_gru
            rag    = st.session_state.run_rag
            safety = st.session_state.run_safety
            conf   = st.session_state.run_conf
            result, log = capture(
                mods["build_clinical_explanation"],
                gru_result         = gru,
                risk_score         = gru["risk_score"],
                primary_assessment = st.session_state.run_primary,
                evidence           = rag["evidence"],
                safety_check       = safety,
                confidence         = conf,
                window_hours       = 6,
            )
            st.session_state.run_explanation = result
            st.session_state.run_logs.append(f"=== EXPLAINER ===\n{log}")
            st.session_state.run_step = 6
            st.rerun()

        elif step == 6:
            gru    = st.session_state.run_gru
            rag    = st.session_state.run_rag
            safety = st.session_state.run_safety
            result, log = capture(
                mods["run_validation_agent"],
                primary_output  = st.session_state.run_primary,
                evidence_text   = rag["top_evidence"],
                patient_summary = gru["clinical_summary"],
                safety_check    = safety,
            )
            st.session_state.run_validation = result
            st.session_state.run_logs.append(f"=== VALIDATION ===\n{log}")
            st.session_state.run_step = 7
            st.rerun()

        elif step == 7:
            # All done — pack results and move to report
            gru = st.session_state.run_gru
            st.session_state.report = {
                "patient_id":   pid,
                "gru":          gru,
                "safety_check": st.session_state.run_safety,
                "rag":          st.session_state.run_rag,
                "confidence":   st.session_state.run_conf,
                "primary":      st.session_state.run_primary,
                "explanation":  st.session_state.run_explanation,
                "validation":   st.session_state.run_validation,
                "timestamp":    datetime.now().isoformat(),
            }
            st.session_state.pipeline_log = "\n".join(st.session_state.run_logs)
            _reset_run_state()
            st.session_state.stage = "report"
            st.rerun()

    except Exception as e:
        import traceback
        st.session_state.run_error = f"Pipeline error at step {step + 1}: {e}\n\n{traceback.format_exc()}"
        st.rerun()


def _reset_run_state():
    for k in ["run_step","run_gru","run_safety","run_rag","run_conf",
              "run_primary","run_explanation","run_validation","run_logs","run_error"]:
        st.session_state[k] = [] if k == "run_logs" else (0 if k == "run_step" else None)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — REPORT (the full structured report viewer)
# ═════════════════════════════════════════════════════════════════════════════
def stage_report():
    r   = st.session_state.report
    if not r:
        st.warning("No report available.")
        if st.button("← Back to start"):
            st.session_state.stage = "landing"
            st.rerun()
        return

    exp      = r.get("explanation",  {})
    val      = r.get("validation",   {})
    gru      = r.get("gru",          {})
    conf     = r.get("confidence",   {})
    safety   = r.get("safety_check", {})
    rag      = r.get("rag",          {})
    pat_info = gru.get("patient_info", {})
    phases   = gru.get("temporal_phases", {})
    pid      = r.get("patient_id", "?")

    risk_fg, risk_bg = risk_color(exp.get("risk_label", "LOW"))
    verdict_fg, verdict_bg = verdict_color(val.get("verdict", ""))
    risk_pct = exp.get("risk_pct", 0)
    conf_pct = conf.get("pct", 0)

    # ── Action bar (new run / download) ───────────────────────────────────────
    st.markdown("""
    <div style="
      display:flex;align-items:center;justify-content:space-between;
      padding:0.6rem 2.5rem;
      border-bottom:1px solid var(--line);
      background:var(--bg-1);
    ">
      <div style="font-family:var(--mono);font-size:0.72rem;color:var(--muted);">
        Analysis complete
        <span style="color:var(--green);margin-left:6px;">✓</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    action_c1, action_c2, action_c3 = st.columns([1, 1, 6])
    with action_c1:
        if st.button("← New Patient", use_container_width=True):
            _reset_run_state()
            st.session_state.stage = "landing"
            st.rerun()
    with action_c2:
        report_txt = _build_txt_report(r)
        ts_dl = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "↓ Export TXT",
            data=report_txt,
            file_name=f"ta_cdss_{pid}_{ts_dl}.txt",
            mime="text/plain",
            use_container_width=True,
        )


    # ══ SECTION 1 — Patient Overview + Risk Banner ═══════════════════════════
    st.markdown(f"""
    <div style="
      display:grid;grid-template-columns:2fr 1fr;gap:1rem;
      margin:1.5rem 0 1rem;
    ">
      <!-- Patient card -->
      <div style="
        background:var(--bg-1);border:1px solid var(--line);border-radius:12px;
        padding:1.4rem 1.8rem;
      ">
        <div style="font-family:var(--mono);font-size:0.62rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.15em;margin-bottom:0.8rem;">
          Patient Overview
        </div>
        <div style="font-family:var(--display);font-size:1.6rem;font-weight:800;
                    color:var(--head);letter-spacing:-0.02em;margin-bottom:0.5rem;">
          #{pid}
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:1.5rem;font-family:var(--mono);font-size:0.78rem;">
          <span><span style="color:var(--muted);">age </span>{pat_info.get('age','?')}</span>
          <span><span style="color:var(--muted);">sex </span>{pat_info.get('gender','?')}</span>
          <span><span style="color:var(--muted);">unit </span>{pat_info.get('unit_type','?')}</span>
          <span><span style="color:var(--muted);">icu </span>{pat_info.get('icu_hours','?')}h</span>
          <span><span style="color:var(--muted);">vent </span>{"Yes " + str(pat_info.get('vent_hours',0)) + "h" if pat_info.get('ventilated') else "No"}</span>
        </div>
        <div style="margin-top:0.7rem;font-size:0.85rem;color:var(--body);">
          {pat_info.get('diagnosis_str','Unknown diagnosis')}
        </div>
      </div>

      <!-- Risk card -->
      <div style="
        background:{risk_bg};border:1px solid {risk_fg};border-radius:12px;
        padding:1.4rem 1.8rem;display:flex;flex-direction:column;justify-content:space-between;
      ">
        <div style="font-family:var(--mono);font-size:0.62rem;color:{risk_fg};
                    text-transform:uppercase;letter-spacing:.15em;">
          Deterioration Risk
        </div>
        <div>
          <div style="font-family:var(--display);font-size:3rem;font-weight:800;
                      color:{risk_fg};line-height:1;margin:0.3rem 0;">
            {risk_pct}%
          </div>
          <div style="font-family:var(--mono);font-size:0.78rem;color:{risk_fg};opacity:0.85;">
            {exp.get('risk_label','?').upper()}
          </div>
        </div>
        <div style="
          display:flex;align-items:center;gap:0.5rem;
          background:{verdict_bg};border-radius:6px;padding:0.4rem 0.8rem;
          margin-top:0.8rem;
        ">
          <span style="font-size:12px;">{"✓" if "APPROVED" in (val.get("verdict","")).upper() else "⚠" if "REVISION" in (val.get("verdict","")).upper() else "✗"}</span>
          <span style="font-family:var(--mono);font-size:0.72rem;color:{verdict_fg};">
            {val.get('verdict','UNKNOWN')}
          </span>
          <span style="font-family:var(--mono);font-size:0.68rem;color:var(--muted);margin-left:auto;">
            Confidence {conf_pct}%
          </span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══ SECTION 2 — Safety Alerts (if any) ═══════════════════════════════════
    if safety.get("triggered"):
        n = safety["n_triggered"]
        sev = safety.get("severity", "?")
        border_col = "#f87171" if sev == "HIGH" else "#fbbf24"
        bg_col = "#1a0000" if sev == "HIGH" else "#1a0d00"
        _rules_html = ""
        for _rule in safety["triggered"]:
            _rules_html += (f"<div style='display:flex;gap:1rem;align-items:baseline;"
                            f"padding:0.4rem 0;border-bottom:1px solid rgba(255,255,255,0.05);'>"
                            f"<span style='font-family:monospace;font-size:0.75rem;color:{border_col};min-width:240px;'>"
                            f"{_rule['meaning']}</span>"
                            f"<span style='font-family:monospace;font-size:0.7rem;color:#6b7280;'>"
                            f"{_rule['feature']} = {_rule['value']} {_rule['operator']} {_rule['threshold']}"
                            f"</span></div>")
        st.markdown(
            f"<div style='background:{bg_col};border:1px solid {border_col};"
            f"border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1rem;'>"
            f"<div style='font-family:monospace;font-size:0.68rem;color:{border_col};"
            f"text-transform:uppercase;letter-spacing:.12em;margin-bottom:0.8rem;'>"
            f"&#9888; {n} Safety Alert{'s' if n > 1 else ''} &middot; Severity {sev}</div>"
            f"{_rules_html}</div>",
            unsafe_allow_html=True,
        )

    # ══ SECTION 3 — Summary + Why Now ════════════════════════════════════════
    top_line = exp.get("top_line", "—")
    why_now  = exp.get("delta_why_now") or exp.get("why_now", "—")

    st.markdown(f"""
    <div style="
      display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;
    ">
      <div style="background:var(--bg-2);border:1px solid var(--line);
                  border-radius:10px;padding:1.2rem 1.5rem;">
        <div style="font-family:var(--mono);font-size:0.62rem;color:var(--cyan);
                    text-transform:uppercase;letter-spacing:.12em;margin-bottom:0.6rem;">
          Clinical Summary
        </div>
        <div style="font-size:0.9rem;color:var(--head);line-height:1.6;">{top_line}</div>
      </div>
      <div style="background:var(--bg-2);border:1px solid var(--line);
                  border-radius:10px;padding:1.2rem 1.5rem;">
        <div style="font-family:var(--mono);font-size:0.62rem;color:var(--amber);
                    text-transform:uppercase;letter-spacing:.12em;margin-bottom:0.6rem;">
          Why Flagged Now
        </div>
        <div style="font-size:0.85rem;color:var(--body);line-height:1.6;">{why_now}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══ SECTION 4 — Temporal Analysis ════════════════════════════════════════
    if phases:
        _wmap = {"Early": ("Hour 0-2", "#4ade80"), "Mid": ("Hour 2-4", "#fbbf24"), "Recent": ("Hour 4-6", "#f87171")}
        _vitals = ["SpO2", "Heart Rate", "Respiration Rate", "Systolic BP", "Lactate", "PaO2_FiO2", "FiO2"]
        _phase_cols = ""
        for _label, _vals in phases.items():
            _win, _col = _wmap.get(_label, (_label, "#6b7280"))
            _rows = ""
            for _vn in _vitals:
                _v = _vals.get(_vn, 0)
                if _v:
                    _rows += (f"<div style='display:flex;justify-content:space-between;padding:3px 0;"
                              f"font-family:monospace;font-size:0.72rem;border-bottom:1px solid #2a2a38;'>"
                              f"<span style='color:#6b7280;'>{_vn}</span>"
                              f"<span style='color:#f9fafb;'>{_v:.1f}</span></div>")
            if not _rows:
                _rows = "<span style='color:#6b7280;font-size:0.72rem;'>no data</span>"
            _phase_cols += (f"<div style='flex:1;min-width:180px;background:#18181f;border:1px solid #2a2a38;"
                            f"border-radius:10px;padding:1rem 1.2rem;'>"
                            f"<div style='font-family:monospace;font-size:0.62rem;color:{_col};"
                            f"text-transform:uppercase;letter-spacing:.12em;margin-bottom:0.8rem;'>{_win}</div>"
                            f"{_rows}</div>")
        st.markdown(
            f"<div style='margin-bottom:1rem;'>"
            f"<div style='font-family:monospace;font-size:0.62rem;color:#6b7280;"
            f"text-transform:uppercase;letter-spacing:.12em;margin-bottom:0.6rem;'>"
            f"Temporal Analysis - 6-Hour Window</div>"
            f"<div style='display:flex;gap:0.8rem;flex-wrap:wrap;'>{_phase_cols}</div></div>",
            unsafe_allow_html=True,
        )

    # ══ SECTION 5 — What To Watch ════════════════════════════════════════════
    watch_items = exp.get("watch_for", [])
    if watch_items:
        _items_html = ""
        for _item in watch_items:
            _items_html += (f"<div style='display:flex;gap:0.8rem;align-items:flex-start;"
                            f"padding:0.55rem 0.8rem;background:#1f1f2a;border-radius:6px;margin-bottom:0.4rem;'>"
                            f"<span style='color:#67e8f9;font-size:12px;margin-top:2px;'>&#9670;</span>"
                            f"<span style='font-size:0.84rem;color:#d1d5db;line-height:1.5;'>{_item}</span></div>")
        st.markdown(
            f"<div style='background:#18181f;border:1px solid #2a2a38;border-radius:10px;"
            f"padding:1.2rem 1.5rem;margin-bottom:1rem;'>"
            f"<div style='font-family:monospace;font-size:0.62rem;color:#67e8f9;"
            f"text-transform:uppercase;letter-spacing:.12em;margin-bottom:0.8rem;'>What To Watch Next</div>"
            f"{_items_html}</div>",
            unsafe_allow_html=True,
        )

    # ══ SECTION 6 — Evidence ═════════════════════════════════════════════════
    evidence = rag.get("evidence", [])
    if evidence:
        st.markdown(f"""
        <div style="font-family:var(--mono);font-size:0.62rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.12em;margin-bottom:0.6rem;">
          Supporting Evidence — {len(evidence)} Articles
          &nbsp;·&nbsp; Retrieval confidence {rag.get('confidence',0):.2f}
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"▶ View {len(evidence)} Supporting Articles"):
            for i, ev in enumerate(evidence, 1):
                insight = ev.get("clinical_insight", "")
                year    = ev.get("year", "?")
                score   = ev.get("score", 0)
                st.markdown(f"""
                <div style="
                  background:var(--bg-3);border:1px solid var(--line);
                  border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.6rem;
                ">
                  <div style="font-weight:500;font-size:0.86rem;color:var(--head);margin-bottom:0.3rem;">
                    [{i}] {ev.get('title','No title')}
                  </div>
                  <div style="font-family:var(--mono);font-size:0.68rem;color:var(--muted);margin-bottom:0.4rem;">
                    {year} · PMID {ev.get('pmid','?')} · score {score:.3f}
                  </div>
                  {f'<div style="font-size:0.8rem;color:var(--cyan);font-style:italic;border-left:2px solid var(--cyan);padding-left:0.6rem;">↳ {insight}</div>' if insight else ''}
                </div>
                """, unsafe_allow_html=True)

    # ══ SECTION 7 — Full Diagnosis ════════════════════════════════════════════
    primary = r.get("primary", "No assessment generated.")
    with st.expander("📋 Full Clinical Assessment"):
        st.markdown(f"""
        <div style="
          background:var(--bg-1);border:1px solid var(--line);border-radius:8px;
          padding:1.2rem 1.5rem;font-family:var(--mono);font-size:0.8rem;
          color:var(--body);line-height:1.8;white-space:pre-wrap;
        ">{primary}</div>
        """, unsafe_allow_html=True)

    # ══ SECTION 8 — Validation + Actions ═════════════════════════════════════
    verdict  = val.get("verdict", "UNKNOWN")
    concern  = val.get("concern", "")
    val_text = val.get("validation_text", "")
    actions  = val.get("actions", [])

    _actions_html = ""
    for _a in actions:
        _actions_html += (f"<div style='display:flex;gap:0.6rem;align-items:flex-start;"
                          f"margin-bottom:0.4rem;font-size:0.82rem;color:#d1d5db;'>"
                          f"<span style='color:#67e8f9;'>&#9658;</span>{_a}</div>")
    if not _actions_html:
        _actions_html = "<span style='color:#6b7280;font-size:0.82rem;'>No specific actions recommended.</span>"

    st.markdown(
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem;'>"
        f"<div style='background:{verdict_bg};border:1px solid {verdict_fg};"
        f"border-radius:10px;padding:1.2rem 1.5rem;'>"
        f"<div style='font-family:monospace;font-size:0.62rem;color:{verdict_fg};"
        f"text-transform:uppercase;letter-spacing:.12em;margin-bottom:0.6rem;'>Validation &middot; 3C3H</div>"
        f"<div style='font-size:1.2rem;font-weight:700;color:{verdict_fg};margin-bottom:0.4rem;'>{verdict}</div>"
        f"<div style='font-size:0.82rem;color:#d1d5db;line-height:1.5;'>"
        f"{concern if concern else 'No specific concern noted.'}</div></div>"
        f"<div style='background:#18181f;border:1px solid #2a2a38;"
        f"border-radius:10px;padding:1.2rem 1.5rem;'>"
        f"<div style='font-family:monospace;font-size:0.62rem;color:#67e8f9;"
        f"text-transform:uppercase;letter-spacing:.12em;margin-bottom:0.6rem;'>Recommended Actions</div>"
        f"{_actions_html}</div></div>",
        unsafe_allow_html=True,
    )

    # ══ SECTION 9 — Benchmark (if available) ═════════════════════════════════
    bench_score = exp.get("benchmark_score", None)
    if bench_score is not None:
        st.markdown(f"""
        <div style="
          background:var(--bg-2);border:1px solid var(--line);border-radius:8px;
          padding:0.8rem 1.2rem;margin-top:1rem;
          display:flex;align-items:center;gap:1rem;
        ">
          <span style="font-family:var(--mono);font-size:0.68rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.12em;">
            MedAgentsBench
          </span>
          <span style="font-family:var(--display);font-size:1.4rem;font-weight:700;
                        color:var(--cyan);">
            {bench_score:.1f}%
          </span>
        </div>
        """, unsafe_allow_html=True)

    # ══ Pipeline log (collapsible) ════════════════════════════════════════════
    with st.expander("📜 Pipeline Execution Log"):
        log_clean = strip_ansi(st.session_state.pipeline_log)
        st.markdown(f"""
        <div style="
          background:var(--bg-1);border:1px solid var(--line);border-radius:8px;
          padding:1rem;font-family:var(--mono);font-size:0.7rem;color:var(--muted);
          max-height:320px;overflow-y:auto;white-space:pre-wrap;
        ">{log_clean}</div>
        """, unsafe_allow_html=True)



# ── Plain-text report builder ─────────────────────────────────────────────────
def _build_txt_report(r):
    exp      = r.get("explanation",  {})
    val      = r.get("validation",   {})
    gru      = r.get("gru",          {})
    conf     = r.get("confidence",   {})
    safety   = r.get("safety_check", {})
    rag      = r.get("rag",          {})
    pat_info = gru.get("patient_info", {})
    phases   = gru.get("temporal_phases", {})
    W = 65
    lines = []
    lines.append("=" * W)
    lines.append("  TA-CDSS — Clinical Decision Support Report (Research-Grade)")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * W)
    lines.append("")
    lines.append(f"PATIENT:     {r.get('patient_id','N/A')}")
    lines.append(f"AGE/GENDER:  {pat_info.get('age','?')} / {pat_info.get('gender','?')}")
    lines.append(f"UNIT:        {pat_info.get('unit_type','?')}")
    lines.append(f"ICU STAY:    {pat_info.get('icu_hours','?')}h  |  Ventilated: {pat_info.get('ventilated','?')} ({pat_info.get('vent_hours','?')}h)")
    lines.append(f"DIAGNOSIS:   {pat_info.get('diagnosis_str','unknown')}")
    lines.append(f"RISK LEVEL:  {exp.get('risk_label','?')}  —  {exp.get('risk_pct','?')}%")
    lines.append(f"CONFIDENCE:  {conf.get('pct','?')}% ({conf.get('label','unknown')})")
    lines.append(f"SUMMARY:     {exp.get('top_line','N/A')}")
    lines.append("")
    if safety and safety.get("triggered"):
        lines.append("-" * W)
        lines.append(f"SAFETY ALERTS  ({safety['n_triggered']} rules  —  severity: {safety['severity']})")
        lines.append("-" * W)
        for rule in safety["triggered"]:
            lines.append(f"  !! {rule['meaning']}")
            lines.append(f"     {rule['feature']}={rule['value']} {rule['operator']} threshold {rule['threshold']}")
        lines.append("")
    lines.append("-" * W)
    lines.append("WHY NOW")
    lines.append("-" * W)
    lines.append(exp.get("delta_why_now") or exp.get("why_now", "N/A"))
    lines.append("")
    if phases:
        lines.append("-" * W)
        lines.append("TEMPORAL ANALYSIS")
        lines.append("-" * W)
        wm = {"Early": "Hour 0–2", "Mid": "Hour 2–4", "Recent": "Hour 4–6"}
        for label, vals in phases.items():
            items = [f"{k}={v:.1f}" for k, v in vals.items() if v and v != 0]
            if items:
                lines.append(f"  {wm.get(label,label)}: {', '.join(items)}")
        lines.append("")
    lines.append("-" * W)
    lines.append("WHAT TO WATCH")
    lines.append("-" * W)
    for item in exp.get("watch_for", []):
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("-" * W)
    lines.append("FULL CLINICAL ASSESSMENT")
    lines.append("-" * W)
    lines.append(r.get("primary", "N/A").replace("**", "").replace("__", ""))
    lines.append("")
    evidence = rag.get("evidence", [])
    lines.append("-" * W)
    lines.append(f"EVIDENCE  ({len(evidence)} articles  |  query: {rag.get('query','')})")
    lines.append("-" * W)
    for ev in evidence:
        lines.append(f"  [{ev.get('year','?')}] {ev.get('title','?')}")
        if ev.get("clinical_insight"):
            lines.append(f"  Insight: {ev['clinical_insight']}")
        lines.append("")
    lines.append("-" * W)
    lines.append("VALIDATION")
    lines.append("-" * W)
    lines.append(f"  Verdict: {val.get('verdict','?')}")
    if val.get("concern"):
        lines.append(f"  Concern: {val['concern']}")
    if val.get("validation_text"):
        lines.append(val["validation_text"])
    lines.append("")
    if val.get("actions"):
        lines.append("-" * W)
        lines.append("RECOMMENDED ACTIONS")
        lines.append("-" * W)
        for a in val["actions"]:
            lines.append(f"  {a}")
        lines.append("")
    lines.append("=" * W)
    lines.append("  END OF REPORT")
    lines.append("=" * W)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════════════
topbar()

stage = st.session_state.stage
if   stage == "landing":  stage_landing()
elif stage == "running":  stage_running()
elif stage == "report":   stage_report()
else:
    st.session_state.stage = "landing"
    st.rerun()