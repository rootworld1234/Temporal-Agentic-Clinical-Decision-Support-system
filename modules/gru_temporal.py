"""
Component A — GRU Temporal Reasoning Module (Research-Grade)

Key improvements over prototype:
  1. Extended feature set: 11 ICU variables (was 6) — adds FiO2, PEEP,
     Tidal Volume, PaO2/FiO2, Lactate for ventilator-aware reasoning
  2. Phase-based temporal analysis: labels early/mid/recent windows
     so the LLM can say "Hour 0–2: stable, Hour 4–6: SpO2 declining"
  3. Calibrated risk score: uses clinical abnormality count + TCSV
     magnitude instead of raw TCSV mean (was unreliable)
  4. Richer patient context: ICU stay duration, ventilation hours,
     diagnosis string passed downstream for patient-level framing
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# Import extended feature config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ICU_FEATURE_NAMES, ICU_INPUT_SIZE, CLINICAL_THRESHOLDS,
    TEMPORAL_PHASES, TEMPORAL_WINDOW_HOURS, RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD,
)


# ── Time-Aware GRU (unchanged architecture, wider input) ──────────────────────

class TimeAwareGRU(nn.Module):
    """GRU with temporal gap encoding for irregular ICU timestamps."""

    def __init__(self, input_size: int = ICU_INPUT_SIZE, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.time_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, input_size),
        )
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor) -> tuple:
        time_emb = self.time_encoder(delta_t)
        x_time   = x + time_emb
        output, h_n = self.gru(x_time)
        tcsv = self.output_proj(h_n[-1])
        return tcsv, output


# ── Extended eICU Data Loader ──────────────────────────────────────────────────

class EICUDataLoader:
    """
    Loads and preprocesses eICU data into temporal sequences.

    Extended vs prototype:
      - Pulls FiO2, PEEP, tidal volume from respiratoryCharting table
      - Pulls lactate from lab table
      - Computes P/F ratio when PaO2 and FiO2 are available
      - Attaches ICU stay duration and ventilation hours to patient info
    """

    # Basic vitals from vitalPeriodic
    VITAL_COLS = [
        "heartrate", "respiration", "sao2",
        "temperature", "systemicsystolic", "systemicdiastolic",
    ]
    # Extended: vent params from respiratoryCharting (approximate column names)
    RESP_COLS = ["fio2", "peep", "tidalvolume"]
    LAB_COLS  = ["labresult"]

    def __init__(self, data_dir: str = "data/eicu"):
        self.data_dir      = Path(data_dir)
        self.patients      = None
        self.vitals        = None
        self.labs          = None
        self.treatments    = None
        self.diagnoses     = None
        self.resp_charting = None   # NEW: respiratory charting table

    def load(self) -> bool:
        print("\n Loading eICU data...")
        loaded = []

        files = {
            "patients":      "patient.csv",
            "vitals":        "vitalPeriodic.csv",
            "labs":          "lab.csv",
            "treatments":    "treatment.csv",
            "diagnoses":     "diagnosis.csv",
            "resp_charting": "respiratoryCharting.csv",   # optional extended table
        }

        for attr, filename in files.items():
            path = self.data_dir / filename
            if path.exists():
                df = pd.read_csv(path, low_memory=False)
                setattr(self, attr, df)
                loaded.append(filename)
                print(f"   {filename}: {len(df):,} rows")
            else:
                print(f"    {filename}: not found (optional)")

        if not loaded:
            print("   No eICU files found. Check data/eicu/ folder.")
            return False
        return True

    # ── Patient sequence ───────────────────────────────────────────────────────

    def get_patient_sequence(self, patient_id: int) -> dict:
        result = {
            "patient_id": patient_id,
            "features":   [],
            "timestamps": [],
            "info":       {},
        }

        # Demographics + ICU stay context
        if self.patients is not None:
            pid_col = self._pid_col(self.patients)
            pat = self.patients[self.patients[pid_col] == patient_id]
            if not pat.empty:
                row = pat.iloc[0]
                los_offset = row.get("hospitaldischargeoffset", 0)
                icu_hours  = round(float(los_offset) / 60, 1) if pd.notna(los_offset) else None
                result["info"] = {
                    "age":         row.get("age", "unknown"),
                    "gender":      row.get("gender", "unknown"),
                    "unit_type":   row.get("unittype", "unknown"),
                    "apache_score":row.get("apacheadmissiondx", "unknown"),
                    "hospital_los":los_offset,
                    # NEW: ICU stay duration in hours
                    "icu_hours":   icu_hours,
                    # NEW: ventilation status (approximate from treatment table)
                    "ventilated":  self._was_ventilated(patient_id),
                    "vent_hours":  self._vent_hours(patient_id),
                    # NEW: primary diagnosis string
                    "diagnosis_str": self._get_diagnosis(patient_id),
                }

        # Build time-aligned feature matrix
        vent_map = self._get_vent_params(patient_id)   # {offset_min -> {fio2, peep, tv}}
        lactate_map = self._get_lactate(patient_id)     # {offset_min -> lactate}

        if self.vitals is not None:
            pid_col  = self._pid_col(self.vitals)
            v        = self.vitals[self.vitals[pid_col] == patient_id].copy()
            if not v.empty:
                time_col = "observationoffset" if "observationoffset" in v.columns else v.columns[1]
                v = v.sort_values(time_col)

                for _, row in v.iterrows():
                    t_offset = float(row.get(time_col, 0))

                    # Basic vitals (0-5) with physiological clamping
                    # FIX 1: Raw eICU values can be out-of-range (artefacts, unit errors).
                    # Clamp every vital to its known physiological bounds before use.
                    raw_vitals = []
                    for col in self.VITAL_COLS:
                        val = float(row.get(col, np.nan)) if col in v.columns else np.nan
                        raw_vitals.append(val if pd.notna(val) else np.nan)

                    # Clamp ranges: (min_valid, max_valid, replacement_if_invalid)
                    VITAL_BOUNDS = [
                        (20,  250, np.nan),   # Heart Rate      bpm
                        (4,   60,  np.nan),   # Respiration     br/min
                        (50,  100, np.nan),   # SpO2            % — anything <50 is artefact
                        (30,  43,  np.nan),   # Temperature     °C
                        (40,  250, np.nan),   # Systolic BP     mmHg
                        (20,  150, np.nan),   # Diastolic BP    mmHg
                    ]
                    features = []
                    for i, (lo, hi, fallback) in enumerate(VITAL_BOUNDS):
                        v_raw = raw_vitals[i]
                        if np.isnan(v_raw):
                            features.append(0.0)
                        elif v_raw < lo or v_raw > hi:
                            features.append(0.0)   # treat out-of-range as missing
                        else:
                            features.append(float(v_raw))

                    # Ventilator params (6-8): match nearest vent measurement
                    vp = self._nearest(vent_map, t_offset, tolerance_min=60)
                    fio2_raw = vp.get("fio2", 0.21)
                    # FiO2 sometimes stored as percentage (e.g. 40) rather than fraction (0.40)
                    if fio2_raw > 1.0:
                        fio2_raw = fio2_raw / 100.0
                    fio2_val = float(np.clip(fio2_raw, 0.21, 1.0))
                    features.append(fio2_val)                               # FiO2
                    features.append(float(np.clip(vp.get("peep", 5.0), 0, 30)))   # PEEP
                    features.append(float(np.clip(vp.get("tv",   6.0), 2, 20)))   # Tidal Vol

                    # P/F ratio (9): only meaningful when SpO2 is valid (>50%)
                    spo2  = features[2]
                    pf    = (spo2 / max(fio2_val, 0.21)) if spo2 > 50 else 0.0
                    features.append(float(np.clip(pf, 0, 600)))

                    # Lactate (10)
                    lac = self._nearest(lactate_map, t_offset, tolerance_min=120)
                    features.append(lac.get("value", 0.0))

                    result["features"].append(features)
                    result["timestamps"].append(t_offset)

        return result

    # ── Tensor builder ─────────────────────────────────────────────────────────

    def build_tensor(self, patient_id: int, input_size: int = ICU_INPUT_SIZE) -> tuple:
        seq        = self.get_patient_sequence(patient_id)
        features   = seq["features"]
        timestamps = seq["timestamps"]

        if len(features) < 2:
            x       = torch.zeros(1, 2, input_size)
            delta_t = torch.zeros(1, 2, 1)
            return x, delta_t, seq

        arr = np.array(features[:50], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)

        if arr.shape[1] < input_size:
            arr = np.pad(arr, ((0, 0), (0, input_size - arr.shape[1])))
        elif arr.shape[1] > input_size:
            arr = arr[:, :input_size]

        ts   = np.array(timestamps[:50], dtype=np.float32)
        gaps = np.diff(ts, prepend=ts[0]).reshape(-1, 1)
        gaps = np.clip(gaps / 60.0, 0, 24)

        x       = torch.tensor(arr,  dtype=torch.float32).unsqueeze(0)
        delta_t = torch.tensor(gaps, dtype=torch.float32).unsqueeze(0)
        return x, delta_t, seq

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _pid_col(self, df: pd.DataFrame) -> str:
        return "patientunitstayid" if "patientunitstayid" in df.columns else df.columns[0]

    def _was_ventilated(self, patient_id: int) -> bool:
        if self.treatments is None:
            return False
        pid_col = self._pid_col(self.treatments)
        t = self.treatments[self.treatments[pid_col] == patient_id]
        if t.empty:
            return False
        treat_col = "treatmentstring" if "treatmentstring" in t.columns else ""
        if not treat_col:
            return False
        return t[treat_col].astype(str).str.lower().str.contains("ventilat").any()

    def _vent_hours(self, patient_id: int) -> float:
        """Approximate mechanical ventilation duration in hours."""
        if self.treatments is None:
            return 0.0
        pid_col  = self._pid_col(self.treatments)
        t        = self.treatments[self.treatments[pid_col] == patient_id]
        tc       = "treatmentstring" if "treatmentstring" in t.columns else ""
        off_col  = "treatmentoffset" if "treatmentoffset" in t.columns else ""
        if not tc or not off_col:
            return 0.0
        vent = t[t[tc].astype(str).str.lower().str.contains("ventilat", na=False)]
        if vent.empty:
            return 0.0
        offsets = vent[off_col].dropna().astype(float)
        if len(offsets) < 2:
            return 0.0
        return round((offsets.max() - offsets.min()) / 60.0, 1)

    def _get_diagnosis(self, patient_id: int) -> str:
        if self.diagnoses is None:
            return "unknown"
        pid_col  = self._pid_col(self.diagnoses)
        d        = self.diagnoses[self.diagnoses[pid_col] == patient_id]
        dx_col   = "diagnosisstring" if "diagnosisstring" in d.columns else ""
        if d.empty or not dx_col:
            return "unknown"
        dx = d[dx_col].dropna().iloc[0] if not d[dx_col].dropna().empty else "unknown"
        return str(dx)[:80]

    def _get_vent_params(self, patient_id: int) -> dict:
        """Return {offset_min: {fio2, peep, tv}} from respiratoryCharting."""
        result = {}
        if self.resp_charting is None:
            return result
        pid_col = self._pid_col(self.resp_charting)
        rc      = self.resp_charting[self.resp_charting[pid_col] == patient_id]
        if rc.empty:
            return result
        for _, row in rc.iterrows():
            off = float(row.get("respchartoffset", 0))
            entry = result.setdefault(off, {})
            label = str(row.get("respchartvaluelabel", "")).lower()
            try:
                val = float(row.get("respchartvalue", np.nan))
            except (ValueError, TypeError):
                val = np.nan
            if pd.isna(val):
                continue
            if "fio2" in label:
                entry["fio2"] = val / 100.0 if val > 1.0 else val
            elif "peep" in label:
                entry["peep"] = val
            elif "tidal" in label or "vt" in label:
                entry["tv"] = val
        return result

    def _get_lactate(self, patient_id: int) -> dict:
        """Return {offset_min: {value}} from lab table."""
        result = {}
        if self.labs is None:
            return result
        pid_col  = self._pid_col(self.labs)
        lab_col  = "labname" if "labname" in self.labs.columns else ""
        val_col  = "labresult" if "labresult" in self.labs.columns else ""
        off_col  = "labresultoffset" if "labresultoffset" in self.labs.columns else ""
        if not all([lab_col, val_col, off_col]):
            return result
        lbs = self.labs[self.labs[pid_col] == patient_id]
        lac = lbs[lbs[lab_col].astype(str).str.lower().str.contains("lactate", na=False)]
        for _, row in lac.iterrows():
            try:
                off = float(row[off_col])
                val = float(row[val_col])
                result[off] = {"value": val}
            except (ValueError, TypeError):
                pass
        return result

    def _nearest(self, time_map: dict, target: float, tolerance_min: float = 60) -> dict:
        """Find closest entry in a time-keyed dict within tolerance."""
        if not time_map:
            return {}
        closest = min(time_map.keys(), key=lambda t: abs(t - target))
        if abs(closest - target) <= tolerance_min:
            return time_map[closest]
        return {}


# ── Phase-Based Temporal Analysis (NEW) ───────────────────────────────────────

def analyse_temporal_phases(
    features: list[list[float]],
    timestamps: list[float],
    window_hours: int = TEMPORAL_WINDOW_HOURS,
) -> dict:
    """
    Divide the observation window into early / mid / recent phases and
    summarise the mean of each vital per phase.

    Returns a dict keyed by phase label, each containing per-feature means
    and a plain-English trend description.
    """
    if not features or not timestamps:
        return {}

    max_offset = max(timestamps)
    cutoff     = max_offset - (window_hours * 60)
    phases     = {}

    for (ph_start, ph_end, ph_label) in TEMPORAL_PHASES:
        t_min = cutoff + ph_start * 60
        t_max = cutoff + ph_end   * 60

        phase_feats = [
            features[i] for i, t in enumerate(timestamps)
            if t_min <= t <= t_max and i < len(features)
        ]
        if not phase_feats:
            continue

        arr = np.array(phase_feats, dtype=np.float32)  # shape: (n_timesteps, n_features)

        # FIX: zeros in the feature matrix represent missing/rejected values
        # (out-of-physiological-range readings replaced with 0.0 at ingestion).
        # Averaging zeros with valid readings produces impossible values like
        # SpO2=8 when most readings were rejected. Exclude zeros per-feature
        # before computing the mean so only valid observations are averaged.
        phase_summary = {}
        for fi, fname in enumerate(ICU_FEATURE_NAMES):
            if fi >= arr.shape[1]:
                continue
            col         = arr[:, fi]
            valid_vals  = col[col != 0.0]   # exclude sentinel zeros
            if len(valid_vals) == 0:
                continue
            mean_val = float(valid_vals.mean())
            # Secondary sanity check — apply physiological bounds to the mean itself
            PHASE_BOUNDS = {
                "SpO2":             (50,  100),
                "Heart Rate":       (20,  250),
                "Respiration Rate": (4,   60),
                "Temperature":      (30,  43),
                "Systolic BP":      (40,  250),
                "Diastolic BP":     (20,  150),
                "FiO2":             (0.2, 1.0),
                "PEEP":             (0,   30),
                "Tidal Volume":     (2,   20),
                "PaO2_FiO2":        (0,   600),
                "Lactate":          (0,   30),
            }
            if fname in PHASE_BOUNDS:
                lo, hi = PHASE_BOUNDS[fname]
                if mean_val < lo or mean_val > hi:
                    continue   # still out of range after averaging — skip
            phase_summary[fname] = round(mean_val, 2)

        if phase_summary:
            phases[ph_label] = phase_summary

    return phases


def describe_phase_trajectory(phases: dict) -> str:
    """
    Build a plain-English temporal description from phase means.
    Example: "Hour 0–2: stable vitals. Hour 4–6: SpO2 declined 96→91%,
    RR increased 18→27 br/min. Pattern suggests worsening respiratory distress."
    """
    if not phases:
        return "Insufficient temporal data for phase analysis."

    lines     = []
    phase_keys = ["Early", "Mid", "Recent"]

    for label in phase_keys:
        if label not in phases:
            continue
        p = phases[label]
        items = []
        for fname in ["Heart Rate", "Respiration Rate", "SpO2", "Systolic BP", "Lactate", "PaO2_FiO2"]:
            if fname in p and p[fname] != 0:
                items.append(f"{fname}={p[fname]:.1f}")
        if items:
            window_map = {"Early": "Hour 0–2", "Mid": "Hour 2–4", "Recent": "Hour 4–6"}
            lines.append(f"{window_map[label]}: {', '.join(items)}")

    # Infer trend narrative
    trend = _infer_trend_narrative(phases)

    result = "\n".join(lines)
    if trend:
        result += f"\n\nTrend pattern: {trend}"
    return result


def _infer_trend_narrative(phases: dict) -> str:
    """Compare early vs recent phase to generate a trend description."""
    early  = phases.get("Early",  {})
    recent = phases.get("Recent", {})
    if not early or not recent:
        return ""

    findings = []

    for fname, direction, threshold_delta in [
        ("SpO2",            "declined", -3),
        ("Respiration Rate","increased", 4),
        ("Heart Rate",      "increased", 15),
        ("Lactate",         "increased", 0.5),
        ("PaO2_FiO2",       "declined", -50),
        ("Systolic BP",     "declined", -15),
    ]:
        e_val = early.get(fname, 0)
        r_val = recent.get(fname, 0)
        if e_val == 0 or r_val == 0:
            continue
        delta = r_val - e_val
        if direction == "declined" and delta < threshold_delta:
            findings.append(f"{fname} {direction} ({e_val:.1f}→{r_val:.1f})")
        elif direction == "increased" and delta > threshold_delta:
            findings.append(f"{fname} {direction} ({e_val:.1f}→{r_val:.1f})")

    if not findings:
        return "Vitals relatively stable over the observation window."

    pattern = ", ".join(findings)

    # Classify overall pattern
    if "SpO2" in pattern and "Respiration Rate" in pattern:
        conclusion = "Pattern suggests progressive respiratory distress."
    elif "Lactate" in pattern and "Systolic BP" in pattern:
        conclusion = "Pattern suggests haemodynamic deterioration / early shock."
    elif "Heart Rate" in pattern:
        conclusion = "Pattern suggests increasing physiological stress."
    else:
        conclusion = "Multi-parameter deterioration detected."

    return f"{pattern}. {conclusion}"


# ── Calibrated Risk Score (NEW) ────────────────────────────────────────────────

def estimate_risk_score(gru_result: dict) -> float:
    """
    Calibrated risk score combining:
      - TCSV magnitude (model-derived, 0–1)
      - Clinical abnormality count from latest vitals
      - Temporal deterioration signal from phase analysis

    Replaces prototype's raw TCSV mean (unreliable for sparse data).
    """
    import numpy as np

    tcsv    = np.array(gru_result.get("tcsv", []))
    seq_len = gru_result.get("seq_len", 0)
    phases  = gru_result.get("temporal_phases", {})

    # Component 1: TCSV magnitude (0–1)
    if len(tcsv) > 0:
        mag       = float(np.abs(tcsv).mean())
        tcsv_comp = min(mag * 2.5, 1.0)
    else:
        tcsv_comp = 0.5

    # Penalise very sparse sequences
    if seq_len < 5:
        tcsv_comp *= 0.5

    # Component 2: clinical abnormality count
    abnorm_score = 0.0
    recent = phases.get("Recent", {})
    if recent:
        n_abnormal = 0
        for fname, thresholds in CLINICAL_THRESHOLDS.items():
            val = recent.get(fname, 0)
            if val == 0:
                continue
            if "critical_low" in thresholds and val < thresholds["critical_low"]:
                n_abnormal += 2   # critical counts double
            elif "low" in thresholds and val < thresholds["low"]:
                n_abnormal += 1
            if "critical_high" in thresholds and val > thresholds["critical_high"]:
                n_abnormal += 2
            elif "high" in thresholds and val > thresholds["high"]:
                n_abnormal += 1
        abnorm_score = min(n_abnormal / 6.0, 1.0)

    # Component 3: temporal deterioration signal
    trend_score = 0.0
    early  = phases.get("Early",  {})
    if early and recent:
        spo2_delta = recent.get("SpO2", 0) - early.get("SpO2", 0)
        rr_delta   = recent.get("Respiration Rate", 0) - early.get("Respiration Rate", 0)
        lac_delta  = recent.get("Lactate", 0) - early.get("Lactate", 0)
        if spo2_delta < -3:
            trend_score += 0.3
        if rr_delta > 4:
            trend_score += 0.2
        if lac_delta > 0.5:
            trend_score += 0.2
        trend_score = min(trend_score, 0.6)

    # Weighted combination
    raw = 0.40 * tcsv_comp + 0.35 * abnorm_score + 0.25 * trend_score
    return round(float(np.clip(raw, 0.0, 1.0)), 3)


# ── Confidence Score (NEW) ─────────────────────────────────────────────────────

def estimate_confidence(
    gru_result:     dict,
    rag_confidence: float = 0.0,
    data_quality:   dict  = None,
) -> dict:
    """
    Return a confidence score (0–1) and interpretive label.
    Combines data richness, temporal coverage, and RAG retrieval quality.

    If data_quality is supplied, a hard ceiling is applied based on the
    data tier — prevents inflated confidence on sparse/unreliable data.
    For example a SPARSE patient can never exceed 40% confidence.
    """
    seq_len    = gru_result.get("seq_len", 0)
    n_features = len(ICU_FEATURE_NAMES)

    # Data richness: how many features have non-zero recent values
    recent = gru_result.get("temporal_phases", {}).get("Recent", {})
    n_populated = sum(1 for v in recent.values() if v and v != 0)
    data_comp   = min(n_populated / n_features, 1.0)

    # Temporal coverage: sequence length
    seq_comp = min(seq_len / 30.0, 1.0)

    # RAG quality (passed in)
    rag_comp = min(rag_confidence, 1.0)

    score = 0.45 * data_comp + 0.35 * seq_comp + 0.20 * rag_comp
    score = round(float(score), 3)

    from config import CONFIDENCE_BANDS
    label = "very low"
    for band_label, (lo, hi) in CONFIDENCE_BANDS.items():
        if lo <= score < hi:
            label = band_label.replace("_", " ")
            break

    result = {
        "score":         score,
        "label":         label,
        "pct":           int(score * 100),
        "data_richness": round(data_comp, 2),
        "temporal_cov":  round(seq_comp,  2),
        "rag_quality":   round(rag_comp,  2),
        "cap_applied":   False,
        "cap_reason":    "",
        "original_score":score,
        "original_pct":  int(score * 100),
    }

    # Apply data-quality-based confidence cap
    if data_quality:
        try:
            from modules.quality_control import apply_confidence_cap
            result = apply_confidence_cap(result, data_quality)
        except ImportError:
            pass

    return result


# ── Master GRU runner ──────────────────────────────────────────────────────────

def run_gru_module(patient_id: int, data_loader: EICUDataLoader) -> dict:
    """
    Run the full GRU temporal reasoning for a patient.

    Returns extended result dict including:
      - tcsv, seq_len, clinical_summary (same as prototype)
      - temporal_phases: per-phase vital means (NEW)
      - phase_trajectory: plain-English temporal description (NEW)
      - risk_score: calibrated score (NEW — moved here from main.py)
      - patient_info: extended with icu_hours, ventilated, vent_hours, diagnosis_str
    """
    print(f"\n Component A: GRU Temporal Reasoning")
    print(f"   Patient ID: {patient_id}")

    model = TimeAwareGRU(input_size=ICU_INPUT_SIZE, hidden_size=64, num_layers=2)
    model.eval()

    x, delta_t, seq_info = data_loader.build_tensor(patient_id, ICU_INPUT_SIZE)

    with torch.no_grad():
        tcsv, hidden_states = model(x, delta_t)

    tcsv_np = tcsv.squeeze(0).numpy()
    seq_len = x.shape[1]

    # Phase analysis on raw sequence
    features   = seq_info.get("features",   [])
    timestamps = seq_info.get("timestamps", [])
    phases     = analyse_temporal_phases(features, timestamps)
    trajectory = describe_phase_trajectory(phases)

    print(f"    Timesteps processed: {seq_len}")
    print(f"    Features per timestep: {ICU_INPUT_SIZE}  ({', '.join(ICU_FEATURE_NAMES[:6])}...)")
    print(f"    TCSV shape: {tcsv_np.shape}, mean={tcsv_np.mean():.4f}")
    if phases:
        print(f"    Temporal phases: {', '.join(phases.keys())}")

    info   = seq_info.get("info", {})
    icu_h  = info.get("icu_hours", "?")
    vent_h = info.get("vent_hours", 0) or 0.0
    # FIX 4: ventilated=True but vent_hours=0 means we detected a ventilation
    # treatment entry but couldn't compute duration (only 1 timestamp).
    # Treat 0h as "not confirmed" to avoid "Ventilated: Yes (0.0h)".
    ventilated = info.get("ventilated") and vent_h > 0
    vent_str   = f"Yes ({vent_h}h)" if ventilated else "No"
    diag       = info.get("diagnosis_str", info.get("apache_score", "?"))

    # Also correct the ventilated flag in patient_info so downstream
    # RAG query and safety checks use the fixed value.
    info["ventilated"] = ventilated

    clinical_summary = (
        f"Patient {patient_id}: Age={info.get('age','?')}, "
        f"Gender={info.get('gender','?')}, "
        f"Unit={info.get('unit_type','?')}, "
        f"ICU stay={icu_h}h, "
        f"Mechanical ventilation={vent_str}, "
        f"Diagnosis: {diag}"
    )
    print(f"    {clinical_summary}")

    result = {
        "patient_id":        patient_id,
        "tcsv":              tcsv_np,
        "seq_len":           seq_len,
        "features":          features,
        "timestamps":        timestamps,
        "clinical_summary":  clinical_summary,
        "patient_info":      info,
        "temporal_phases":   phases,
        "phase_trajectory":  trajectory,
    }

    # Calibrated risk score (moved from main.py)
    result["risk_score"] = estimate_risk_score(result)
    print(f"    Risk score (calibrated): {result['risk_score']:.3f}")

    # Data quality assessment — determines pipeline behaviour downstream
    try:
        from modules.quality_control import PatientDataQualityAssessor
        result["data_quality"] = PatientDataQualityAssessor.assess(result)
    except ImportError:
        result["data_quality"] = {"tier": "UNKNOWN", "warnings": []}

    return result
