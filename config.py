"""
config.py -- TA-CDSS Central Configuration (Research-Grade)

Improvements over prototype:
  - Extended ICU feature set (ventilator, oxygenation, metabolic)
  - Confidence score thresholds
  - Temporal analysis windows
  - Safety rule thresholds
"""

# ── Gemini API ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyAjWvuoaSHRYEoBAlon32ZBi9S2OUqcYi8"
GEMINI_MODEL   = "gemini-2.5-flash"
FREE_TIER_RPM  = 15
FREE_TIER_RPD  = 1500

# ── LM Studio (local fallback) ─────────────────────────────────────────────────
LM_STUDIO_URL     = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_TIMEOUT = 300

AVAILABLE_MODELS = {
    "1": ("qwen2.5-7b-instruct-1m",            "Qwen 2.5 7B Instruct 1M"),
    "2": ("deepseek-r1-distill-qwen-7b",        "DeepSeek R1 Distill Qwen 7B"),
    "3": ("mistralai/mistral-7b-instruct-v0.3", "Mistral 7B Instruct v0.3"),
    "4": ("llama-3.2-3b-instruct",              "LLaMA 3.2 3B Instruct"),
}

# ── Extended ICU Feature Set ───────────────────────────────────────────────────
# Prototype used 6 basic vitals. Research-grade adds ventilator + metabolic vars.
ICU_FEATURE_NAMES = [
    # Basic vitals (indices 0-5) — same as prototype
    "Heart Rate",
    "Respiration Rate",
    "SpO2",
    "Temperature",
    "Systolic BP",
    "Diastolic BP",
    # Ventilator parameters (indices 6-8)
    "FiO2",         # Fraction of inspired oxygen (0.21–1.0)
    "PEEP",         # Positive end-expiratory pressure (cmH2O)
    "Tidal Volume", # mL/kg ideal body weight
    # Oxygenation / metabolic (indices 9-10)
    "PaO2_FiO2",    # P/F ratio — key ARDS marker
    "Lactate",      # mmol/L — sepsis / shock severity
]
ICU_INPUT_SIZE = len(ICU_FEATURE_NAMES)  # 11

# ── Clinical Thresholds (used by safety validator & explainer) ─────────────────
CLINICAL_THRESHOLDS = {
    "Heart Rate":      {"low": 50,   "high": 100,  "critical_high": 130, "unit": "bpm"},
    "Respiration Rate":{"low": 10,   "high": 25,   "critical_high": 30,  "unit": "br/min"},
    "SpO2":            {"low": 94,   "critical_low": 90, "high": 100,     "unit": "%"},
    "Temperature":     {"low": 36.0, "high": 38.3, "critical_high": 39.5,"unit": "°C"},
    "Systolic BP":     {"low": 90,   "critical_low": 80, "high": 140,     "unit": "mmHg"},
    "Diastolic BP":    {"low": 60,   "high": 90,   "unit": "mmHg"},
    "FiO2":            {"low": 0.21, "high": 0.60, "critical_high": 0.80,"unit": "fraction"},
    "PEEP":            {"low": 0,    "high": 8,    "critical_high": 15,  "unit": "cmH2O"},
    "Tidal Volume":    {"low": 4,    "high": 8,    "unit": "mL/kg IBW"},
    "PaO2_FiO2":       {"critical_low": 100, "low": 200, "high": 500,    "unit": "mmHg"},
    "Lactate":         {"low": 0,    "high": 2.0,  "critical_high": 4.0, "unit": "mmol/L"},
}

# ── Safety Rules (for agentic validator) ──────────────────────────────────────
# Each rule: (feature, operator, threshold, clinical_meaning)
SAFETY_RULES = [
    ("SpO2",            "<",  92,   "Severe hypoxaemia — extubation unsafe"),
    ("Respiration Rate",">",  30,   "Respiratory distress — weaning failure risk"),
    ("Heart Rate",      ">",  120,  "Tachycardia — haemodynamic stress"),
    ("Systolic BP",     "<",  80,   "Severe hypotension — vasopressor review needed"),
    ("Lactate",         ">",  4.0,  "High lactate — shock / tissue hypoperfusion"),
    ("PaO2_FiO2",       "<",  150,  "Moderate-severe ARDS — prone positioning threshold"),
    ("FiO2",            ">",  0.60, "High FiO2 requirement — oxygen toxicity risk"),
]

# ── Risk Thresholds ────────────────────────────────────────────────────────────
RISK_HIGH_THRESHOLD   = 0.70
RISK_MEDIUM_THRESHOLD = 0.40

# ── Confidence Score Bands ─────────────────────────────────────────────────────
CONFIDENCE_BANDS = {
    "high":     (0.70, 1.00),
    "moderate": (0.45, 0.70),
    "low":      (0.20, 0.45),
    "very_low": (0.00, 0.20),
}

# ── Temporal Analysis Windows ──────────────────────────────────────────────────
TEMPORAL_WINDOW_HOURS = 6    # Primary trend window
TEMPORAL_PHASES = [          # Sub-windows for phase analysis
    (0, 2,  "Early"),
    (2, 4,  "Mid"),
    (4, 6,  "Recent"),
]
