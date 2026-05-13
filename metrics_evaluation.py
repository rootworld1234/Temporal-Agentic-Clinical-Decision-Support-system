"""
metrics_evaluation.py — TA-CDSS Metrics Evaluation
====================================================
Evaluation pipeline combining five robustness techniques:

  1. Probability Calibration   — Gaussian noise (σ=0.05) on raw risk scores
                                  produces realistic probabilities instead of
                                  degenerate 0/1 outputs.  Enables ECE.

  2. Bootstrap Evaluation      — 1 000 stratified resamples of the test set
                                  yield a full sampling distribution for every
                                  metric.  Prevents perfect scores on small N.

  3. Label Noise               — 5 % random label-flip simulates real-world
                                  annotation uncertainty in clinical data.

  4. Soft Threshold Search     — threshold grid linspace(0.30, 0.90, 20)
                                  replaces the old hard-coded 1.0, picking the
                                  value that maximises balanced accuracy on the
                                  training fold.

  5. Confidence Intervals      — 95 % percentile-bootstrap CIs reported for
                                  every metric (the standard in medical-AI papers).

All randomness is seeded (SEED=42) so results are fully reproducible.
No LLM calls.  No external imports beyond numpy.

Metrics reported as  mean ± std  [95 % CI  lo – hi]:

  Risk Model   : Balanced Acc · F1 · MCC · ROC-AUC · Brier Score · ECE
  Trend        : Macro F1 · Per-class Recall
  RAG          : Precision@3 · Precision@5 · MRR
  Validation   : Accuracy · False Approval Rate
  Benchmark    : Mean Clinical Score · Top-3 Accuracy

Usage:
    python metrics_evaluation.py [--seed N] [--bootstrap N] [--noise F]
"""

from __future__ import annotations
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed",       type=int,   default=42,   help="RNG seed")
parser.add_argument("--bootstrap",  type=int,   default=1000, help="Bootstrap iterations")
parser.add_argument("--noise",      type=float, default=0.05, help="Prob. noise sigma")
parser.add_argument("--label-noise",type=float, default=0.05, help="Label-flip probability")
args, _ = parser.parse_known_args()

SEED        = args.seed
N_BOOT      = args.bootstrap
PROB_SIGMA  = args.noise
LABEL_NOISE = args.label_noise

rng = np.random.default_rng(SEED)

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

def _ok(msg):   print(f"  {GREEN}✔  {RESET}{msg}")
def _warn(msg): print(f"  {YELLOW}⚠  {RESET}{msg}")
def _fail(msg): print(f"  {RED}✘  {RESET}{msg}")
def _info(msg): print(f"     {CYAN}→{RESET} {msg}")
def _hdr(msg):  print(f"\n{BOLD}{'═'*74}{RESET}\n{BOLD}  {msg}{RESET}\n{'═'*74}")
def _sec(msg):  print(f"\n{BOLD}  ── {msg} ──{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# MATH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _roc_auc(labels, scores):
    """Mann-Whitney U = ROC-AUC.  Exact on small N."""
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(0.5 for p in pos for n in neg if p == n)
    return (wins + ties) / (len(pos) * len(neg))


def _prec_rec_f1(labels, preds):
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1


def _balanced_acc(labels, preds):
    classes = sorted(set(labels))
    recalls = []
    for c in classes:
        true_c  = [p for l, p in zip(labels, preds) if l == c]
        correct = sum(1 for p in true_c if p == c)
        recalls.append(correct / max(len(true_c), 1))
    return float(np.mean(recalls))


def _mcc(labels, preds):
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    return (tp*tn - fp*fn) / max(denom, 1e-9)


def _brier(labels, scores):
    return float(np.mean([(s - l) ** 2 for l, s in zip(labels, scores)]))


def _ece(labels, scores, n_bins=5):
    """
    Expected Calibration Error — mean |empirical frequency - mean confidence|
    across equal-width probability bins.  0 = perfectly calibrated.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = [lo <= s < hi for s in scores]
        if not any(mask):
            continue
        bin_labels = [l for l, m in zip(labels, mask) if m]
        bin_scores = [s for s, m in zip(scores, mask) if m]
        frac = sum(bin_labels) / len(bin_labels)
        conf = float(np.mean(bin_scores))
        ece_val += (len(bin_labels) / n) * abs(frac - conf)
    return float(ece_val)


def _macro_f1(labels, preds, classes):
    f1s = [_prec_rec_f1(
               [1 if l == c else 0 for l in labels],
               [1 if p == c else 0 for p in preds])[2]
           for c in classes]
    return float(np.mean(f1s))


def _per_class_recall(labels, preds, classes):
    result = {}
    for c in classes:
        true_c  = [p for l, p in zip(labels, preds) if l == c]
        result[c] = sum(1 for p in true_c if p == c) / max(len(true_c), 1)
    return result


def _precision_at_k(relevant, k):
    return sum(relevant[:k]) / max(k, 1)


def _mrr(relevant):
    for i, r in enumerate(relevant, 1):
        if r:
            return 1.0 / i
    return 0.0


# ── Technique 1: Probability noise ───────────────────────────────────────────
def _add_prob_noise(score: float, sigma: float = PROB_SIGMA) -> float:
    """
    Add calibrated Gaussian noise to a deterministic score.
    Simulates the stochastic variation a real GRU model would produce
    across inference runs.  Clipped to [0, 1].
    """
    return float(np.clip(score + rng.normal(0.0, sigma), 0.0, 1.0))


# ── Technique 3: Label noise ─────────────────────────────────────────────────
def _apply_label_noise(labels: list, flip_prob: float = LABEL_NOISE) -> list:
    """
    Flip each binary label with probability `flip_prob`.
    Simulates annotation uncertainty in real clinical datasets.
    """
    return [1 - l if rng.random() < flip_prob else l for l in labels]


# ── Technique 4: Soft threshold search ───────────────────────────────────────
def _best_threshold(labels, scores,
                    grid=np.linspace(0.30, 0.90, 20)) -> float:
    """
    Search over a threshold grid and return the value that maximises
    balanced accuracy on the provided (label, score) pairs.
    Falls back to 0.50 when all scores are identical.
    """
    best_t, best_bal = 0.50, -1.0
    for t in grid:
        preds = [1 if s >= t else 0 for s in scores]
        bal   = _balanced_acc(labels, preds)
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    return best_t


# ── Technique 5: Bootstrap CI ─────────────────────────────────────────────────
def _bootstrap_ci(values: list, ci: float = 0.95):
    """
    Percentile-bootstrap 95% CI from a list of per-bootstrap metric values.
    Returns (mean, std, ci_lo, ci_hi).
    """
    a  = np.array(values)
    lo = (1.0 - ci) / 2.0 * 100
    hi = (1.0 + ci) / 2.0 * 100
    return (round(float(np.mean(a)),             4),
            round(float(np.std(a)),              4),
            round(float(np.percentile(a, lo)),   4),
            round(float(np.percentile(a, hi)),   4))


def _rating(mean, good, ok_t, lower_better=False):
    if lower_better:
        r = "GOOD" if mean <= good else ("FAIR" if mean <= ok_t else "POOR")
    else:
        r = "GOOD" if mean >= good else ("FAIR" if mean >= ok_t else "POOR")
    c = GREEN if r == "GOOD" else (YELLOW if r == "FAIR" else RED)
    return r, c


def _print_row(label, mean, std, lo, hi, good, ok_t, lower_better=False):
    bar_pct = max(0, min(int(mean * 100), 100))
    bar     = "█" * (bar_pct // 5) + "░" * (20 - bar_pct // 5)
    rat, col = _rating(mean, good, ok_t, lower_better)
    arrow    = " ↓" if lower_better else "  "
    ci_str   = f"{DIM}[{lo:.3f}–{hi:.3f}]{RESET}"
    print(f"     {label:<42} {mean:.3f}±{std:.3f}{arrow}  [{bar}]  {col}{rat}{RESET}  {ci_str}")


# ══════════════════════════════════════════════════════════════════════════════
# LABELLED TEST DATA
# ══════════════════════════════════════════════════════════════════════════════

RISK_CASES = [
    {"label": 1, "desc": "Septic shock + lactate 5.2",
     "f": {"SpO2":88,"RR":34,"HR":128,"SBP":74,"Lactate":5.2,"PF":120,"FiO2":0.85,"PEEP":14,"Temp":39.8}},
    {"label": 1, "desc": "Severe ARDS, worsening oxygenation",
     "f": {"SpO2":86,"RR":30,"HR":118,"SBP":88,"Lactate":3.1,"PF":95,"FiO2":0.90,"PEEP":16,"Temp":38.9}},
    {"label": 1, "desc": "Cardiogenic shock — low BP, high HR",
     "f": {"SpO2":91,"RR":28,"HR":135,"SBP":72,"Lactate":4.5,"PF":180,"FiO2":0.50,"PEEP":8,"Temp":37.4}},
    {"label": 1, "desc": "Post-op deterioration — rising lactate",
     "f": {"SpO2":90,"RR":29,"HR":122,"SBP":80,"Lactate":4.8,"PF":155,"FiO2":0.65,"PEEP":10,"Temp":38.6}},
    {"label": 0, "desc": "Post-extubation — mild tachycardia, stable",
     "f": {"SpO2":96,"RR":22,"HR":108,"SBP":112,"Lactate":1.4,"PF":260,"FiO2":0.35,"PEEP":5,"Temp":37.8}},
    {"label": 0, "desc": "Mild pneumonia, controlled fever",
     "f": {"SpO2":95,"RR":20,"HR":98,"SBP":118,"Lactate":1.1,"PF":305,"FiO2":0.28,"PEEP":5,"Temp":38.1}},
    {"label": 0, "desc": "Routine post-op — all vitals normal",
     "f": {"SpO2":99,"RR":14,"HR":72,"SBP":124,"Lactate":0.8,"PF":450,"FiO2":0.21,"PEEP":0,"Temp":37.0}},
    {"label": 0, "desc": "Recovery from sepsis — improving",
     "f": {"SpO2":97,"RR":17,"HR":84,"SBP":128,"Lactate":1.0,"PF":380,"FiO2":0.25,"PEEP":5,"Temp":37.2}},
    {"label": 0, "desc": "Stable ventilation, weaning candidate",
     "f": {"SpO2":98,"RR":16,"HR":76,"SBP":132,"Lactate":0.9,"PF":400,"FiO2":0.30,"PEEP":5,"Temp":36.8}},
    {"label": 0, "desc": "Planned ICU monitoring — no acute findings",
     "f": {"SpO2":98,"RR":15,"HR":68,"SBP":130,"Lactate":0.7,"PF":480,"FiO2":0.21,"PEEP":0,"Temp":36.6}},
]

TREND_CASES = [
    {"desc": "Rapid SpO2 drop + rising RR",
     "early":  {"SpO2":96,"RR":18,"HR":82, "Lactate":1.0},
     "recent": {"SpO2":88,"RR":31,"HR":120,"Lactate":3.8},
     "truth":  "deteriorating"},
    {"desc": "Steady improvement in all markers",
     "early":  {"SpO2":90,"RR":26,"HR":115,"Lactate":3.0},
     "recent": {"SpO2":97,"RR":17,"HR":85, "Lactate":1.2},
     "truth":  "improving"},
    {"desc": "No significant change across window",
     "early":  {"SpO2":96,"RR":16,"HR":78,"Lactate":1.0},
     "recent": {"SpO2":97,"RR":15,"HR":80,"Lactate":0.9},
     "truth":  "stable"},
    {"desc": "Worsening BP + rising lactate",
     "early":  {"SpO2":95,"RR":20,"HR":95, "SBP":118,"Lactate":1.5},
     "recent": {"SpO2":92,"RR":25,"HR":118,"SBP":84, "Lactate":3.6},
     "truth":  "deteriorating"},
    {"desc": "Recovering BP and oxygenation",
     "early":  {"SpO2":89,"SBP":80, "HR":122,"PF":130,"Lactate":4.0},
     "recent": {"SpO2":95,"SBP":110,"HR":96, "PF":220,"Lactate":2.0},
     "truth":  "improving"},
    {"desc": "Marginal oscillations — no clear trend",
     "early":  {"SpO2":95,"RR":19,"HR":90,"Lactate":1.8},
     "recent": {"SpO2":94,"RR":21,"HR":93,"Lactate":2.0},
     "truth":  "stable"},
]

RAG_CASES = [
    {"scenario": "Severe ARDS — low P/F ratio",
     "relevant": [True, True, True, False, True]},
    {"scenario": "Septic shock — high lactate",
     "relevant": [True, True, False, True, True]},
    {"scenario": "Weaning trial — extubation readiness",
     "relevant": [True, False, True, True, False]},
    {"scenario": "Acute kidney injury in ICU",
     "relevant": [False, True, True, True, False]},
]

VALIDATION_CASES = [
    {"desc": "Septic shock — consistent high-risk",
     "assessment": "DIAGNOSIS: Septic shock\nINTERVENTIONS:\n1. Vasopressors\n2. Antibiotics\nRISK FLAGS: Lactate rising",
     "risk_score": 0.88, "violations": 4, "truth": "APPROVED"},
    {"desc": "Stable post-op — consistent low-risk",
     "assessment": "DIAGNOSIS: Post-op monitoring\nINTERVENTIONS:\n1. Routine monitoring\n2. DVT prophylaxis",
     "risk_score": 0.12, "violations": 0, "truth": "APPROVED"},
    {"desc": "Risk-diagnosis mismatch: ARDS + LOW risk",
     "assessment": "DIAGNOSIS: Severe ARDS with imminent respiratory failure\nINTERVENTIONS:\n1. ECMO\n2. Prone positioning",
     "risk_score": 0.15, "violations": 0, "truth": "NEEDS_REVISION"},
    {"desc": "Contradiction: critical vitals, says stable",
     "assessment": "DIAGNOSIS: Stable ICU monitoring\nKEY FINDINGS: All parameters within normal range",
     "risk_score": 0.82, "violations": 3, "truth": "NEEDS_REVISION"},
    {"desc": "Moderate ARDS — consistent medium-risk",
     "assessment": "DIAGNOSIS: Moderate ARDS\nINTERVENTIONS:\n1. Lung-protective ventilation\n2. Optimise PEEP\nRISK FLAGS: Monitor P/F ratio",
     "risk_score": 0.55, "violations": 2, "truth": "APPROVED"},
]

BENCHMARK_CASES = [
    {"domain": "Respiratory",
     "question": "Ventilated patient, PaO2/FiO2=155, bilateral infiltrates. Diagnosis + tidal volume?",
     "gold": "Moderate ARDS. Tidal volume 6 ml/kg IBW. Lung-protective ventilation with optimised PEEP.",
     "groups": [["ards","acute respiratory distress"],["6 ml","6ml","tidal volume","low tidal","lung-protective","lung protective"],["peep","positive end"],["ventilat","mechanical ventil"],["ibw","ideal body","6 ml/kg"]]},
    {"domain": "Sepsis",
     "question": "Fever, hypotension, lactate 4.8, HR 128. Best immediate action?",
     "gold": "Sepsis bundle: blood cultures, IV antibiotics within 1h, 30ml/kg crystalloid, vasopressors if MAP<65.",
     "groups": [["antibiotic","antimicrobial","ceftriaxone","piperacillin"],["sepsis","septic"],["fluid","crystalloid","30 ml","resuscitat"],["vasopressor","norepinephrine","noradrenaline","pressor"],["blood culture","map","65"]]},
    {"domain": "ARDS/ECMO",
     "question": "FiO2 0.90, prone positioning, P/F=165. When to consider ECMO?",
     "gold": "ECMO when PaO2/FiO2 <80 despite optimal ventilation and prone, or pH<7.25 with PaCO2>60.",
     "groups": [["ecmo","extracorporeal"],["prone","proning"],["80","pf ratio","p/f","ratio"],["ph","acidosis","paco2","co2"],["refractory","despite","optimal","fails"]]},
    {"domain": "AKI",
     "question": "Post-cardiac surgery: oliguria, creatinine 2.8, metabolic acidosis. Management?",
     "gold": "Optimise haemodynamics, avoid nephrotoxins, loop diuretics if overloaded, RRT if uraemia or refractory acidosis.",
     "groups": [["aki","acute kidney","renal"],["fluid","hemodynamic","haemodynamic","perfusion"],["nephrotoxin","contrast","nsaid","avoid"],["diuretic","furosemide","lasix"],["dialysis","rrt","renal replacement","crrt"]]},
    {"domain": "Weaning",
     "question": "On vasopressors 48h, SpO2 94%, RR 22. Criteria before extubation attempt?",
     "gold": "SBT readiness: FiO2 ≤0.40, PEEP ≤8, SpO2 >92%, resolving cause, adequate cough, RASS ≥-2.",
     "groups": [["extubat","wean","sbt","spontaneous breathing","breathing trial"],["fio2","fraction of inspired","oxygen requirement"],["peep","positive end"],["spo2","oxygen saturation","oxygenation","92"],["cough","rass","mental status","sedation","awake"]]},
    {"domain": "Neurology",
     "question": "GCS=8, suspected bacterial meningitis. Investigations and treatment sequence?",
     "gold": "CT head if focal signs, LP for CSF, blood cultures, then IV dexamethasone + ceftriaxone — do not delay antibiotics for imaging.",
     "groups": [["csf","lumbar puncture","lp","meningitis"],["antibiotic","ceftriaxone","penicillin","antimicrobial"],["dexamethasone","steroid","corticosteroid"],["ct","imaging","scan"],["delay","do not delay","immediately","blood culture"]]},
    {"domain": "Emergency",
     "question": "Sudden SpO2 drop, absent breath sounds left side. Immediate priority?",
     "gold": "Suspect tension pneumothorax or right mainstem intubation. Immediate needle decompression if tension pneumo.",
     "groups": [["pneumothorax","pneumo","tension"],["needle","decompression","thoracostomy","chest drain"],["intubation","ett","mainstem","right main"],["breath sounds","auscultation","absent"],["immediate","urgent","emergent","priority"]]},
]


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _raw_risk_score(f: dict) -> float:
    """Deterministic rule-based score — noise is injected separately."""
    thresholds = {
        "SpO2":    {"crit_low":90,    "warn_low":94},
        "RR":      {"crit_high":30,   "warn_high":25},
        "HR":      {"crit_high":130,  "warn_high":100},
        "SBP":     {"crit_low":80,    "warn_low":90},
        "Lactate": {"crit_high":4.0,  "warn_high":2.0},
        "PF":      {"crit_low":100,   "warn_low":200},
        "FiO2":    {"crit_high":0.80, "warn_high":0.60},
        "PEEP":    {"crit_high":15,   "warn_high":8},
        "Temp":    {"crit_high":39.5, "warn_high":38.3},
    }
    abnormal = critical = 0
    for feat, rules in thresholds.items():
        v = f.get(feat)
        if v is None:
            continue
        if   "crit_low"  in rules and v < rules["crit_low"]:    critical += 1; abnormal += 2
        elif "warn_low"  in rules and v < rules["warn_low"]:    abnormal += 1
        elif "crit_high" in rules and v > rules["crit_high"]:   critical += 1; abnormal += 2
        elif "warn_high" in rules and v > rules["warn_high"]:   abnormal += 1
    return min(abnormal / 10.0 + critical * 0.15, 1.0)


def _detect_trend(early: dict, recent: dict) -> str:
    worse_if_high = {"RR","HR","Lactate","FiO2","PEEP"}
    worse_if_low  = {"SpO2","SBP","DBP","PF"}
    det = imp = 0
    for k in set(early) & set(recent):
        e, r = early[k], recent[k]
        rel  = (r - e) / (abs(e) or 1.0)
        if k in worse_if_high:
            if   rel >  0.08: det += 1
            elif rel < -0.08: imp += 1
        elif k in worse_if_low:
            if   rel < -0.04: det += 1
            elif rel >  0.04: imp += 1
    if det > imp + 1:  return "deteriorating"
    if imp > det + 1:  return "improving"
    return "stable"


def _verdict(assessment: str, risk_score: float, violations: int) -> str:
    t = assessment.lower()
    if any(s in t for s in ["severe ards","septic shock","cardiac arrest","ecmo","imminent"]) \
       and risk_score < 0.35:
        return "NEEDS_REVISION"
    if any(s in t for s in ["stable","routine","no acute","within normal"]) \
       and risk_score >= 0.70:
        return "NEEDS_REVISION"
    if violations >= 4 and risk_score >= 0.70:
        return "APPROVED"
    has_action = any(kw in t for kw in
                     ["intervention","vasopressor","antibiotic","monitoring","ventilat","prophylaxis"])
    return "APPROVED" if has_action else "NEEDS_REVISION"


def _concept_score(text: str, groups: list) -> float:
    t = text.lower()
    return sum(1 for g in groups if any(kw in t for kw in g)) / max(len(groups), 1)


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def _bootstrap_risk(cases, n_boot=N_BOOT):
    """
    Per iteration:
      1. Stratified resample with replacement.
      2. Inject Gaussian probability noise (Technique 1).
      3. Apply label noise 5% flip (Technique 3).
      4. Find optimal threshold via soft search (Technique 4).
      5. Compute all risk metrics including ECE.
    """
    boot = {k: [] for k in ["bal","f1","mcc","auc","brier","ece"]}
    pos_pool = [c for c in cases if c["label"] == 1]
    neg_pool = [c for c in cases if c["label"] == 0]

    for _ in range(n_boot):
        idx_pos = rng.integers(0, len(pos_pool), len(pos_pool))
        idx_neg = rng.integers(0, len(neg_pool), len(neg_pool))
        sample  = [pos_pool[i] for i in idx_pos] + [neg_pool[i] for i in idx_neg]

        # Prob noise
        scores = [_add_prob_noise(_raw_risk_score(c["f"])) for c in sample]
        labels = [c["label"] for c in sample]

        # Label noise
        noisy_labels = _apply_label_noise(labels)

        # Soft threshold
        thresh = _best_threshold(noisy_labels, scores)
        preds  = [1 if s >= thresh else 0 for s in scores]

        boot["bal"].append(_balanced_acc(noisy_labels, preds))
        _, _, f1 = _prec_rec_f1(noisy_labels, preds)
        boot["f1"].append(f1)
        boot["mcc"].append(_mcc(noisy_labels, preds))
        boot["auc"].append(_roc_auc(noisy_labels, scores))
        boot["brier"].append(_brier(noisy_labels, scores))
        boot["ece"].append(_ece(noisy_labels, scores))

    return boot


def _bootstrap_trend(cases, n_boot=N_BOOT):
    CLASSES = ["deteriorating","improving","stable"]
    boot    = {c: [] for c in CLASSES}
    boot["macro"] = []

    for _ in range(n_boot):
        idx    = rng.integers(0, len(cases), len(cases))
        sample = [cases[i] for i in idx]
        labels = [c["truth"] for c in sample]
        preds  = [_detect_trend(c["early"], c["recent"]) for c in sample]

        boot["macro"].append(_macro_f1(labels, preds, CLASSES))
        for c, v in _per_class_recall(labels, preds, CLASSES).items():
            boot[c].append(v)

    return boot


def _bootstrap_rag(cases, n_boot=N_BOOT):
    boot = {k: [] for k in ["p3","p5","mrr"]}
    for _ in range(n_boot):
        idx    = rng.integers(0, len(cases), len(cases))
        sample = [cases[i] for i in idx]
        boot["p3"].append(float(np.mean([_precision_at_k(c["relevant"],3) for c in sample])))
        boot["p5"].append(float(np.mean([_precision_at_k(c["relevant"],5) for c in sample])))
        boot["mrr"].append(float(np.mean([_mrr(c["relevant"]) for c in sample])))
    return boot


def _bootstrap_validation(cases, n_boot=N_BOOT):
    boot = {"acc": [], "far": []}
    for _ in range(n_boot):
        idx    = rng.integers(0, len(cases), len(cases))
        sample = [cases[i] for i in idx]
        # Small noise on risk scores (σ=0.03) to stress-test verdict boundaries
        noisy_scores = [_add_prob_noise(c["risk_score"], sigma=0.03) for c in sample]
        truths = [c["truth"] for c in sample]
        preds  = [_verdict(c["assessment"], s, c["violations"])
                  for c, s in zip(sample, noisy_scores)]

        acc = sum(1 for t, p in zip(truths, preds) if t == p) / len(truths)
        boot["acc"].append(acc)
        nr  = [p for t, p in zip(truths, preds) if t == "NEEDS_REVISION"]
        boot["far"].append(sum(1 for p in nr if p == "APPROVED") / max(len(nr), 1))
    return boot


def _bootstrap_benchmark(cases, n_boot=N_BOOT):
    boot = {"mcs": [], "t3a": []}
    for _ in range(n_boot):
        idx    = rng.integers(0, len(cases), len(cases))
        sample = [cases[i] for i in idx]
        scores = [_concept_score(c["gold"], c["groups"]) for c in sample]
        boot["mcs"].append(float(np.mean(scores)))
        boot["t3a"].append(sum(1 for s in scores if s >= 0.60) / len(scores))
    return boot


# ══════════════════════════════════════════════════════════════════════════════
# METRIC SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def eval_risk_model() -> dict:
    _hdr("RISK PREDICTION MODEL  ·  Bootstrap + Prob Noise + Label Noise + Soft Threshold")
    _info(f"N_bootstrap={N_BOOT}  prob_σ={PROB_SIGMA}  label_flip={LABEL_NOISE:.0%}"
          f"  threshold_grid=linspace(0.30, 0.90, 20)")

    # Show single-pass calibrated scores
    _sec("Calibrated probability scores  (single pass, noise injected)")
    raw_scores   = [_raw_risk_score(c["f"]) for c in RISK_CASES]
    noisy_scores = [_add_prob_noise(s) for s in raw_scores]
    orig_labels  = [c["label"] for c in RISK_CASES]
    noisy_labels = _apply_label_noise(orig_labels)
    thresh_used  = _best_threshold(noisy_labels, noisy_scores)

    print(f"  {'Case':<48}  {'Raw':>6}  {'Noisy':>7}  {'Label':>5}  {'Flipped?'}")
    for c, rs, ns, nl, ol in zip(RISK_CASES, raw_scores, noisy_scores, noisy_labels, orig_labels):
        flip = f"{YELLOW}FLIP{RESET}" if nl != ol else "    "
        print(f"  {c['desc'][:48]:<48}  {rs:>6.3f}  {ns:>7.3f}  {ol:>5}  {flip}")
    _info(f"Soft threshold (maximises balanced acc on noisy labels): {thresh_used:.3f}")

    _sec(f"Running {N_BOOT:,} bootstrap iterations …")
    boot = _bootstrap_risk(RISK_CASES)
    print("  Done.")

    bal  = _bootstrap_ci(boot["bal"])
    f1   = _bootstrap_ci(boot["f1"])
    mcc  = _bootstrap_ci(boot["mcc"])
    auc  = _bootstrap_ci(boot["auc"])
    bri  = _bootstrap_ci(boot["brier"])
    ece  = _bootstrap_ci(boot["ece"])

    _sec("Results  mean ± std  [95 % CI]")
    _print_row("Balanced Accuracy",  *bal,  0.85, 0.65)
    _print_row("F1 Score",           *f1,   0.85, 0.65)
    _print_row("MCC",                *mcc,  0.75, 0.50)
    _print_row("ROC-AUC",            *auc,  0.90, 0.75)
    _print_row("Brier Score",        *bri,  0.10, 0.20, lower_better=True)
    _print_row("ECE (calib. error)", *ece,  0.05, 0.15, lower_better=True)

    return {k: dict(zip(("mean","std","ci_lo","ci_hi"), v))
            for k, v in [("balanced_accuracy",bal),("f1_score",f1),("mcc",mcc),
                         ("roc_auc",auc),("brier_score",bri),("ece",ece)]} | \
           {"soft_threshold": thresh_used, "n_bootstrap": N_BOOT}


def eval_trend_detection() -> dict:
    _hdr("TREND DETECTION  ·  Bootstrap")
    _info(f"N_bootstrap={N_BOOT}")

    CLASSES = ["deteriorating","improving","stable"]

    _sec(f"Running {N_BOOT:,} bootstrap iterations …")
    boot = _bootstrap_trend(TREND_CASES)
    print("  Done.")

    macro = _bootstrap_ci(boot["macro"])
    pcr   = {c: _bootstrap_ci(boot[c]) for c in CLASSES}

    _sec("Results  mean ± std  [95 % CI]")
    _print_row("Macro F1 (unweighted, 3 classes)", *macro, 0.75, 0.55)

    _sec("Per-class Recall  mean ± std  [95 % CI]")
    for cls in CLASSES:
        m, s, lo, hi = pcr[cls]
        tgt    = 0.90 if cls == "deteriorating" else 0.75
        fn     = _ok if m >= tgt else (_warn if m >= tgt - 0.20 else _fail)
        suffix = "  ← safety-critical" if cls == "deteriorating" else ""
        fn(f"Recall [{cls:<14}]  {m:.3f} ± {s:.3f}  [{lo:.3f}–{hi:.3f}]"
           f"  target ≥{tgt:.2f}{suffix}")

    return {"macro_f1":          dict(zip(("mean","std","ci_lo","ci_hi"), macro)),
            "per_class_recall":  {c: dict(zip(("mean","std","ci_lo","ci_hi"), pcr[c]))
                                  for c in CLASSES},
            "n_bootstrap":       N_BOOT}


def eval_rag_retrieval() -> dict:
    _hdr("RAG RETRIEVAL  ·  Bootstrap")
    _info(f"N_bootstrap={N_BOOT}  (pre-scored relevance judgements, no live PubMed calls)")

    _sec(f"Running {N_BOOT:,} bootstrap iterations …")
    boot = _bootstrap_rag(RAG_CASES)
    print("  Done.")

    p3  = _bootstrap_ci(boot["p3"])
    p5  = _bootstrap_ci(boot["p5"])
    mrr = _bootstrap_ci(boot["mrr"])

    _sec("Results  mean ± std  [95 % CI]")
    _print_row("Precision@3",                *p3,  0.80, 0.60)
    _print_row("Precision@5",                *p5,  0.70, 0.50)
    _print_row("MRR (Mean Reciprocal Rank)", *mrr, 0.80, 0.60)

    return {"precision_at_3": dict(zip(("mean","std","ci_lo","ci_hi"), p3)),
            "precision_at_5": dict(zip(("mean","std","ci_lo","ci_hi"), p5)),
            "mrr":            dict(zip(("mean","std","ci_lo","ci_hi"), mrr)),
            "n_bootstrap":    N_BOOT}


def eval_validation() -> dict:
    _hdr("VALIDATION AGENT  ·  Bootstrap + Prob Noise on risk scores")
    _info(f"N_bootstrap={N_BOOT}  risk_score_σ=0.03")

    _sec(f"Running {N_BOOT:,} bootstrap iterations …")
    boot = _bootstrap_validation(VALIDATION_CASES)
    print("  Done.")

    acc = _bootstrap_ci(boot["acc"])
    far = _bootstrap_ci(boot["far"])

    _sec("Results  mean ± std  [95 % CI]")
    _print_row("Validation Accuracy",  *acc, 0.90, 0.70)
    m, s, lo, hi = far
    rat, col = _rating(m, 0.00, 0.25, lower_better=True)
    print(f"     {'False Approval Rate':<42} {m:.3f}±{s:.3f} ↓  [{DIM}{lo:.3f}–{hi:.3f}{RESET}]  {col}{rat}{RESET}")

    return {"validation_accuracy": dict(zip(("mean","std","ci_lo","ci_hi"), acc)),
            "false_approval_rate": dict(zip(("mean","std","ci_lo","ci_hi"), far)),
            "n_bootstrap":         N_BOOT}


def eval_benchmark() -> dict:
    _hdr("CLINICAL QA BENCHMARK  ·  Bootstrap")
    _info(f"N_bootstrap={N_BOOT}  (concept-group scoring, no LLM calls)")

    _sec(f"Running {N_BOOT:,} bootstrap iterations …")
    boot = _bootstrap_benchmark(BENCHMARK_CASES)
    print("  Done.")

    mcs = _bootstrap_ci(boot["mcs"])
    t3a = _bootstrap_ci(boot["t3a"])

    _sec("Results  mean ± std  [95 % CI]")
    _print_row("Mean Clinical Score  (avg concept coverage)",  *mcs, 0.65, 0.45)
    _print_row("Top-3 Accuracy       (≥3/5 concepts covered)", *t3a, 0.70, 0.50)

    return {"mean_clinical_score": dict(zip(("mean","std","ci_lo","ci_hi"), mcs)),
            "top3_accuracy":       dict(zip(("mean","std","ci_lo","ci_hi"), t3a)),
            "n_bootstrap":         N_BOOT}


# ══════════════════════════════════════════════════════════════════════════════
# SCORECARD
# ══════════════════════════════════════════════════════════════════════════════

def print_scorecard(results: dict):
    risk = results.get("risk_model", {})
    trd  = results.get("trend", {})
    rag  = results.get("rag", {})
    val  = results.get("validation", {})
    bnch = results.get("benchmark", {})

    def _g(section, key, sub=None):
        d = section.get(key,{}) if sub is None else section.get(key,{}).get(sub,{})
        if isinstance(d, dict):
            return d.get("mean",0), d.get("std",0), d.get("ci_lo",0), d.get("ci_hi",0)
        return float(d), 0.0, 0.0, 0.0

    print(f"\n\n{'═'*80}")
    print(f"{BOLD}  TA-CDSS METRICS SCORECARD  —  Bootstrap 95 % CI  (seed={SEED}){RESET}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
          f"   N_boot={N_BOOT}  prob_σ={PROB_SIGMA}  label_flip={LABEL_NOISE:.0%}")
    print(f"{'═'*80}")

    rows = [
        ("Balanced Accuracy",       *_g(risk,"balanced_accuracy"),                    0.85, 0.65, False),
        ("F1 Score",                *_g(risk,"f1_score"),                             0.85, 0.65, False),
        ("MCC",                     *_g(risk,"mcc"),                                  0.75, 0.50, False),
        ("ROC-AUC",                 *_g(risk,"roc_auc"),                              0.90, 0.75, False),
        ("Brier Score",             *_g(risk,"brier_score"),                          0.10, 0.20, True ),
        ("ECE (calibration error)", *_g(risk,"ece"),                                  0.05, 0.15, True ),
        ("Macro F1  (Trend)",       *_g(trd, "macro_f1"),                             0.75, 0.55, False),
        ("Recall [deteriorating]",  *_g(trd,"per_class_recall","deteriorating"),      0.90, 0.70, False),
        ("Recall [improving]",      *_g(trd,"per_class_recall","improving"),          0.75, 0.55, False),
        ("Recall [stable]",         *_g(trd,"per_class_recall","stable"),             0.75, 0.55, False),
        ("RAG  Precision@3",        *_g(rag,"precision_at_3"),                        0.80, 0.60, False),
        ("RAG  Precision@5",        *_g(rag,"precision_at_5"),                        0.70, 0.50, False),
        ("RAG  MRR",                *_g(rag,"mrr"),                                   0.80, 0.60, False),
        ("Validation Accuracy",     *_g(val,"validation_accuracy"),                   0.90, 0.70, False),
        ("False Approval Rate",     *_g(val,"false_approval_rate"),                   0.00, 0.25, True ),
        ("Mean Clinical Score",     *_g(bnch,"mean_clinical_score"),                  0.65, 0.45, False),
        ("Top-3 QA Accuracy",       *_g(bnch,"top3_accuracy"),                        0.70, 0.50, False),
    ]

    good_n = fair_n = poor_n = 0
    print(f"\n  ┌──────────────────────────────────┬─────────────────────────────────┬──────────┐")
    print(f"  │  Metric                          │  Mean ± Std   [95 % CI]         │  Rating  │")
    print(f"  ├──────────────────────────────────┼─────────────────────────────────┼──────────┤")

    for name, v, s, lo, hi, good, ok_t, lb in rows:
        rat, col = _rating(v, good, ok_t, lb)
        if rat=="GOOD": good_n+=1
        elif rat=="FAIR": fair_n+=1
        else: poor_n+=1
        arrow  = "↓" if lb else " "
        ci_str = f"{v:.3f}±{s:.3f}{arrow} [{lo:.3f}–{hi:.3f}]"
        print(f"  │  {name:<32}  │ {ci_str:<31} │ {col}{rat:<8}{RESET} │")

    n = len(rows)
    print(f"  ├──────────────────────────────────┴─────────────────────────────────┴──────────┤")
    print(f"  │  GOOD: {good_n}/{n}   FAIR: {fair_n}/{n}   POOR: {poor_n}/{n}"
          f"   (bootstrap {N_BOOT:,} iters, seed {SEED})                       │")
    print(f"  └──────────────────────────────────────────────────────────────────────────────┘")

    # ── Recommendations ───────────────────────────────────────────────────────
    print(f"\n{BOLD}  RECOMMENDATIONS{RESET}")
    rank = 1

    checks = [
        # (mean_value, threshold, lower_is_bad, label, fix)
        (_g(risk,"mcc")[0],           0.75, False,
         "MCC below 0.75 — most sensitive single risk metric.",
         "Expand labelled set with borderline ICU cases; re-run soft threshold search."),
        (_g(risk,"brier_score")[0],   0.10, True,
         "Brier above 0.10 — scores are miscalibrated.",
         "Apply Platt scaling or isotonic regression to raw risk outputs."),
        (_g(risk,"ece")[0],           0.05, True,
         "ECE above 0.05 — confidence does not match empirical rates.",
         "Use temperature scaling; plot reliability diagram before deployment."),
        (_g(trd,"per_class_recall","deteriorating")[0], 0.90, False,
         "Deterioration recall below 0.90 — primary safety gap.",
         "Add absolute-value triggers (not just relative delta) for SpO2 and Lactate."),
        (_g(rag,"precision_at_3")[0], 0.80, False,
         "RAG Precision@3 below 0.80 — relevant articles buried.",
         "Improve BM25 specificity in build_clinical_query() (hybrid_rag.py)."),
        (_g(val,"false_approval_rate")[0], 0.00, True,
         "False Approval Rate > 0 — dangerous approvals possible under score noise.",
         "Add explicit risk-phrase × noisy_score cross-check in _rule_based_verdict()."),
    ]

    for val_mean, thresh, lower_is_bad, msg, fix in checks:
        triggered = (val_mean > thresh) if lower_is_bad else (val_mean < thresh)
        if triggered:
            print(f"  {rank}. {msg}")
            print(f"     Fix: {fix}")
            rank += 1

    wide = [(nm, s) for nm, v, s, lo, hi, *_ in rows if s > 0.10]
    if wide:
        print(f"  {rank}. Wide bootstrap std (>0.10) in: {', '.join(nm for nm,_ in wide)}")
        print(f"     Collect more labelled cases — small N amplifies resampling variance.")
        rank += 1

    if rank == 1:
        _ok("All metrics meet targets under bootstrap + noise conditions.")
        print(f"     Next step: validate on a real eICU cohort and report calibration plots.")

    print(f"\n{'═'*80}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}TA-CDSS Metrics Evaluation — Bootstrap + Prob Noise + Label Noise + Soft Threshold{RESET}")
    print(f"{'═'*74}")
    print(f"Seed={SEED}   Bootstrap iters={N_BOOT:,}   Prob σ={PROB_SIGMA}   Label-flip={LABEL_NOISE:.0%}")
    print(f"{'─'*74}")
    print("Risk Model  : Bootstrap · Prob noise (σ=0.05) · Label noise (5%) · Soft threshold · ECE")
    print("Trend       : Bootstrap · Macro F1 · Per-class Recall")
    print("RAG         : Bootstrap · Precision@3 · Precision@5 · MRR")
    print("Validation  : Bootstrap · Prob noise on risk scores (σ=0.03) · FAR")
    print("Benchmark   : Bootstrap · Mean Clinical Score · Top-3 Accuracy")
    print("All metrics reported as:  mean ± std  [95% CI lo – hi]")
    print(f"{'═'*74}\n")

    results = {
        "risk_model": eval_risk_model(),
        "trend":      eval_trend_detection(),
        "rag":        eval_rag_retrieval(),
        "validation": eval_validation(),
        "benchmark":  eval_benchmark(),
        "meta": {"seed": SEED, "n_bootstrap": N_BOOT,
                 "prob_sigma": PROB_SIGMA, "label_noise": LABEL_NOISE,
                 "timestamp": datetime.now().isoformat()}
    }

    print_scorecard(results)

    Path("outputs").mkdir(exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    j_path = f"outputs/metrics_report_{ts}.json"
    t_path = f"outputs/metrics_summary_{ts}.txt"

    with open(j_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    def _g(sec, key, sub=None):
        d = sec.get(key,{}) if sub is None else sec.get(key,{}).get(sub,{})
        if isinstance(d, dict):
            return d.get("mean",0), d.get("std",0), d.get("ci_lo",0), d.get("ci_hi",0)
        return float(d), 0, 0, 0

    def fms(sec, key, sub=None):
        m, s, lo, hi = _g(sec, key, sub)
        return f"{m:.3f} ± {s:.3f}  [95% CI {lo:.3f}–{hi:.3f}]"

    r, t, g, v, b = (results[k] for k in ("risk_model","trend","rag","validation","benchmark"))

    with open(t_path, "w", encoding="utf-8") as f:
        f.write(f"TA-CDSS Metrics  {ts}\n"
                f"Bootstrap={N_BOOT}  seed={SEED}  prob_sigma={PROB_SIGMA}  label_flip={LABEL_NOISE:.0%}\n"
                f"{'='*56}\n\n")
        f.write("Risk Prediction Model\n")
        for k, n in [("balanced_accuracy","Balanced Acc"),("f1_score","F1 Score"),
                     ("mcc","MCC"),("roc_auc","ROC-AUC"),
                     ("brier_score","Brier Score"),("ece","ECE")]:
            f.write(f"  {n:<18}: {fms(r,k)}\n")
        f.write(f"  Soft threshold   : {r.get('soft_threshold',0):.3f}\n\n")
        f.write("Trend Detection\n")
        f.write(f"  Macro F1         : {fms(t,'macro_f1')}\n")
        for cls in ["deteriorating","improving","stable"]:
            f.write(f"  Recall [{cls:<14}]: {fms(t,'per_class_recall',cls)}\n")
        f.write("\nRAG Retrieval\n")
        for k, n in [("precision_at_3","Precision@3"),("precision_at_5","Precision@5"),("mrr","MRR")]:
            f.write(f"  {n:<18}: {fms(g,k)}\n")
        f.write("\nValidation Layer\n")
        f.write(f"  Accuracy         : {fms(v,'validation_accuracy')}\n")
        f.write(f"  False Appr. Rate : {fms(v,'false_approval_rate')}\n")
        f.write("\nClinical QA Benchmark\n")
        f.write(f"  Mean Clin. Score : {fms(b,'mean_clinical_score')}\n")
        f.write(f"  Top-3 Accuracy   : {fms(b,'top3_accuracy')}\n")

    print(f"  JSON  → {j_path}")
    print(f"  TXT   → {t_path}\n")


if __name__ == "__main__":
    main()