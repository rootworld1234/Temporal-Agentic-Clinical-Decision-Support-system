"""
diagnostics.py -- TA-CDSS Diagnostic Suite (Research-Grade)

Updated to work with improved module signatures:
  - gru_temporal: run_gru_module now returns risk_score and temporal_phases
  - agentic_validation: run_safety_rule_check is new
  - hybrid_rag: accepts patient_info and temporal_phases kwargs
"""

import json
import re
import sys
import time
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.gru_temporal       import EICUDataLoader, TimeAwareGRU, run_gru_module
from modules.hybrid_rag         import run_hybrid_rag
from modules.llm_client         import call_llm, call_llm_local, build_messages, get_active_backend
from modules.agentic_validation import run_validation_agent, run_safety_rule_check
import torch

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):     print(f"  {GREEN}PASS{RESET}  {msg}")
def warn(msg):   print(f"  {YELLOW}WARN{RESET}  {msg}")
def fail(msg):   print(f"  {RED}FAIL{RESET}  {msg}")
def info(msg):   print(f"         {msg}")
def header(msg): print(f"\n{BOLD}{'═'*60}{RESET}\n{BOLD}  {msg}{RESET}\n{'═'*60}")
def section(msg):print(f"\n{BOLD}── {msg} ──{RESET}")


# ── Diagnostic 1: eICU Data ───────────────────────────────────────────────────

def diag_eicu_data(data_dir="data/eicu") -> dict:
    header("DIAGNOSTIC 1 — eICU Data Quality")
    loader = EICUDataLoader(data_dir=data_dir)
    loader.load()
    results = {}

    section("File Coverage")
    for attr, fname in [
        ("patients","patient.csv"),("vitals","vitalPeriodic.csv"),
        ("labs","lab.csv"),("treatments","treatment.csv"),
        ("diagnoses","diagnosis.csv"),("resp_charting","respiratoryCharting.csv")
    ]:
        df = getattr(loader, attr, None)
        if df is not None:
            ok(f"{fname}: {len(df):,} rows"); results[fname] = len(df)
        else:
            warn(f"{fname}: not found (optional for extended features)"); results[fname] = 0

    section("Timestep Coverage (sample 20 patients)")
    if loader.patients is not None:
        from config import ICU_INPUT_SIZE
        pid_col    = "patientunitstayid" if "patientunitstayid" in loader.patients.columns else loader.patients.columns[0]
        sample_ids = loader.patients[pid_col].dropna().unique()[:20]
        counts = []
        for pid in sample_ids:
            _, _, seq = loader.build_tensor(int(pid), input_size=ICU_INPUT_SIZE)
            counts.append(len(seq["features"]))
        avg = np.mean(counts)
        info(f"Timesteps per patient — avg: {avg:.1f}, min: {min(counts)}, max: {max(counts)}")
        if avg >= 20:   ok(f"Average timesteps ({avg:.1f}) sufficient for GRU")
        elif avg >= 5:  warn(f"Average timesteps ({avg:.1f}) low — add vitalPeriodic.csv")
        else:           fail(f"Average timesteps ({avg:.1f}) too low")
        results["avg_timesteps"] = avg

    return results


# ── Diagnostic 2: GRU ────────────────────────────────────────────────────────

def diag_gru(data_dir="data/eicu") -> dict:
    header("DIAGNOSTIC 2 — GRU Temporal Reasoning (11 features)")
    results = {}
    loader  = EICUDataLoader(data_dir=data_dir)
    loader.load()

    if loader.patients is None:
        fail("No patient data — skipping GRU diagnostics."); return results

    from config import ICU_INPUT_SIZE
    pid_col    = "patientunitstayid" if "patientunitstayid" in loader.patients.columns else loader.patients.columns[0]
    sample_ids = loader.patients[pid_col].dropna().unique()[:10]

    section("TCSV Diversity Check")
    tcsv_list = []
    for pid in sample_ids:
        try:
            r = run_gru_module(int(pid), loader)
            tcsv_list.append(r["tcsv"])
            if r.get("temporal_phases"):
                info(f"Patient {pid}: phases={list(r['temporal_phases'].keys())}, "
                     f"risk={r['risk_score']:.3f}")
        except Exception as e:
            warn(f"Patient {pid} failed: {e}")

    if len(tcsv_list) >= 2:
        sims = []
        for i in range(len(tcsv_list)):
            for j in range(i+1, len(tcsv_list)):
                a, b = tcsv_list[i], tcsv_list[j]
                sim  = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                sims.append(sim)
        avg_sim = np.mean(sims)
        info(f"Avg TCSV cosine similarity: {avg_sim:.4f}")
        if avg_sim < 0.95:   ok(f"TCSVs diverse (sim={avg_sim:.3f})")
        elif avg_sim < 0.99: warn(f"TCSVs somewhat similar (sim={avg_sim:.3f})")
        else:                fail(f"TCSVs nearly identical (sim={avg_sim:.3f}) — GRU collapsing")
        results["avg_tcsv_similarity"] = float(avg_sim)

    section("Gradient Sanity Check")
    model = TimeAwareGRU(input_size=ICU_INPUT_SIZE, hidden_size=64)
    x, dt = torch.randn(4, 10, ICU_INPUT_SIZE), torch.rand(4, 10, 1)
    tcsv, _ = model(x, dt)
    tcsv.mean().backward()
    ok("Gradients flow correctly") if all(p.grad is not None for p in model.parameters()) else fail("Gradient flow broken")
    results["gradients_ok"] = True

    section("Safety Rule Check")
    dummy_phases = {"Recent": {"SpO2": 88, "Respiration Rate": 32, "Heart Rate": 125, "Lactate": 5.0}}
    sc = run_safety_rule_check(dummy_phases)
    if sc["n_triggered"] >= 3:
        ok(f"Safety rule check triggered {sc['n_triggered']} rules on critical dummy data")
    else:
        warn(f"Safety rule check triggered only {sc['n_triggered']} rules (expected ≥3)")
    results["safety_rules_ok"] = sc["n_triggered"] >= 3

    return results


# ── Diagnostic 3: RAG ────────────────────────────────────────────────────────

def diag_rag() -> dict:
    header("DIAGNOSTIC 3 — Hybrid RAG Retrieval Quality")
    results = {}
    tests = [
        {
            "patient_info":    {"ventilated": True, "diagnosis_str": "pneumonia respiratory failure"},
            "temporal_phases": {"Recent": {"SpO2": 89, "Respiration Rate": 28, "PaO2_FiO2": 160}},
            "keywords":        ["ventilat", "respiratory", "ards", "oxygen", "extubat"],
        },
        {
            "patient_info":    {"ventilated": False, "diagnosis_str": "sepsis"},
            "temporal_phases": {"Recent": {"Lactate": 3.5, "Systolic BP": 82}},
            "keywords":        ["sepsis", "vasopressor", "antibiotic", "lactate", "shock"],
        },
    ]
    scores = []
    for test in tests:
        rag = run_hybrid_rag(
            clinical_query  = "",
            patient_summary = "ICU patient",
            patient_info    = test["patient_info"],
            temporal_phases = test["temporal_phases"],
            top_k           = 5,
        )
        conf    = rag["confidence"]
        ev_text = rag["top_evidence"].lower()
        found   = [k for k in test["keywords"] if k in ev_text]
        cov     = len(found) / len(test["keywords"])
        scores.append(conf)
        info(f"Query: {rag['query'][:60]}")
        info(f"Confidence: {conf:.3f} | Keyword coverage: {len(found)}/{len(test['keywords'])} ({cov*100:.0f}%)")
        (ok if conf >= 0.3 else warn)(f"Confidence {conf:.3f}")
        (ok if cov  >= 0.6 else warn)(f"Coverage {cov*100:.0f}%")

        # Check clinical_insight populated
        has_insights = any(e.get("clinical_insight") for e in rag["evidence"])
        (ok if has_insights else warn)("Evidence articles have clinical_insight fields")

    results["avg_confidence"] = float(np.mean(scores)) if scores else 0.0
    return results


# ── Diagnostic 4: LLM ────────────────────────────────────────────────────────

def diag_llm() -> dict:
    header("DIAGNOSTIC 4 — LLM Backend")
    results = {}

    section("Backend Connectivity")
    resp = call_llm_local(build_messages("Reply with exactly: SYSTEM_OK", ""), max_tokens=20)
    if "" in resp:
        fail(f"No LLM backend reachable: {resp}")
        results["connected"] = False
        return results
    ok(f"Backend responding via {get_active_backend()}: '{resp[:50]}'")
    results["connected"] = True

    time.sleep(1)

    section("Response Length Check")
    resp = call_llm_local(
        build_messages(
            "You are an ICU physician. Be thorough.",
            "65yo ICU patient: fever, hypotension, lactate 4.2. Give: (1) diagnosis, (2) key findings, (3) management, (4) monitoring.",
        ), max_tokens=800)
    length = len(resp)
    info(f"Response length: {length} chars | backend: {get_active_backend()}")
    (ok if length >= 400 else warn if length >= 150 else fail)(f"Response length {length} chars")
    results["response_length"] = length

    time.sleep(1)

    section("Hallucination Risk Check")
    hall_resp = call_llm_local(
        build_messages(
            "You are a clinical pharmacist.",
            "A colleague asks about 'Nexaflozin-7' for septic shock. Is this real? If not, say so clearly.",
        ), max_tokens=200).lower()

    phrases = ["not familiar","not aware","do not recognize","unfamiliar","not a real",
               "does not exist","no such","not real","fictional","unrecognized","unknown medication"]
    expressed = any(p in hall_resp for p in phrases)
    (ok if expressed else warn)(
        "Correctly flagged fake drug" if expressed
        else f"May not have flagged fake drug: '{hall_resp[:100]}'"
    )
    results["hallucination_check"] = expressed

    return results


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(all_results: dict):
    header("DIAGNOSTIC SUMMARY")
    issues, warnings = [], []

    ts = all_results.get("data",{}).get("avg_timesteps", 0)
    if ts < 5:   issues.append("GRU: Very few timesteps — add vitalPeriodic.csv")
    elif ts < 20: warnings.append(f"GRU: Low timesteps ({ts:.1f} avg)")

    sim = all_results.get("gru",{}).get("avg_tcsv_similarity", 0)
    if sim > 0.99: issues.append("GRU: TCSVs too similar — model not differentiating")

    if not all_results.get("gru",{}).get("safety_rules_ok", True):
        warnings.append("Safety rules: fewer rules triggered than expected on critical data")

    conf = all_results.get("rag",{}).get("avg_confidence", 0)
    if conf < 0.2: warnings.append(f"RAG: Low retrieval confidence ({conf:.3f})")

    if not all_results.get("llm",{}).get("connected", False):
        issues.append("LLM: No backend reachable — set GEMINI_API_KEY or start LM Studio")

    rl = all_results.get("llm",{}).get("response_length", 0)
    if rl < 100:  issues.append(f"LLM: Responses very short ({rl} chars)")
    elif rl < 400: warnings.append(f"LLM: Responses short ({rl} chars)")

    if not all_results.get("llm",{}).get("hallucination_check", True):
        warnings.append("LLM: Possible hallucination on fake drug test")

    if not issues and not warnings:
        print(f"\n  {GREEN}{BOLD}All diagnostics passed!{RESET}")
    else:
        if issues:
            print(f"\n  {RED}{BOLD}Critical Issues ({len(issues)}):{RESET}")
            for i in issues: print(f"     {i}")
        if warnings:
            print(f"\n  {YELLOW}{BOLD}Warnings ({len(warnings)}):{RESET}")
            for w in warnings: print(f"      {w}")

    print(f"\n{'═'*60}")
    print("  Quick fixes:")
    print("  1. Gemini key missing  → set GEMINI_API_KEY in config.py")
    print("  2. LM Studio fallback  → start server, load model, port 1234")
    print("  3. Short responses     → LM Studio: Context=4096, MaxTokens=1024")
    print("  4. Low timesteps       → add vitalPeriodic.csv to data/eicu/")
    print("  5. Extended features   → add respiratoryCharting.csv for FiO2/PEEP")
    print(f"{'═'*60}\n")


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}TA-CDSS Diagnostic Suite (Research-Grade){RESET}\n")
    print("Which diagnostics to run?")
    print("  [1] Data quality   [2] GRU   [3] RAG   [4] LLM   [5] All")
    choice = input("\nEnter choice (1-5, default=5): ").strip() or "5"

    all_results = {}
    if choice in ("1","5"): all_results["data"] = diag_eicu_data()
    if choice in ("2","5"): all_results["gru"]  = diag_gru()
    if choice in ("3","5"): all_results["rag"]  = diag_rag()
    if choice in ("4","5"): all_results["llm"]  = diag_llm()

    print_summary(all_results)

    Path("outputs").mkdir(exist_ok=True)
    from datetime import datetime
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"outputs/diagnostics_{ts}.json"
    with open(fname, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"   Saved: {fname}\n")


if __name__ == "__main__":
    main()
