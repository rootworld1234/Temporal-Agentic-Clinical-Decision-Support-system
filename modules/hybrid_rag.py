"""
Component B — Context-Aware Hybrid RAG (Research-Grade)

Key improvements over prototype:
  1. Clinically-filtered queries: builds richer PubMed queries from
     ventilator status, diagnosis string, and critical findings
     (was generic "ICU management treatment outcomes")
  2. Evidence summarisation: each returned article includes a
     clinical_insight field — a 1-sentence takeaway extracted by
     the LLM framed around the patient's specific findings
     (prototype only returned raw titles + truncated abstracts)
  3. Keyword-guided relevance filter: hard-filters articles that
     don't contain at least one clinical keyword from the query
  4. BM25 + semantic re-ranking logic unchanged (solid foundation)

LLM Routing (Stable Architecture):
  RAG Insight Extraction (extract_clinical_insight_llm) -> call_llm_local (LM Studio)
  BM25, scoring, filtering -> fully local (no LLM)
"""

import requests
import time
import math
import re
from datetime import datetime
from typing import Optional
from collections import Counter

# Quality control — imported lazily to avoid circular deps at module load time
def _qc():
    try:
        from modules.quality_control import (
            RAGCache, RAGQualityGate, patient_fingerprint,
            _extract_patient_concepts,
        )
        return RAGCache, RAGQualityGate, patient_fingerprint, _extract_patient_concepts
    except ImportError:
        return None, None, None, None


# ── BM25 ───────────────────────────────────────────────────────────────────────

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus    = []
        self.doc_freqs = []
        self.idf       = {}
        self.avgdl     = 0

    def fit(self, corpus: list):
        self.corpus    = corpus
        self.doc_freqs = []
        df             = Counter()
        total_len      = 0
        for doc in corpus:
            tokens = self._tokenize(doc)
            total_len += len(tokens)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            for word in freq:
                df[word] += 1
        self.avgdl = total_len / max(len(corpus), 1)
        N = len(corpus)
        for word, freq in df.items():
            self.idf[word] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query: str, top_k: int = 5) -> list:
        tokens = self._tokenize(query)
        scores = []
        for i, freq in enumerate(self.doc_freqs):
            dl    = sum(freq.values())
            score = 0.0
            for token in tokens:
                if token in freq:
                    tf    = freq[token]
                    idf   = self.idf.get(token, 0)
                    score += idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                    )
            scores.append((i, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def _tokenize(self, text: str) -> list:
        return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())


# ── Cosine similarity ──────────────────────────────────────────────────────────

def cosine_similarity_text(query: str, doc: str) -> float:
    def tf_vector(text):
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        counts = Counter(tokens)
        total  = max(sum(counts.values()), 1)
        return {k: v / total for k, v in counts.items()}

    q_vec  = tf_vector(query)
    d_vec  = tf_vector(doc)
    vocab  = set(q_vec) | set(d_vec)
    dot    = sum(q_vec.get(w, 0) * d_vec.get(w, 0) for w in vocab)
    norm_q = math.sqrt(sum(v**2 for v in q_vec.values()))
    norm_d = math.sqrt(sum(v**2 for v in d_vec.values()))
    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_q * norm_d)


# ── PubMed API ─────────────────────────────────────────────────────────────────

PUBMED_SEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# NCBI requires tool name + email for hosted/cloud environments.
# Without these, cloud IPs get throttled or blocked (3 req/sec limit drops to 0).
# Register free at: https://www.ncbi.nlm.nih.gov/account/
def _ncbi_email() -> str:
    """Read email from Streamlit secrets if available, else fall back to config."""
    try:
        import streamlit as st
        return st.secrets.get("NCBI_EMAIL", "ta.cdss.research@gmail.com")
    except Exception:
        return "ta.cdss.research@gmail.com"   # replace with your real email

NCBI_TOOL  = "TA-CDSS"

HEADERS = {
    "User-Agent": f"TA-CDSS/2.0 (tool={NCBI_TOOL}; research use; contact={_ncbi_email()})",
    "Accept":     "application/json",
}


def pubmed_search(query: str, max_results: int = 20, years_back: int = 5) -> list:
    current_year = datetime.now().year
    min_year     = current_year - years_back
    params = {
        "db":      "pubmed",
        "term":    f"{query} AND ({min_year}:{current_year}[pdat])",
        "retmax":  max_results,
        "retmode": "json",
        "sort":    "pub+date",
        "email":   _ncbi_email(),
        "tool":    NCBI_TOOL,
    }
    try:
        resp = requests.get(PUBMED_SEARCH_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("esearchresult", {}).get("idlist", [])
        if len(results) < 8:
            params["term"] = f"{query} AND ({current_year - 8}:{current_year}[pdat])"
            resp    = requests.get(PUBMED_SEARCH_URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("esearchresult", {}).get("idlist", [])
        return results
    except requests.exceptions.ConnectionError as e:
        if "10061" in str(e) or "refused" in str(e).lower():
            print(f"    PubMed blocked — using fallback corpus.")
        else:
            print(f"    PubMed connection error: {e}")
        return []
    except Exception as e:
        print(f"    PubMed search error: {e}")
        return []


def pubmed_fetch_abstracts(pmids: list) -> list:
    if not pmids:
        return []
    articles = []
    try:
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json",
                  "email": _ncbi_email(), "tool": NCBI_TOOL}
        resp   = requests.get(PUBMED_SUMMARY_URL, params=params, timeout=15)
        resp.raise_for_status()
        summaries = resp.json().get("result", {})
    except Exception as e:
        print(f"    PubMed summary error: {e}")
        summaries = {}

    time.sleep(0.4)

    try:
        params   = {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "text",
                    "email": _ncbi_email(), "tool": NCBI_TOOL}
        resp     = requests.get(PUBMED_FETCH_URL, params=params, timeout=20)
        resp.raise_for_status()
        raw_text = resp.text
    except Exception as e:
        print(f"    PubMed fetch error: {e}")
        raw_text = ""

    abstract_blocks = raw_text.split("\n\n\n")

    for i, pmid in enumerate(pmids):
        summary  = summaries.get(pmid, {})
        title    = summary.get("title", "Unknown Title")
        pub_date = summary.get("pubdate", "2000")
        year_match = re.search(r"(\d{4})", str(pub_date))
        year     = int(year_match.group(1)) if year_match else 2000
        abstract = abstract_blocks[i].strip() if i < len(abstract_blocks) else ""
        articles.append({
            "pmid":     pmid,
            "title":    title,
            "abstract": abstract,
            "year":     year,
            "pub_date": pub_date,
        })
    return articles


def recency_weight(year: int) -> float:
    current_year = datetime.now().year
    age = current_year - year
    if age <= 1:  return 1.0
    elif age <= 2: return 0.90
    elif age <= 4: return 0.70
    elif age <= 6: return 0.45
    elif age <= 9: return 0.20
    else:          return 0.05


# ── Clinical Query Builder (NEW) ───────────────────────────────────────────────

def build_clinical_query(
    patient_info: dict,
    temporal_phases: dict,
    base_query: str = "",
) -> str:
    """
    FIX 3: Build a PubMed query that prioritises observable clinical findings
    (rising RR, falling SpO2, high lactate) over raw diagnosis strings.

    Diagnosis strings from eICU like "cardiovascular|ventricular disorders|
    congestive heart failure" produce irrelevant results when the patient is
    actually showing respiratory deterioration. The fix is:
      1. Symptom signals come first (what the data shows NOW)
      2. Diagnosis string is appended only if short and meaningful
      3. Hard cap at 10 words to keep queries focused
    """
    import re
    recent = temporal_phases.get("Recent", {})
    terms  = []

    # Priority 1: Respiratory signals (most common ICU deterioration pathway)
    spo2_val = recent.get("SpO2", 100)
    rr_val   = recent.get("Respiration Rate", 0)
    pf_val   = recent.get("PaO2_FiO2", 500)

    if rr_val > 25 or spo2_val < 94 or pf_val < 250:
        if patient_info.get("ventilated"):
            terms.append("mechanical ventilation respiratory deterioration ICU monitoring")
        else:
            terms.append("respiratory failure hypoxemia ICU management")

        if pf_val < 200:
            terms.append("ARDS oxygenation low tidal volume")
        elif spo2_val < 92:
            terms.append("hypoxemia oxygen therapy ICU")

    # Priority 2: Haemodynamic / sepsis signals
    lactate_val = recent.get("Lactate", 0)
    sbp_val     = recent.get("Systolic BP", 120)
    if lactate_val > 2.0 or sbp_val < 90:
        terms.append("sepsis vasopressors lactate haemodynamic ICU")

    # Priority 3: Ventilator weaning — only if actively ventilated and stable oxygenation
    if patient_info.get("ventilated") and spo2_val >= 94 and not terms:
        terms.append("mechanical ventilation weaning extubation readiness")

    # Priority 4: Diagnosis string.
    # FIX 1: Always extract and include the diagnosis, even when symptom signals
    # are present — for specific diagnoses like "angioedema airway obstruction"
    # the diagnosis IS the best search term and should lead the query.
    diag = str(patient_info.get("diagnosis_str", patient_info.get("apache_score", "")))
    diag_term = ""
    if diag and diag.lower() not in ("unknown", "nan", ""):
        # Strip eICU pipe-delimited hierarchy — keep the most specific segment
        segments   = re.split(r"[|/\\]", diag)
        best_seg   = segments[-1].strip() if segments else diag
        clean_diag = re.sub(r"[^a-zA-Z0-9 ]", " ", best_seg).strip()
        diag_term  = " ".join(clean_diag.split()[:5])

    if diag_term:
        # If diagnosis is specific (not generic like "respiratory failure"),
        # put it FIRST so PubMed retrieves condition-specific literature.
        GENERIC_DIAG_TERMS = {"respiratory failure", "cardiac arrest", "sepsis",
                              "icu", "critical care", "mechanical ventilation"}
        is_specific = not any(g in diag_term.lower() for g in GENERIC_DIAG_TERMS)
        if is_specific:
            terms.insert(0, diag_term)   # diagnosis leads the query
        elif not terms:
            terms.append(f"{diag_term} ICU management")

    # Absolute fallback
    if not terms:
        unit = patient_info.get("unit_type", "ICU")
        terms.append(f"critical care {unit} monitoring outcomes")

    # Join — diagnosis first, then symptom signals — cap at 12 words
    full  = " ".join(terms)
    words = full.split()
    return " ".join(words[:12])


# ── Clinical Keyword Filter (NEW) ─────────────────────────────────────────────

CLINICAL_KEYWORD_GROUPS = {
    "respiratory": ["ventilat", "respiratory", "ards", "extubat", "oxygen", "spo2", "fio2", "peep", "hypoxia"],
    "cardiac":     ["cardiac", "heart", "arrhythmia", "atrial", "ventricular", "tachycardia"],
    "sepsis":      ["sepsis", "septic", "antibiotic", "infection", "bacteremia", "vasopressor"],
    "shock":       ["shock", "hypotension", "vasopressor", "norepinephrine", "fluid resuscitat"],
    "renal":       ["renal", "kidney", "creatinine", "dialysis", "rrt", "aki"],
    "metabolic":   ["lactate", "acidosis", "glucose", "electrolyte"],
    "icu_general": ["icu", "critical care", "intensive care", "mortality", "outcome"],
}


def _get_query_keywords(query: str) -> set:
    """Extract which clinical keyword groups are relevant to the query."""
    query_lower = query.lower()
    relevant    = set()
    for group, keywords in CLINICAL_KEYWORD_GROUPS.items():
        if any(kw in query_lower for kw in keywords):
            relevant.add(group)
    if not relevant:
        relevant.add("icu_general")
    return relevant


# FIX 3: Exclude paediatric/neonatal papers — they are irrelevant for adult ICU patients
# and were appearing in results for elderly patients (e.g. 84-year-old).
EXCLUDE_KEYWORDS = [
    "neonatal", "neonate", "newborn", "preterm", "premature infant",
    "pediatric", "paediatric", "child", "children", "infant", "toddler",
    "adolescent", "congenital", "birth weight",
]


def _is_paediatric_paper(article: dict) -> bool:
    """Return True if article is clearly about neonatal/paediatric population."""
    text = (article.get("title", "") + " " + article.get("abstract", "")).lower()
    return any(kw in text for kw in EXCLUDE_KEYWORDS)


def _article_passes_filter(article: dict, relevant_groups: set) -> bool:
    """Return True if article is relevant and not paediatric/neonatal."""
    # FIX 3: Hard exclude paediatric papers first
    if _is_paediatric_paper(article):
        return False
    text = (article["title"] + " " + article["abstract"]).lower()
    relevant_keywords = []
    for group in relevant_groups:
        relevant_keywords.extend(CLINICAL_KEYWORD_GROUPS.get(group, []))
    return any(kw in text for kw in relevant_keywords)


# ── Evidence Insight Extractor (NEW) ──────────────────────────────────────────

def extract_clinical_insight(article: dict, patient_context: str) -> str:
    """
    Extract a 1-sentence clinical insight from an abstract by finding
    the sentence most similar to the patient context.

    Prototype only showed raw abstract truncated to 400 chars.
    Research-grade surfaces the most relevant sentence.
    """
    abstract = article.get("abstract", "")
    if not abstract:
        return article.get("title", "")

    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    if not sentences:
        return abstract[:150]

    # Find sentence most similar to patient context
    best_sim  = 0.0
    best_sent = sentences[-1]  # default: last sentence (usually conclusion)

    for sent in sentences:
        sim = cosine_similarity_text(patient_context, sent)
        if sim > best_sim:
            best_sim  = sim
            best_sent = sent

    # Clean up and truncate
    insight = re.sub(r"\s+", " ", best_sent).strip()
    if len(insight) > 200:
        insight = insight[:197] + "..."
    return insight



# ── 3: Demographic Filters ────────────────────────────────────────────────────
# Dynamically built from patient_info so filters match the actual patient.
# Adult ICU patients should never receive neonatal/paediatric literature.

DEMOGRAPHIC_INCLUDE = [
    "adult", "intensive care", "critically ill", "clinical study",
    "clinical trial", "randomized", "cohort", "observational",
]

DEMOGRAPHIC_EXCLUDE = [
    "neonatal", "neonate", "newborn", "preterm", "premature infant",
    "pediatric", "paediatric", "child", "children", "infant", "toddler",
    "adolescent", "congenital", "birth weight",
]


def build_demographic_filters(patient_info: dict) -> dict:
    """
    Build include/exclude keyword lists from patient profile.
    Age > 18 -> enforce adult filters.
    Age unknown -> apply adult filters by default (ICU population).
    """
    include = list(DEMOGRAPHIC_INCLUDE)
    exclude = list(DEMOGRAPHIC_EXCLUDE)

    age_raw = patient_info.get("age", "unknown") if patient_info else "unknown"
    try:
        age = int(str(age_raw).replace(">", "").replace("<", "").strip())
    except (ValueError, TypeError):
        age = None

    if age is not None:
        if age >= 65:
            include += ["elderly", "older adult", "geriatric"]
        if age < 18:
            # Genuine paediatric patient — swap filters
            exclude = [k for k in exclude if k not in ("pediatric", "paediatric", "child", "children")]
            include += ["pediatric", "paediatric"]

    return {"include": include, "exclude": exclude}


def _passes_demographic_filter(article: dict, filters: dict) -> bool:
    """
    Returns True if the article passes demographic relevance check.
    Exclude list is hard — any match fails the article.
    Include list is soft — article passes if it matches ANY include term
    OR if it contains no demographic language at all (neutral paper).
    """
    text = (article.get("title", "") + " " + article.get("abstract", "")).lower()

    # Hard exclude
    for kw in filters.get("exclude", []):
        if kw in text:
            return False

    # Soft include — neutral papers (no demographic language) are allowed through
    include_terms = filters.get("include", [])
    has_demo_language = any(kw in text for kw in
        ["adult", "child", "neonatal", "pediatric", "elderly", "geriatric",
         "infant", "newborn", "adolescent", "patient"])
    if has_demo_language:
        return any(kw in text for kw in include_terms)

    return True  # neutral paper — keep it


# ── 5: Evidence Quality Scoring ───────────────────────────────────────────────
# Replaces flat recency * hybrid with a weighted clinical importance score:
#   0.5 * semantic_similarity
#   0.3 * recency_score
#   0.2 * study_quality_bonus
#
# Study quality bonuses (additive, capped at 0.40):
#   systematic review / meta-analysis  -> +0.20
#   randomized controlled trial        -> +0.15
#   clinical trial / prospective       -> +0.10
#   guideline / consensus statement    -> +0.10
#   recent article (<= 3 years)        -> +0.10  (stacks with recency_score)

STUDY_TYPE_PATTERNS = [
    (r"systematic review|meta.analysis",            0.20),
    (r"randomized.controlled|randomised.controlled|rct", 0.15),
    (r"clinical trial|prospective.study|prospective.cohort", 0.10),
    (r"guideline|consensus statement|practice recommendation", 0.10),
]


def study_quality_bonus(article: dict) -> float:
    """Return a quality bonus (0.0 – 0.40) based on study design."""
    text  = (article.get("title", "") + " " + article.get("abstract", "")).lower()
    bonus = 0.0
    for pattern, weight in STUDY_TYPE_PATTERNS:
        if re.search(pattern, text):
            bonus += weight
    # Extra recency bonus for very recent articles
    current_year = datetime.now().year
    if (current_year - article.get("year", 0)) <= 3:
        bonus += 0.10
    return min(bonus, 0.40)


def evidence_score(semantic_sim: float, norm_bm25: float,
                   article: dict) -> tuple:
    """
    Compute final evidence score with clinical importance weighting.

    Returns (final_score, score_breakdown_dict).
    """
    recency  = recency_weight(article["year"])
    quality  = study_quality_bonus(article)
    hybrid   = 0.5 * norm_bm25 + 0.5 * semantic_sim

    # Weighted combination
    score = (
        0.50 * hybrid    +   # semantic + BM25 relevance
        0.30 * recency   +   # publication recency
        0.20 * quality       # study design quality
    )

    breakdown = {
        "semantic":  round(hybrid,   3),
        "recency":   round(recency,  3),
        "quality":   round(quality,  3),
        "final":     round(score,    3),
    }
    return score, breakdown


# ── 6: LLM-Powered Clinical Insight Extraction ────────────────────────────────
# Upgrades the existing cosine-similarity sentence picker to an LLM call
# that returns a single structured clinical takeaway sentence.
# Falls back to the cosine method if the LLM is unavailable.

def extract_clinical_insight_llm(article: dict, patient_context: str) -> str:
    """
    Use the LLM to extract a one-sentence clinical takeaway from the abstract,
    framed around the patient context.

    Example output:
      "Patients with acute respiratory failure on mechanical ventilation showed
       significantly lower mortality with lung-protective ventilation strategies."
    """
    abstract = article.get("abstract", "").strip()
    if not abstract:
        return article.get("title", "No abstract available.")

    try:
        from modules.llm_client import call_llm_local, build_messages
        messages = build_messages(
            "You are a clinical evidence analyst for an ICU decision support system. "
            "Your job is to extract a single complete clinical takeaway sentence from a "
            "medical abstract, framed around the patient context. "
            "Rules: output ONLY the takeaway sentence. No labels. No quotes. "
            "No preamble like \'This abstract describes\'. "
            "Start directly with the clinical finding.",
            f"Patient context:\n{patient_context[:250]}\n\n"
            f"Abstract:\n{abstract[:800]}\n\n"
            f"Write one complete sentence stating the most clinically relevant finding "
            f"from this abstract for the patient above. Begin with the subject of the finding "
            f"(e.g. \'Patients with...\', \'Angioedema airway obstruction...\', \'Early intervention...\'):"
        )
        result = call_llm_local(messages, temperature=0.1, max_tokens=250)
        result = result.strip().strip('"').strip("'")

        # Strip common filler prefixes Gemini adds despite instructions
        FILLER_PREFIXES = [
            "clinical takeaway:", "takeaway:", "key insight:", "insight:",
            "the clinical takeaway is:", "one-sentence takeaway:",
            "the takeaway is:", "in one sentence:",
        ]
        result_lower = result.lower()
        for prefix in FILLER_PREFIXES:
            if result_lower.startswith(prefix):
                result = result[len(prefix):].strip()
                break

        # Reject error strings, very short fragments, or unclosed sentences
        is_truncated = len(result) > 10 and not result[-1] in ".!?)"
        if ("WARNING" in result or "failed" in result.lower()
                or "error" in result.lower() or len(result) < 20
                or len(result) > 400 or is_truncated):
            raise ValueError("LLM returned invalid or truncated insight")
        return result
    except Exception:
        pass

    # Fallback: cosine-similarity sentence picker
    return extract_clinical_insight(article, patient_context)


# ── Secondary Query Builder ────────────────────────────────────────────────────

def _build_secondary_query(primary_query: str, patient_info: dict) -> str:
    """
    Build a complementary PubMed query orthogonal to the primary one.
    Targets intervention/management literature that the symptom-signal
    primary query may miss (e.g. weaning protocols, vasopressor guidelines).

    Returns an empty string if no useful secondary angle can be constructed
    (avoids a wasted network round-trip).
    """
    if not patient_info:
        return ""

    ventilated = patient_info.get("ventilated", False)
    diag = str(patient_info.get("diagnosis_str", "")).lower()
    apache = patient_info.get("apache_score", 0) or 0

    # Ventilated patients: add protocol / weaning management angle
    if ventilated:
        return "mechanical ventilation weaning protocol ICU outcomes"

    # Sepsis / infection angle when not already the primary query lead
    if "sepsis" in diag or "infect" in diag or "bacteremia" in diag:
        if "sepsis" not in primary_query.lower():
            return "sepsis management bundle early goal-directed therapy ICU"

    # High severity (APACHE > 20) — add mortality / organ failure angle
    try:
        if int(apache) > 20:
            return "critical illness organ failure ICU mortality prediction"
    except (ValueError, TypeError):
        pass

    # Respiratory failure angle when not already covered
    if any(k in diag for k in ["respiratory", "pneumonia", "ards"]):
        if "respiratory" not in primary_query.lower():
            return "acute respiratory failure lung protective ventilation outcomes"

    return ""


# ── Main Hybrid RAG Function ───────────────────────────────────────────────────

def run_hybrid_rag(
    clinical_query:       str,
    patient_summary:      str,
    patient_info:         dict  = None,
    temporal_phases:      dict  = None,
    top_k:                int   = 5,
    confidence_threshold: float = 0.3,
    use_cache:            bool  = True,
    apply_quality_gate:   bool  = True,
) -> dict:
    """
    Full hybrid RAG pipeline with QC layer:
      0. Cache lookup — return stable result for identical patient context  [QC#1]
      1. Build clinically-specific query from patient context
      2. BM25 sparse retrieval via PubMed
      3. Demographic filter
      4. Clinical keyword relevance filter
      5. Evidence quality scoring: 0.5*semantic + 0.3*recency + 0.2*quality
      6. LLM-powered clinical insight extraction per article
      7. Adaptive expansion if confidence low
      8. RAG quality gate — hard relevance threshold + coherence check      [QC#3]
      9. Store result in cache                                               [QC#1]
    """
    print(f"\n Component B: Hybrid RAG Retrieval")

    # ── Step 0: Cache lookup ──────────────────────────────────────────────────
    RAGCache, RAGQualityGate, _patient_fingerprint, _extract_patient_concepts = _qc()
    fingerprint = _patient_fingerprint(patient_info or {}, temporal_phases or {}) if _patient_fingerprint else "nocache"

    if use_cache and RAGCache:
        # Build the query first so the cache key is query-specific
        if patient_info and temporal_phases:
            _preview_query = build_clinical_query(patient_info, temporal_phases, clinical_query)
        else:
            _preview_query = clinical_query
        cached = RAGCache.get(_preview_query, fingerprint)
        if cached is not None:
            return cached

    # ── Step 1: Build specific query ─────────────────────────────────────────
    if patient_info and temporal_phases:
        specific_query = build_clinical_query(patient_info, temporal_phases, clinical_query)
        print(f"   Specific query: {specific_query[:80]}")
    else:
        specific_query = clinical_query
        print(f"   Query: {specific_query[:80]}...")

    relevant_groups  = _get_query_keywords(specific_query)
    demo_filters     = build_demographic_filters(patient_info)        # [#3]
    print(f"   Clinical focus: {', '.join(relevant_groups)}")
    print(f"   Patient filters: include={demo_filters['include'][:3]}... "
          f"exclude={demo_filters['exclude'][:3]}...")

    # Step 2: PubMed search
    print(f"    Phase 1: BM25 PubMed search...")
    pmids = pubmed_search(specific_query, max_results=20)

    if not pmids:
        print("     No PubMed results. Trying fallback queries.")
        pmids = pubmed_search("ICU critical care management sepsis", max_results=15)
        if not pmids:
            pmids = pubmed_search("intensive care unit outcomes treatment", max_results=15)

    print(f"    Retrieved {len(pmids)} candidate articles")

    # Second complementary query to broaden evidence coverage
    second_query = _build_secondary_query(specific_query, patient_info)
    if second_query:
        extra_pmids = pubmed_search(second_query, max_results=10)
        merged_pmids = list(dict.fromkeys(pmids + [p for p in extra_pmids if p not in pmids]))
        if len(merged_pmids) > len(pmids):
            print(f"    Secondary query '{second_query[:50]}' added {len(merged_pmids)-len(pmids)} more PMIDs")
            pmids = merged_pmids

    articles = pubmed_fetch_abstracts(pmids[:20])

    if not articles:
        return {
            "query":        specific_query,
            "evidence":     [],
            "top_evidence": "No evidence retrieved from PubMed.",
            "confidence":   0.0,
        }

    # Step 3: Demographic filter [#3]
    demo_filtered = [a for a in articles if _passes_demographic_filter(a, demo_filters)]
    if len(demo_filtered) < 3:
        demo_filtered = articles   # fallback if filter too aggressive
    n_excluded = len(articles) - len(demo_filtered)
    if n_excluded:
        print(f"    Demographic filter: excluded {n_excluded} non-adult papers")

    # Step 4: Clinical keyword relevance filter
    filtered = [a for a in demo_filtered if _article_passes_filter(a, relevant_groups)]
    if len(filtered) < 3:
        filtered = demo_filtered
    print(f"    After all filters: {len(filtered)}/{len(articles)} articles")

    # BM25 index
    corpus      = [f"{a['title']} {a['abstract']}" for a in filtered]
    bm25        = BM25()
    bm25.fit(corpus)
    bm25_scores = dict(bm25.score(specific_query, top_k=len(filtered)))
    max_bm25    = max((s for _, s in bm25_scores.items()), default=1.0)

    # Step 5: Evidence quality scoring [#5]
    print(f"    Phase 2: Evidence quality scoring (0.5 semantic + 0.3 recency + 0.2 quality)...")
    full_query = f"{specific_query} {patient_summary}"
    ranked     = []

    for i, article in enumerate(filtered):
        doc_text       = f"{article['title']} {article['abstract']}"
        semantic_score = cosine_similarity_text(full_query, doc_text)
        norm_bm25      = bm25_scores.get(i, 0.0) / max(max_bm25, 1e-9)
        final_score, breakdown = evidence_score(semantic_score, norm_bm25, article)
        ranked.append({
            **article,
            "score":          final_score,
            "score_breakdown": breakdown,
            "recency_weight": breakdown["recency"],
            "quality_bonus":  breakdown["quality"],
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    # Drop very old articles if enough recent ones exist
    current_year = datetime.now().year
    recent_only  = [r for r in ranked if (current_year - r["year"]) <= 10]
    if len(recent_only) >= top_k:
        ranked = recent_only
    elif recent_only:
        ranked = recent_only + [r for r in ranked if r not in recent_only]

    # Adaptive expansion
    top_confidence = ranked[0]["score"] if ranked else 0.0
    if top_confidence < confidence_threshold:
        print(f"    Low confidence ({top_confidence:.3f}). Expanding search...")
        extra_pmids    = pubmed_search(f"{specific_query} treatment guidelines", max_results=15)
        extra_articles = pubmed_fetch_abstracts(extra_pmids[:10])
        for a in extra_articles:
            if not _passes_demographic_filter(a, demo_filters):
                continue
            doc_text = f"{a['title']} {a['abstract']}"
            sem      = cosine_similarity_text(full_query, doc_text)
            sc, bd   = evidence_score(sem, 0.0, a)
            ranked.append({**a, "score": sc, "score_breakdown": bd})
        ranked.sort(key=lambda x: x["score"], reverse=True)

    top_results = ranked[:top_k]

    # Step 6: LLM-powered clinical insight extraction [#6]
    print(f"    Phase 3: Extracting clinical insights via LLM...")
    for r in top_results:
        r["clinical_insight"] = extract_clinical_insight_llm(r, patient_summary)

    print(f"    Top {len(top_results)} evidence articles selected")
    for i, r in enumerate(top_results[:3]):
        age       = current_year - r["year"]
        age_label = "new" if age <= 3 else "mid" if age <= 6 else "old"
        bd        = r.get("score_breakdown", {})
        print(f"      [{i+1}] {r['title'][:55]}... ({age_label} {r['year']})")
        print(f"           Score={r['score']:.3f} "
              f"(sem={bd.get('semantic',0):.2f} "
              f"rec={bd.get('recency',0):.2f} "
              f"qual={bd.get('quality',0):.2f})")
        print(f"           Insight: {r['clinical_insight'][:100]}...")

    # Format for LLM — includes quality breakdown and LLM insight
    evidence_text = "\n\n".join([
        f"[{i+1}] Title: {r['title']}\n"
        f"    Year: {r['year']} | Score: {r['score']:.3f} "
        f"(quality={r.get('quality_bonus',0):.2f})\n"
        f"    Key Insight: {r['clinical_insight']}\n"
        f"    Abstract: {r['abstract'][:600]}..."
        for i, r in enumerate(top_results)
    ])

    # ── Step 8: RAG quality gate ──────────────────────────────────────────────
    if apply_quality_gate and RAGQualityGate and _extract_patient_concepts:
        patient_concepts = _extract_patient_concepts(patient_info or {}, temporal_phases or {})
        rag_result_raw = {
            "query":        specific_query,
            "evidence":     top_results,
            "top_evidence": evidence_text,
            "confidence":   top_confidence,
        }
        final_result = RAGQualityGate.evaluate(rag_result_raw, patient_concepts)
        # Rebuild top_evidence from filtered evidence if gate modified the list
        if not final_result.get("fallback_used") and final_result.get("evidence"):
            filtered_ev = final_result["evidence"]
            final_result["top_evidence"] = "\n\n".join([
                f"[{i+1}] Title: {r['title']}\n"
                f"    Year: {r['year']} | Score: {r['score']:.3f} "
                f"(quality={r.get('quality_bonus',0):.2f})\n"
                f"    Key Insight: {r.get('clinical_insight','')}\n"
                f"    Abstract: {r['abstract'][:600]}..."
                for i, r in enumerate(filtered_ev)
            ])
    else:
        final_result = {
            "query":        specific_query,
            "evidence":     top_results,
            "top_evidence": evidence_text,
            "confidence":   top_confidence,
            "quality_gate": "SKIP",
            "fallback_used": False,
        }

    # ── Step 9: Cache store ───────────────────────────────────────────────────
    if use_cache and RAGCache:
        RAGCache.put(specific_query, fingerprint, final_result)

    return final_result