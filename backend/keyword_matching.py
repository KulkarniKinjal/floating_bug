"""
keyword_matching.py
--------------------
Person 2's "keyword leg": BM25 overall relevance score + alias-aware
skill-presence detection with evidence sentences.

Built against the AGREED SHARED FORMAT:

    resumes = {"resume_01": "full text...", "resume_02": "full text...", ...}

    jd = {
        "must_have": ["Node.js", "MongoDB", "REST APIs"],
        "nice_to_have": ["Docker", "AWS"],
        "role_summary": "short text",
    }

    # per-candidate result this module fills its part of:
    {
        "candidate_id": "resume_07",
        "keyword_score": 0.75,
        "skill_coverage": {
            "Node.js":    {"present": True,  "evidence": "built REST APIs with Express..."},
            "Kubernetes": {"present": False, "evidence": None},
        },
    }

skill_coverage is FLAT — must_have and nice_to_have skills live in the same
dict, keyed by exact skill name. Whether a given skill counts as "required"
is decided later, by whoever reads jd["must_have"] vs jd["nice_to_have"] —
not by this file. This file only ever answers: is this skill string (or an
alias of it) present in this resume text, and if so, which sentence proves it.

Nothing here talks to the semantic leg — this file only knows about exact
text presence (via aliases), never embeddings.
"""

import re
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Alias map — extend this as you hit real resumes during testing.
# Keys are the canonical skill name (as it will appear in the JD's
# required/nice-to-have list); values are alternate strings that should
# also count as a match.
# ---------------------------------------------------------------------------

ALIAS_MAP: Dict[str, List[str]] = {
    "javascript": ["js", "javascript", "es6", "ecmascript"],
    "typescript": ["ts", "typescript"],
    "python": ["python", "py"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "natural language processing": ["natural language processing", "nlp"],
    "kubernetes": ["kubernetes", "k8s"],
    "docker": ["docker", "containerization", "containerisation"],
    "amazon web services": ["aws", "amazon web services"],
    "google cloud platform": ["gcp", "google cloud platform", "google cloud"],
    "microsoft azure": ["azure", "microsoft azure"],
    "rest apis": ["rest api", "rest apis", "restful api", "restful apis", "rest"],
    "graphql": ["graphql", "graph ql"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "structured query language"],
    "nosql": ["nosql", "no-sql"],
    "mongodb": ["mongodb", "mongo"],
    "node.js": ["node.js", "nodejs", "node"],
    "react": ["react", "react.js", "reactjs"],
    "vue": ["vue", "vue.js", "vuejs"],
    "angular": ["angular", "angular.js", "angularjs"],
    "git": ["git", "github", "gitlab", "version control"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "data structures and algorithms": ["dsa", "data structures and algorithms",
                                        "data structures & algorithms"],
    "object oriented programming": ["oop", "object oriented programming",
                                     "object-oriented programming"],
}


def _aliases_for(skill: str) -> List[str]:
    """Return the alias list for a skill, falling back to just the skill name itself."""
    key = skill.strip().lower()
    return ALIAS_MAP.get(key, [key])


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9\.\+#/]+")


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Collapse excessive whitespace first so sentence boundaries are cleaner
    cleaned = re.sub(r"\s+", " ", text).strip()
    raw_sentences = _SENTENCE_SPLIT_RE.split(cleaned)
    return [s.strip() for s in raw_sentences if s.strip()]


def tokenize(text: str) -> List[str]:
    """Simple lowercase tokenizer that keeps things like 'node.js', 'c++', 'ci/cd' intact."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _alias_pattern(alias: str) -> re.Pattern:
    """
    Build a whole-phrase, case-insensitive regex for an alias.
    Escapes special chars but still matches things like 'node.js' or 'c++'
    as a contiguous phrase with word boundaries around it where sensible.
    """
    escaped = re.escape(alias.strip())
    # Use lookaround boundaries instead of \b so 'c++' / 'node.js' still work
    # (\b doesn't behave well around punctuation-heavy tokens).
    return re.compile(rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 1. Skill coverage + evidence (the alias-aware presence check)
# ---------------------------------------------------------------------------

def find_skill_evidence(resume_text: str, skill: str) -> Optional[str]:
    """
    Search resume_text for `skill` or any of its aliases.
    Returns the first sentence containing a match, or None if not found.
    """
    if not resume_text:
        return None

    sentences = _split_sentences(resume_text)
    aliases = _aliases_for(skill)
    patterns = [_alias_pattern(a) for a in aliases]

    for sentence in sentences:
        for pattern in patterns:
            if pattern.search(sentence):
                return sentence
    return None


def compute_skill_coverage(resume_text: str, jd: Dict) -> Dict[str, Dict]:
    """
    Build the FLAT skill_coverage dict for one candidate, per the agreed
    shared format:

        {"Node.js": {"present": True, "evidence": "..."}, ...}

    Covers every skill in jd["must_have"] + jd["nice_to_have"] combined —
    this file doesn't track which bucket a skill came from; that's read
    back out of jd by whoever needs it (explanation.py).
    """
    all_skills = list(jd.get("must_have", []) or []) + list(jd.get("nice_to_have", []) or [])

    coverage = {}
    for skill in all_skills:
        evidence = find_skill_evidence(resume_text, skill)
        coverage[skill] = {
            "present": evidence is not None,
            "evidence": evidence,
        }
    return coverage


# ---------------------------------------------------------------------------
# 2. BM25 overall keyword score (this becomes `keyword_score`)
# ---------------------------------------------------------------------------

def build_jd_query_text(jd: Dict) -> str:
    """
    Turn the structured jd dict into one query string for BM25.
    Skills are repeated a few times to weight them above the free-text
    role_summary — a resume that mentions the actual required skills should
    score higher than one that just shares generic words with the summary.
    """
    must_have = jd.get("must_have", []) or []
    nice_to_have = jd.get("nice_to_have", []) or []
    role_summary = jd.get("role_summary", "") or ""

    weighted_skills = " ".join(must_have * 3 + nice_to_have * 2)
    return f"{weighted_skills} {role_summary}".strip()


def compute_keyword_scores(resume_texts: List[str], jd: Dict) -> List[float]:
    """
    Score every resume against the structured JD with BM25, then min-max
    normalize to 0-1 so it's comparable with the semantic leg's cosine scores.

    resume_texts: list of raw resume text, in the ORDER you want scores back in.
    jd:           the structured jd dict (must_have / nice_to_have / role_summary).

    Returns: list of floats, same length/order as resume_texts.
    """
    if not resume_texts:
        return []

    tokenized_corpus = [tokenize(t) for t in resume_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenize(build_jd_query_text(jd))
    raw_scores = bm25.get_scores(query_tokens)  # numpy array, higher = more relevant

    return _min_max_normalize(list(raw_scores))


def _min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        # All candidates scored identically (or only 1 candidate) — avoid /0,
        # give everyone a flat mid score instead of a misleading 0 or 1.
        return [0.5 for _ in scores]
    return [round((s - lo) / (hi - lo), 4) for s in scores]


# ---------------------------------------------------------------------------
# 3. Convenience: run the whole keyword leg for a whole batch in one call
# ---------------------------------------------------------------------------

def run_keyword_leg(resumes: Dict[str, str], jd: Dict) -> List[Dict]:
    """
    Given the shared `resumes` dict ({resume_id: text}) and the shared
    structured `jd` dict, return a list of partial candidate dicts matching
    the agreed per-candidate contract — just the fields this leg owns:

        [
          {
            "candidate_id": "resume_01",
            "keyword_score": 0.83,
            "skill_coverage": {"Node.js": {"present": True, "evidence": "..."}, ...},
          },
          ...
        ]

    The fusion step (final_score) and the semantic leg (semantic_score) are
    merged in separately — this function never touches those fields.
    """
    candidate_ids = list(resumes.keys())
    resume_texts = [resumes[cid] for cid in candidate_ids]

    keyword_scores = compute_keyword_scores(resume_texts, jd)

    output = []
    for cid, text, kw_score in zip(candidate_ids, resume_texts, keyword_scores):
        coverage = compute_skill_coverage(text, jd)
        output.append({
            "candidate_id": cid,
            "keyword_score": kw_score,
            "skill_coverage": coverage,
        })
    return output
