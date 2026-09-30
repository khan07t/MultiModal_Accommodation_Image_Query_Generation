"""Automatic checks on generated questions."""

import re

import numpy as np

GENERIC_PATTERNS = [
    r"what is the (overall )?quality",
    r"does the .* (have|include) (modern|standard) features",
    r"is the .* well-maintained",
    r"what amenities are (included|available)",
    r"would this .* meet",
    r"what is the condition of",
    r"is it in good condition",
    r"does it work (well|properly)",
    r"is there (a|an) .*\?$",
    r"what features does",
    r"is the .* functional",
    r"does the .* function",
    r"is the .* adequate",
    r"is the .* suitable",
]

OBVIOUS_QUESTIONS = {
    "bathtub": ["is there a bathtub", "does the bathroom have a tub", "is there a tub in the room",
                "does it include a bathtub"],
    "tv": ["is there a tv", "does the room have a television", "is there a television", "can i see a tv"],
    "kettle": ["is there a kettle", "can you make tea or coffee", "is there a kettle in the room",
               "does the room have a kettle"],
    "hairdryer": ["is there a hairdryer", "do they provide a hairdryer", "is a hairdryer included",
                  "does the room have a hairdryer"],
    "mirror": ["is there a mirror", "does the bathroom have a mirror", "is a mirror provided",
               "can i see a mirror"],
}


def is_fallback_question(question, amenity=None, extended=True):
    """True if a question is too short, generic, obvious from the photo, or a fallback.

    extended=True is the rule used for the model comparison. The profile run was
    checked with the shorter rule (first ten patterns, first two obvious phrases).
    """
    if not question or not isinstance(question, str):
        return True
    q = question.lower().strip()
    if len(q) < 10:
        return True
    patterns = GENERIC_PATTERNS if extended else GENERIC_PATTERNS[:10]
    if any(re.search(p, q) for p in patterns):
        return True
    obvious = OBVIOUS_QUESTIONS.get((amenity or "").lower(), [])
    if amenity and any(o in q for o in (obvious if extended else obvious[:2])):
        return True
    return bool(re.search(r"fallback q\d+", q))


def fallback_rate(questions, amenity=None, extended=True):
    flags = [is_fallback_question(q, amenity, extended) for q in questions]
    return sum(flags), len(flags), 100 * sum(flags) / len(flags) if flags else 0.0


def length_points(question):
    """Length score used in the specificity metric: 10-20 words is ideal."""
    n = len(question.split())
    if 10 <= n <= 20:
        return 0.25
    if 8 <= n < 10 or 20 < n <= 25:
        return 0.15
    return 0.0


SPECIFIC_TERMS = ["size", "capacity", "type", "setting", "control", "feature", "modern", "clean",
                  "large", "small", "fast", "quick", "slow", "powerful", "spacious", "compact",
                  "deep", "shallow", "dual", "single", "multiple", "separate", "integrated"]


def specificity_score(question):
    """0-1: quantities, feature terms, a question verb, and a sensible length."""
    if not question or not isinstance(question, str):
        return 0.0
    q = question.lower()
    score = 0.0
    if re.search(r"\d+|multiple|several|enough|sufficient", q):
        score += 0.25
    if any(t in q for t in SPECIFIC_TERMS):
        score += 0.30
    if any(w in q for w in ["can", "does", "is", "will", "would", "could", "should", "has", "have"]):
        score += 0.20
    return score + length_points(question)


PROFILE_KEYWORDS = {
    "single": ["solo", "alone", "individual", "myself", "i ", "one person", "by myself", "my own",
               "personally", "independently"],
    "couple": ["two", "both", "together", "partners", "we ", "dual", "pair", "romantic", "shared",
               "each other", "us "],
    "group": ["multiple", "family", "group", "children", "everyone", "several", "all of us", "kids",
              "elderly", "accommodate"],
}


def profile_alignment(question, profile):
    """0-1: how clearly a question speaks to its traveller profile."""
    if not question or not isinstance(question, str):
        return 0.0
    q = question.lower()
    matches = sum(kw in q for kw in PROFILE_KEYWORDS.get(profile, []))
    if matches >= 2:
        return 1.0
    if matches == 1:
        return 0.6
    weak = {"single": ["my", "i "], "couple": ["two", "both"], "group": ["multiple", "family"]}
    return 0.4 if any(w in q for w in weak.get(profile, [])) else 0.2


# --- embedding based metrics (sentence-transformers, all-MiniLM-L6-v2) ---

def load_encoder(name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def semantic_diversity(questions, encoder):
    """1 - mean pairwise cosine similarity within one image's question set."""
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(encoder.encode(questions))
    return 1 - sim[np.triu_indices(len(sim), k=1)].mean()


def profile_distinctiveness(questions_a, questions_b, encoder, n=15):
    """1 - mean cosine similarity between two profiles' question sets."""
    from sklearn.metrics.pairwise import cosine_similarity
    return 1 - cosine_similarity(encoder.encode(questions_a[:n]), encoder.encode(questions_b[:n])).mean()


def bertscore_vs_references(questions, references):
    """Best-match BERTScore F1 of each question against the reference set."""
    from bert_score import score
    out = []
    for q in questions:
        _, _, f1 = score([q] * len(references), references, lang="en",
                         model_type="microsoft/deberta-base-mnli", verbose=False)
        out.append(f1.max().item())
    return float(np.mean(out)), float(np.std(out))
