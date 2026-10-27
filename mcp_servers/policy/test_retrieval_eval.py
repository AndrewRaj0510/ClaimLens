"""Retrieval eval on the real wordings with the real embedding model.

Each question names, per wording, a substring the section path of a correct
answer contains (e.g. "PART D" and "EXCLUSIONS"). Reports hit@1 and hit@3 for
vector-only, full-text-only and hybrid search, which is the evidence for using
hybrid. Needs the ingested dev database and the model, so it's opt-in:

    RETRIEVAL_EVAL=1 python -m pytest test_retrieval_eval.py -s
"""

import os

import pytest

from main import PolicyStore

# question -> {wording_id: substrings that must all appear in the section path}
CASES = {
    "Is a windshield broken by a rock covered?": {
        "iso_pap_1998": ["PART D"],
        "progressive_9611a": ["PART IV"],
        "geico_a30": ["SECTION III"],
    },
    "My car was damaged by hail while parked": {
        "iso_pap_1998": ["PART D"],
        "progressive_9611a": ["PART IV"],
        "geico_a30": ["SECTION III"],
    },
    "damage due to wear and tear, freezing or mechanical breakdown": {
        "iso_pap_1998": ["PART D", "EXCLUSIONS"],
        "progressive_9611a": ["PART IV", "EXCLUSIONS"],
        "geico_a30": ["SECTION III", "EXCLUSIONS"],
    },
    "How soon do I have to tell the insurer about an accident?": {
        "iso_pap_1998": ["PART E"],
        "progressive_9611a": ["PART VI"],
        "geico_a30": ["NOTICE"],
    },
    "What happens if I make a false statement about a claim?": {
        "iso_pap_1998": ["FRAUD"],
        "progressive_9611a": ["FRAUD"],
        "geico_a30": ["FRAUD"],
    },
    "What is the most you will pay for damage to my car, actual cash value?": {
        "iso_pap_1998": ["PART D", "LIMIT OF LIABILITY"],
        "progressive_9611a": ["PART IV", "LIMITS OF LIABILITY"],
        "geico_a30": ["SECTION III", "LIMIT OF LIABILITY"],
    },
    "We disagree on the amount of loss, can we use an appraisal?": {
        "iso_pap_1998": ["APPRAISAL"],
        "progressive_9611a": ["APPRAISAL"],
        "geico_a30": ["APPRAISAL"],
    },
    "Can the insurer cancel my policy?": {
        "iso_pap_1998": ["TERMINATION"],
        "progressive_9611a": ["CANCELLATION"],
        "geico_a30": ["CANCELLATION"],
    },
}
# Term-style queries, the way an agent searching a policy often phrases them.
TERM_CASES = {
    "subrogation right to recover payment": {
        "iso_pap_1998": ["RECOVER PAYMENT"],
        "progressive_9611a": ["RECOVER PAYMENT"],
        "geico_a30": ["SUBROGATION"],
    },
    "bail bonds premiums": {
        "iso_pap_1998": ["SUPPLEMENTARY PAYMENTS"],
        "progressive_9611a": ["ADDITIONAL PAYMENTS"],
        "geico_a30": ["ADDITIONAL PAYMENTS"],
    },
    "custom parts or equipment limit": {
        "iso_pap_1998": ["PART D"],
        "progressive_9611a": ["PART IV"],
        "geico_a30": ["SECTION III"],
    },
}

CANDIDATES = 20
# The two alternatives production doesn't use, kept for comparison. Full text
# ORs the query words (plainto_tsquery would AND them) and is computed on the
# fly, so the schema doesn't carry a column only the eval needs.
TEXT_ONLY = f"""
WITH q AS (
    SELECT nullif(replace(plainto_tsquery('english', %(qtext)s)::text, '&', '|'), '')::tsquery AS tsq
)
SELECT chunk_index, section FROM policy_chunks, q
WHERE wording_id = %(wording)s
  AND to_tsvector('english', section || ' ' || content) @@ q.tsq
ORDER BY ts_rank_cd(to_tsvector('english', section || ' ' || content), q.tsq) DESC
LIMIT {CANDIDATES}
"""
VECTOR_ONLY = f"""
SELECT chunk_index, section FROM policy_chunks WHERE wording_id = %(wording)s
ORDER BY embedding <=> %(qvec)s LIMIT {CANDIDATES}
"""


def rrf(*rankings, k=60):
    """Reciprocal rank fusion: score = sum of 1/(k + rank) across rankings."""
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda item: -scores[item])


def hit(sections, expected, k):
    return any(all(s in section for s in expected) for section in sections[:k])


@pytest.mark.skipif(
    not os.getenv("RETRIEVAL_EVAL"), reason="set RETRIEVAL_EVAL=1 to run"
)
def test_retrieval_quality():
    from dotenv import find_dotenv, load_dotenv
    from pgvector.psycopg import register_vector
    from psycopg_pool import ConnectionPool

    from embeddings import Embedder

    load_dotenv(find_dotenv())
    embedder = Embedder()
    pool = ConnectionPool(
        os.environ["DATABASE_URL"],
        min_size=1,
        max_size=2,
        configure=register_vector,
        open=True,
    )
    store = PolicyStore(pool, embedder)
    methods = ("production", "vector", "text", "hybrid")
    results = {}
    with pool.connection() as conn:
        for set_name, cases in (("natural", CASES), ("terms", TERM_CASES)):
            scores = {m: {1: 0, 3: 0} for m in methods}
            n = 0
            for question, per_wording in cases.items():
                qvec = embedder.query(question)
                for wording, expected in per_wording.items():
                    params = {"qvec": qvec, "qtext": question, "wording": wording}
                    vec = conn.execute(VECTOR_ONLY, params).fetchall()
                    txt = conn.execute(TEXT_ONLY, params).fetchall()
                    section_of = dict(vec + txt)
                    fused = rrf([i for i, _ in vec], [i for i, _ in txt])
                    ranked = {
                        "production": [
                            c.section for c in store.search(question, wording, 3)
                        ],
                        "vector": [s for _, s in vec],
                        "text": [s for _, s in txt],
                        "hybrid": [section_of[i] for i in fused],
                    }
                    n += 1
                    for method, sections in ranked.items():
                        for k in (1, 3):
                            scores[method][k] += hit(sections, expected, k)
                    if not hit(ranked["production"], expected, 3):
                        print("MISS", question, wording, ranked["production"])
            results[set_name] = (n, scores)
    pool.close()

    for set_name, (n, scores) in results.items():
        print(f"\n{set_name}: {n} question/wording pairs")
        for method, s in scores.items():
            print(f"{method:>11}: hit@1 {s[1] / n:.2f}  hit@3 {s[3] / n:.2f}")
    for n, scores in results.values():
        assert scores["production"][3] / n >= 0.85
