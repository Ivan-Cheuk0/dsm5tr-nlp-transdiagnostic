"""
DSM-5-TR Symptom Vectorization & Cross-Model Comparison Pipeline
================================================================
Encodes symptom-bearing criteria from `dsm_symptoms_cleaned_full.json` into
dense 768-dimensional vectors using:
  1. `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` (BioBERT NLI/STS)
  2. `NeuML/pubmedbert-base-embeddings` (Uncased PubMedBERT contrastive embeddings)

Filters out non-symptom administrative criteria (structural headers, differential
exclusions, medical/substance attribution, clinical distress/impairment boilerplate,
and pure onset/duration/age thresholds) and exports OSF-ready artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr


MODELS = {
    "biobert_nli_stsb": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "pubmedbert_embeddings": "NeuML/pubmedbert-base-embeddings",
}


def preprocess_for_embedding(
    text: str,
    strip_enumerators: bool = True,
    strip_coding_leaks: bool = True,
) -> str:
    """
    Applies syntax-preserving cleanup prior to transformer encoding:
    - Preserves stopwords and clinical grammar/inflection (no lemmatization).
    - Removes inline parenthetical `(note: ...)` blocks, coding notes, and ICD-10 codes.
    - Strips leading list enumerators ('a. ', '1. ', '2a. ', 'i. ').
    """
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()

    if strip_coding_leaks:
        cleaned = re.split(r"\bcoding note\b", cleaned, flags=re.IGNORECASE)[0].strip()
        cleaned = re.sub(r"\(\s*note:.*?\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b[a-z]\d{2}(?:\.[a-z0-9]{1,5})?\b", "", cleaned, flags=re.IGNORECASE)

    if strip_enumerators:
        cleaned = re.sub(
            r"^(?:[a-z]|\d+[a-z]{0,2}|i{1,3}|iv|v)\.\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def classify_criterion(row: dict, next_row: dict | None = None) -> str:
    """
    Classifies a DSM-5-TR criterion row as either `'symptom'` or a specific
    `'non_symptom:<category>'` rule so administrative/structural criteria do not
    form artificial transdiagnostic clusters.
    """
    rt = row["raw_text"].strip()
    # Strip trailing parenthetical notes before checking structural endings
    rt_no_note = re.sub(r"\s*\(\s*Note:.*?\)\s*$", "", rt, flags=re.IGNORECASE).strip()
    ct = re.sub(
        r"^(?:[a-z]|\d+[a-z]{0,2}|i{1,3}|iv|v)\.\s+",
        "",
        row["cleaned_text"],
        flags=re.IGNORECASE,
    ).strip()
    ct_no_note = re.sub(r"\s*\(\s*note:.*?\)\s*$", "", ct, flags=re.IGNORECASE).strip()

    cid = row["criterion_id"]
    lvl = row["criterion_level"]

    has_children = (
        next_row is not None
        and next_row["disorder"] == row["disorder"]
        and (
            (lvl == "main_criterion" and next_row["criterion_level"] == "sub_symptom")
            or (cid.isdigit() and next_row["criterion_id"].startswith(cid) and next_row["criterion_id"] != cid)
            or (re.match(r"^\d+[a-z]$", cid) and next_row["criterion_id"].startswith(cid) and next_row["criterion_id"] != cid)
        )
    )

    # 1. Structural parent headers that introduce sub-symptoms
    if rt_no_note.endswith(":"):
        if re.search(
            r"^("
            r"either \(1\) or \(2\)|core diagnostic features|suggestive diagnostic features|"
            r"behavioral variant|language variant|three or more of the following.*|"
            r"all three of the following.*|one \(or more\) of the following.*|"
            r"two \(or more\) of the following.*|at least (?:one|two|three|four|five|six) of the following.*|"
            r"presence of (?:one|two|three|four|five|six|\d+) \(or more\) of the following.*|"
            r"both of the following.*|either of the following.*"
            r"):$",
            ct_no_note,
        ):
            return "non_symptom:structural_header"
        if has_children and re.search(
            r"(?:as manifested by|as indicated by|as characterized by|include the following|"
            r"of the following|based on|as defined by|as suggested by|presence of|"
            r"following criteria|following symptoms|following expectations)[^:]*:$",
            ct_no_note,
        ):
            return "non_symptom:structural_header"

    if has_children and re.search(
        r"\b(?:five|four|three|two|six|one|\d+) \(or more\) of the following\b",
        ct_no_note,
    ):
        return "non_symptom:structural_header"

    # 2. Cross-reference / prerequisite criteria
    if (
        re.match(r"^(?:the )?criteria (?:are|have been| have never been) met for\b", ct_no_note)
        or re.match(r"^criterion a for\b[^.]*\bhas never been met\b", ct_no_note)
        or re.match(r"^between episodes of\b", ct_no_note)
    ):
        return "non_symptom:structural_prereq"

    # 3. Differential diagnosis, medical/substance attribution, or rule-out criteria
    if re.search(
        r"("
        r"\bnot better explained by\b|\bnot better accounted for by\b|"
        r"\bnot attributable to\b|\bdoes not occur exclusively\b|\bdo not occur exclusively\b|"
        r"\bis not occurring exclusively\b|\bare not occurring exclusively\b|"
        r"\bhave been ruled out\b|\bhas never been a manic\b|\bnever been a manic\b|"
        r"\bhave never been met\b|\bnever met criteria for\b|"
        r"\bthe additional diagnosis of\b[^.]*\bis made only if\b|"
        r"^there is evidence (?:from|of)\b|^there is documented infection\b|"
        r"^the (?:disturbance|disorder) occurs in the setting of\b|^there is clinically established\b|"
        r"^the involved substance or medication\b|^the temporal course of the neurocognitive deficits\b|"
        r"^no evidence of mixed etiology\b|^the parkinson’s disease clearly precedes\b|"
        r"^evidence of (?:a causative|disproportionate)\b|^clinical criteria are supported by neuroimaging\b|"
        r"^both clinical and genetic\b|^the diagnosis is not applied to\b|"
        r"^the act was not (?:initiated|undertaken)\b|^the act does not meet criteria for\b|"
        r"^the individual is not experiencing\b|"
        r"^more than minimal exposure to alcohol during gestation\b|"
        r"^the fetish(?:es| objects) are not limited to\b|"
        r"^the behavior is not part of\b|^the symptoms do not meet criteria for\b|"
        r"^if manic or major depressive episodes have occurred\b|"
        r"^if another medical condition is present\b|"
        r"^if there is a history of\b"
        r")",
        ct_no_note,
    ):
        return "non_symptom:exclusion_or_etiology"

    # 4. Clinical distress or functional impairment boilerplate
    if (
        re.search(
            r"("
            r"\bclinically significant distress\b|\bclinically significant impairment\b|"
            r"\bimpairment in social, occupational\b|\bimpairment in social, academic\b|"
            r"\bfunctional limitations in effective communication\b|\blimitations in effective communication\b|"
            r"\binterfere with social, academic\b|\binterferes with functioning or development\b|"
            r"\binterfere with, or reduce the quality of, social\b|"
            r"\binterferes with daily routine\b|\binterfere with independence in everyday\b|"
            r"\bdo not interfere with capacity for independence\b|"
            r"\bsignificantly and persistently interferes with activities of daily living\b|"
            r"\binterferes with social, academic, or other activities\b|"
            r"\blevel of functioning in one or more major areas\b|"
            r"\bfunctioning is not markedly impaired\b|"
            r"\bsufficiently severe to cause marked impairment\b|"
            r"\bsufficiently distressing and disabling\b|"
            r"\bunequivocal change in functioning\b|\bnot severe enough to cause marked impairment\b|"
            r"\bchange in functioning are observable by others\b|"
            r"\bsubstantially and quantifiably below those expected\b|"
            r"\bcauses? marked distress or interpersonal difficulty\b|"
            r"\bthe communication deficits, collectively, are not\b|"
            r"\bdistress or impairment in social, occupational\b"
            r")",
            ct_no_note,
        )
        and not has_children
        and lvl == "main_criterion"
    ):
        return "non_symptom:distress_or_impairment"

    # 5. Pure onset, age, duration, setting count, or course threshold criteria
    if (
        re.search(
            r"^("
            r"(?:the )?onset (?:of|is|before)\b|"
            r"symptoms must be present in the early developmental period\b|"
            r"several inattentive or hyperactive-impulsive symptoms (?:were present prior to|are present in two or more settings)\b|"
            r"duration of (?:the|an episode of the) (?:disturbance|episode|condition)\b|"
            r"continuous signs of the disturbance persist for\b|"
            r"an episode of the disorder lasts\b|"
            r"the tics (?:may wax and wane in frequency but have persisted|have been present for less than)\b|"
            r"the (?:disturbance|symptoms|fear|anxiety|avoidance|condition|pattern|difficulties|deficits|episodes|bouts|criteria) (?:is|are|lasts?|persists?|has been present|must have been present|have persisted|develops over a short period|is stable and of long duration|occur)\b[^.]*\b(?:at least|more than|less than|months?|weeks?|years?|days?|hours?|early adulthood|adolescence|developmental period)|"
            r"symptom\(s\) must have (?:been present|begun or worsened)\b|"
            r"the individual (?:is at least|has never|is at least age|experiencing the arousal)\b|"
            r"there is evidence of conduct disorder with onset before\b|"
            r"the learning difficulties begin during school-age years\b|"
            r"the neurocognitive disorder presents immediately after\b|"
            r"the neurocognitive syndrome is temporally related\b|"
            r"onset of the cognitive deficits is temporally related\b|"
            r"there is insidious onset\b|"
            r"the (?:disorder|disturbance) has (?:an )?insidious onset\b|"
            r"in the last year, the individual has, on 5 or more days\b"
            r")",
            ct_no_note,
        )
        and not has_children
    ):
        return "non_symptom:onset_duration_age"

    return "symptom"


def load_and_filter_data(
    json_path: Path,
    filter_non_symptoms: bool = True,
    strip_enumerators: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads `dsm_symptoms_cleaned_full.json`, classifies every row (`symptom` vs `non_symptom:*`),
    and returns `(full_audited_df, filtered_symptom_df)`.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    classifications = []
    for i, r in enumerate(records):
        nxt = records[i + 1] if i + 1 < len(records) else None
        classifications.append(classify_criterion(r, nxt))

    df_all = pd.DataFrame(records)
    df_all["criterion_category"] = classifications
    df_all["is_symptom"] = df_all["criterion_category"] == "symptom"
    df_all["embedding_text"] = df_all["cleaned_text"].apply(
        lambda t: preprocess_for_embedding(t, strip_enumerators=strip_enumerators)
    )

    if filter_non_symptoms:
        df_filtered = df_all[df_all["is_symptom"]].copy().reset_index(drop=True)
    else:
        df_filtered = df_all.copy().reset_index(drop=True)

    return df_all, df_filtered


def encode_with_model(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
    max_seq_length: int = 256,
) -> np.ndarray:
    """
    Loads a SentenceTransformer checkpoint, ensures `max_seq_length >= 256` so longer
    DSM-5-TR criteria are not truncated at 100 tokens, and returns L2-normalized
    768-dimensional embeddings.
    """
    print(f"\n--- Loading model: {model_name} ---")
    model = SentenceTransformer(model_name)
    if getattr(model, "max_seq_length", 0) < max_seq_length:
        print(f"Expanding model.max_seq_length from {model.max_seq_length} -> {max_seq_length}")
        model.max_seq_length = max_seq_length

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"Generated embeddings shape: {embeddings.shape} (dtype={embeddings.dtype})")
    return embeddings


def extract_top_cross_disorder_pairs(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    top_k: int = 100,
) -> pd.DataFrame:
    """Extracts the top cross-disorder symptom pairs ranked by cosine similarity."""
    n = len(df)
    disorders = df["disorder"].tolist()
    crit_ids = df["criterion_id"].tolist()
    levels = df["criterion_level"].tolist()
    texts = df["embedding_text"].tolist()

    # Upper triangle indices where disorders differ
    iu, ju = np.triu_indices(n, k=1)
    diff_mask = np.array([disorders[i] != disorders[j] for i, j in zip(iu, ju)], dtype=bool)
    iu_diff = iu[diff_mask]
    ju_diff = ju[diff_mask]
    sims = sim_matrix[iu_diff, ju_diff]

    # Partition top_k fastest via argpartition
    k = min(top_k, len(sims))
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    rows = []
    for idx in top_idx:
        i, j = int(iu_diff[idx]), int(ju_diff[idx])
        rows.append(
            {
                "disorder_1": disorders[i],
                "criterion_id_1": crit_ids[i],
                "level_1": levels[i],
                "symptom_1": texts[i],
                "disorder_2": disorders[j],
                "criterion_id_2": crit_ids[j],
                "level_2": levels[j],
                "symptom_2": texts[j],
                "cosine_similarity": float(sims[idx]),
            }
        )
    return pd.DataFrame(rows)


def compute_bridge_symptom_centrality(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """
    Identifies transdiagnostic bridge symptoms by counting how many distinct *other*
    disorders have at least one symptom exceeding `threshold` cosine similarity, plus
    computing mean top-10 cross-disorder cosine similarity.
    """
    n = len(df)
    disorders = df["disorder"].tolist()
    results = []

    for i in range(n):
        d_i = disorders[i]
        cross_mask = np.array([disorders[j] != d_i for j in range(n)], dtype=bool)
        cross_sims = sim_matrix[i, cross_mask]
        cross_disorders = [disorders[j] for j in range(n) if disorders[j] != d_i]

        above_thresh_disorders = {
            d_j for sim_val, d_j in zip(cross_sims, cross_disorders) if sim_val >= threshold
        }
        top10_mean = float(np.mean(np.sort(cross_sims)[-10:])) if len(cross_sims) >= 10 else float(np.mean(cross_sims))
        max_sim = float(np.max(cross_sims)) if len(cross_sims) > 0 else 0.0

        results.append(
            {
                "disorder": d_i,
                "criterion_id": df.loc[i, "criterion_id"],
                "criterion_level": df.loc[i, "criterion_level"],
                "embedding_text": df.loc[i, "embedding_text"],
                "bridged_disorder_count": len(above_thresh_disorders),
                "mean_top10_cross_sim": top10_mean,
                "max_cross_sim": max_sim,
            }
        )

    bridge_df = pd.DataFrame(results)
    bridge_df.sort_values(
        by=["bridged_disorder_count", "mean_top10_cross_sim"],
        ascending=[False, False],
        inplace=True,
    )
    return bridge_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Dual-model BioBERT + PubMedBERT DSM-5-TR embedding pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dsm_symptoms_cleaned_full.json"),
        help="Input cleaned DSM-5-TR JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for embeddings and cross-model comparisons.",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Disable non-symptom filtering and embed all 1,568 rows.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for SentenceTransformer encoding.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_all, df_sym = load_and_filter_data(
        args.input,
        filter_non_symptoms=not args.no_filter,
        strip_enumerators=True,
    )

    print("=== Criterion Filtering Summary ===")
    counts = Counter(df_all["criterion_category"])
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    print(f"Total input rows: {len(df_all)} -> Symptom-bearing rows encoded: {len(df_sym)} across {df_sym['disorder'].nunique()} disorders.")

    # Save audit table and filtered symptom JSON
    audit_path = args.output_dir / "criterion_filter_audit.csv"
    df_all[["disorder", "criterion_id", "criterion_level", "criterion_category", "is_symptom", "raw_text", "embedding_text"]].to_csv(
        audit_path, index=False
    )

    filtered_json_path = args.output_dir / "dsm_symptoms_filtered_only.json"
    df_sym[["disorder", "criterion_id", "criterion_level", "raw_text", "cleaned_text", "embedding_text"]].to_json(
        filtered_json_path, orient="records", indent=2, force_ascii=False
    )

    texts = df_sym["embedding_text"].tolist()
    sim_matrices = {}
    embeddings_dict = {}

    for short_name, model_id in MODELS.items():
        emb = encode_with_model(texts, model_id, batch_size=args.batch_size)
        embeddings_dict[short_name] = emb

        # Save 768-d numpy matrix
        npy_path = args.output_dir / f"embeddings_{short_name}_768d.npy"
        np.save(npy_path, emb)

        # Compute cosine similarity matrix
        sim_mat = cosine_similarity(emb)
        sim_matrices[short_name] = sim_mat

        # Save top 100 cross-disorder symptom pairs
        top_pairs = extract_top_cross_disorder_pairs(df_sym, sim_mat, top_k=100)
        pairs_path = args.output_dir / f"top_cross_disorder_pairs_{short_name}.csv"
        top_pairs.to_csv(pairs_path, index=False)

        # Save bridge symptom centrality rankings
        thresh = 0.80 if short_name == "biobert_nli_stsb" else 0.75
        bridge_df = compute_bridge_symptom_centrality(df_sym, sim_mat, threshold=thresh)
        bridge_path = args.output_dir / f"bridge_symptoms_{short_name}.csv"
        bridge_df.to_csv(bridge_path, index=False)

    # Compare the two models' pairwise cross-disorder cosine similarities
    n = len(df_sym)
    iu, ju = np.triu_indices(n, k=1)
    disorders = df_sym["disorder"].tolist()
    diff_mask = np.array([disorders[i] != disorders[j] for i, j in zip(iu, ju)], dtype=bool)

    sims_biobert = sim_matrices["biobert_nli_stsb"][iu[diff_mask], ju[diff_mask]]
    sims_pubmed = sim_matrices["pubmedbert_embeddings"][iu[diff_mask], ju[diff_mask]]

    r_pearson, _ = pearsonr(sims_biobert, sims_pubmed)
    r_spearman, _ = spearmanr(sims_biobert, sims_pubmed)

    # Compute consensus bridge pairs (standardized z-score average across both models)
    z_bio = (sims_biobert - np.mean(sims_biobert)) / np.std(sims_biobert)
    z_pub = (sims_pubmed - np.mean(sims_pubmed)) / np.std(sims_pubmed)
    consensus_score = 0.5 * (z_bio + z_pub)

    iu_diff = iu[diff_mask]
    ju_diff = ju[diff_mask]
    top_k = min(100, len(consensus_score))
    top_idx = np.argpartition(consensus_score, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(consensus_score[top_idx])[::-1]]

    consensus_rows = []
    crit_ids = df_sym["criterion_id"].tolist()
    levels = df_sym["criterion_level"].tolist()
    for idx in top_idx:
        i, j = int(iu_diff[idx]), int(ju_diff[idx])
        consensus_rows.append(
            {
                "disorder_1": disorders[i],
                "criterion_id_1": crit_ids[i],
                "level_1": levels[i],
                "symptom_1": texts[i],
                "disorder_2": disorders[j],
                "criterion_id_2": crit_ids[j],
                "level_2": levels[j],
                "symptom_2": texts[j],
                "biobert_cosine": float(sims_biobert[idx]),
                "pubmedbert_cosine": float(sims_pubmed[idx]),
                "mean_cosine": float(0.5 * (sims_biobert[idx] + sims_pubmed[idx])),
                "consensus_z_score": float(consensus_score[idx]),
            }
        )

    consensus_df = pd.DataFrame(consensus_rows)
    consensus_path = args.output_dir / "consensus_cross_disorder_bridge_pairs.csv"
    consensus_df.to_csv(consensus_path, index=False)

    summary_stats = {
        "total_criteria_in_json": int(len(df_all)),
        "filtered_symptom_criteria": int(len(df_sym)),
        "unique_disorders_represented": int(df_sym["disorder"].nunique()),
        "filter_breakdown": dict(counts),
        "models": MODELS,
        "cross_disorder_pair_count": int(len(sims_biobert)),
        "biobert_cosine_mean": float(np.mean(sims_biobert)),
        "biobert_cosine_std": float(np.std(sims_biobert)),
        "pubmedbert_cosine_mean": float(np.mean(sims_pubmed)),
        "pubmedbert_cosine_std": float(np.std(sims_pubmed)),
        "inter_model_pearson_r": float(r_pearson),
        "inter_model_spearman_rho": float(r_spearman),
    }
    summary_path = args.output_dir / "embedding_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n=== Cross-Model Comparison Summary ===")
    print(f"Cross-disorder pairs evaluated: {len(sims_biobert):,}")
    print(f"BioBERT NLI/STS cosine mean ± SD: {np.mean(sims_biobert):.4f} ± {np.std(sims_biobert):.4f}")
    print(f"PubMedBERT Embeddings cosine mean ± SD: {np.mean(sims_pubmed):.4f} ± {np.std(sims_pubmed):.4f}")
    print(f"Inter-model agreement: Pearson r = {r_pearson:.4f}, Spearman rho = {r_spearman:.4f}")
    print(f"Saved consensus bridge pairs to: {consensus_path}")


if __name__ == "__main__":
    main()
