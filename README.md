# Computational Vectorization of DSM-5-TR Diagnostic Criteria: Transdiagnostic Bridge Symptoms & Symptom Clustering

An open-science computational psychology pipeline that extracts, filters, and vectorizes diagnostic criteria from the *Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition, Text Revision (DSM-5-TR)* into dense 768-dimensional biomedical embedding spaces. The pipeline quantifies cross-disorder semantic overlap across all 20 DSM-5-TR diagnostic chapters, identifies transdiagnostic bridge symptoms, and evaluates empirical symptom clustering using Hierarchical Density-Based Spatial Clustering of Applications with Noise (**HDBSCAN**) and **Support Vector Clustering (SVC)**.

---

## 1. Project Overview & Research Objectives

Categorical nosologies such as the DSM-5-TR organize psychopathology into discrete chapters and disorders, yet high clinical comorbidity and shared symptom criteria suggest substantial transdiagnostic overlap. This project applies domain-specific biomedical transformer models (**SentenceTransformers**) to test three computational questions:

1. **Cross-Disorder Semantic Overlap**: Which DSM-5-TR diagnostic chapters share the highest latent semantic similarity once structural boilerplate and exclusion criteria are removed?
2. **Transdiagnostic Bridge Symptoms**: Which individual symptom constructs act as high-centrality semantic bridges connecting distinct diagnostic classes?
3. **Data-Driven Symptom Taxonomy (HDBSCAN & SVC)**: How do unsupervised density-based clusters (**HDBSCAN**) and radial basis function (**RBF**) **Support Vector Clustering (SVC)** decision boundaries compare against traditional DSM-5-TR chapter boundaries, and how sensitive are these structures to the choice of biomedical embedding model?

---

## 2. Open-Science & Copyright-Safe "Dehydrated" Architecture

To respect [American Psychiatric Association (APA) copyright and permissions policies](https://www.appi.org/support/customer-information/permissions) while enabling full computational reproducibility on GitHub and the Open Science Framework (OSF), this repository uses a **dehydrated open-science architecture**:

* **Included in Public Version Control**:
  * **All Pipeline Source Code**: `parse.py`, `check.py`, `embed_symptoms.py`, `visualize_results.py`, and `requirements.txt`.
  * **Dense 768-Dimensional Embedding Matrices (`.npy`)**: Non-consumptive, non-invertible float32 vectors (`outputs/embeddings_biobert_nli_stsb_768d.npy` and `outputs/embeddings_pubmedbert_embeddings_768d.npy`).
  * **Dehydrated Public Metadata & Checksum Index**: `outputs/dsm_symptoms_metadata_public.json` (`1,033` symptom rows) and `outputs/criterion_filter_audit_public.csv` (`1,568` total criteria rows), which record `disorder`, `chapter`, `criterion_id`, `criterion_level`, `criterion_category`, a non-verbatim `clinical_concept_label`, `word_count`, `text_sha256` checksum, 2D manifold coordinates, and HDBSCAN/SVC cluster assignments.
  * **Public Bridge Symptom Tables & Figures**: `outputs/*_public.csv`, `outputs/embedding_comparison_summary.json`, `outputs/clustering_evaluation_summary.json`, `outputs/figures/*.svg`, and `outputs/figures/dsm5tr_interactive_explorer.html`.
* **Excluded from Public Version Control via `.gitignore` (Local Rehydration Only)**:
  * The source PDF (`DSM 5 TR-APA (2022).pdf`) and verbatim full-text extracts (`dsm_symptoms_cleaned_full.json`, `outputs/dsm_symptoms_filtered_only.json`, `outputs/criterion_filter_audit.csv`, and unredacted CSVs). Researchers with institutional access to the DSM-5-TR PDF can run `python parse.py && python embed_symptoms.py` locally to rehydrate the verbatim text and verify exact checksum alignment against `text_sha256`.

```text
DSM5TR_vec/
├── .gitignore                                             # Excludes .venv/, PDFs & verbatim full-text extracts
├── parse.py                                               # End-to-end PDF diagnostic criteria parser
├── check.py                                               # Dataset continuity & structural integrity audit script
├── embed_symptoms.py                                      # Non-symptom filter & dual-model 768-d embedding pipeline
├── visualize_results.py                                   # Manifold projection, HDBSCAN, SVC & figure generator
├── requirements.txt                                       # Pinned Python dependencies for reproducibility
├── README.md                                              # Project documentation & OSF reproducibility guide
└── outputs/
    ├── dsm_symptoms_metadata_public.json                  # Dehydrated metadata, SHA-256 & 2D cluster index (1,033 rows)
    ├── criterion_filter_audit_public.csv                  # Dehydrated 1,568-row filter classification audit
    ├── embeddings_biobert_nli_stsb_768d.npy               # L2-normalized (1033, 768) float32 matrix (BioBERT NLI/STS)
    ├── embeddings_pubmedbert_embeddings_768d.npy          # L2-normalized (1033, 768) float32 matrix (PubMedBERT)
    ├── bridge_symptoms_biobert_nli_stsb_public.csv        # Public bridge symptom centrality rankings (BioBERT)
    ├── bridge_symptoms_pubmedbert_embeddings_public.csv   # Public bridge symptom centrality rankings (PubMedBERT)
    ├── consensus_cross_disorder_bridge_pairs_public.csv   # Top 100 cross-disorder pairs by consensus z-score
    ├── embedding_comparison_summary.json                  # Distributional & inter-model correlation metrics
    ├── clustering_evaluation_summary.json                 # HDBSCAN/SVC metrics, chapter overlaps & bridge constructs
    └── figures/
        ├── dsm5tr_interactive_explorer.html               # Standalone interactive 2D symptom manifold explorer
        ├── fig1_inter_model_comparison.svg                # Cross-disorder cosine distributions & 2D log-density
        ├── fig2_chapter_overlap_heatmap.svg               # 20 × 20 DSM-5-TR chapter semantic overlap heatmaps
        ├── fig3_manifold_hdbscan_svc.svg                  # 2D manifold with HDBSCAN clusters & SVC RBF contours
        └── fig4_bridge_symptom_ranking_and_network.svg
```

---

## 3. Methodology & Pipeline Architecture

### 3.1. PDF Extraction & Layout Edge-Case Resolution (`parse.py`)
`parse.py` extracts all Section II (`Diagnostic Criteria`) and Section III (`Proposed Criteria`) blocks across pages 127–1377 of the DSM-5-TR PDF (`1,568` total criteria across `177` unique disorder/subtype entries: `723` `main_criterion` and `845` `sub_symptom`):
* **Multi-Line Title Reconstruction**: Reconstructs wrapped titles across page headers while stripping overarching chapter banners and ICD-10-CM codes (`Fxx.xx`, `Gxx.xx`, `Zxx.xx`).
* **Multi-Box & Subtype Preservation**: Preserves distinct criteria blocks for Bipolar I/II episode types (`Manic`, `Hypomanic`, `Major Depressive Episode`), PTSD age variants (`>6 years` vs. `≤6 years`), Tic disorders, Factitious subtypes, Gender Dysphoria age variants, and General Personality Disorder.
* **Sub-Symptom Granularity**: Captures uppercase main criteria (`A`–`I`), numbered sub-criteria (`1`–`13`), lowercase sub-items (`1a`–`1i`, `2a`–`2c` in ADHD, Autism, Alzheimer's, Lewy Bodies), and roman-numeral behavioral sub-items (`1ai`–`1av` in Frontotemporal Neurocognitive Disorder).
* **Un-Lettered Preamble Recovery**: Captures un-lettered `Criterion A` preambles for single-criterion boxes (`Borderline`, `Histrionic`, `Narcissistic`, `Avoidant`, `Dependent`, and `Obsessive-Compulsive` Personality Disorders, plus Section III `Caffeine Use Disorder` and `Internet Gaming Disorder`).
* **Boundary Hygiene**: Prevents premature termination when lowercase `"diagnostic features"` wraps across lines (e.g., `Major or Mild Neurocognitive Disorder with Lewy Bodies` Criterion C) while terminating cleanly on parenthetical `Coding note (see coding table...)` blocks and Section III narrative transitions.

### 3.2. Clinical NLP Preprocessing & Non-Symptom Filtering (`embed_symptoms.py`)
Because transformer self-attention relies on grammatical structure, negation, and clinical qualifiers, preprocessing **preserves all stopwords and inflected word forms** (no lemmatization or stemming):
1. **Text Standardization**: Converts text to lowercase, collapses whitespace, strips stray PDF page numbers and inline ICD-10 codes, removes parenthetical `(note: ...)` administrative remarks, and strips leading list enumerators (`a. `, `1. `, `2a. `, `i. `) into `embedding_text`.
2. **Non-Symptom Administrative Filtering**: To prevent diagnostic boilerplate from forming artificial transdiagnostic clusters, `classify_criterion()` audits all `1,568` rows (`outputs/criterion_filter_audit_public.csv`) and removes `535` non-symptom criteria across five rule categories, retaining **`1,033` symptom-bearing criteria across `170` disorders**:

| Criterion Classification (`criterion_category`) | Count | Share | Description & Examples |
| :--- | ---: | ---: | :--- |
| **`symptom` (Retained for Embedding)** | **1,033** | **65.9%** | Sub-symptoms and standalone symptom-bearing main criteria (e.g., `Delirium` A & C, `Anorexia Nervosa` A–C) |
| `non_symptom:exclusion_or_etiology` | 225 | 14.3% | Differential diagnosis rule-outs and substance/medical attribution |
| `non_symptom:structural_header` | 151 | 9.6% | Parent list headers introducing sub-items |
| `non_symptom:distress_or_impairment` | 102 | 6.5% | Clinical significance / occupational impairment boilerplate |
| `non_symptom:onset_duration_age` | 41 | 2.6% | Standalone chronological/duration thresholds |
| `non_symptom:structural_prereq` | 16 | 1.0% | Cross-box prerequisites |

### 3.3. Dual-Model 768-Dimensional Vectorization (`embed_symptoms.py`)
Each of the `1,033` symptom texts is encoded into an L2-normalized `768`-dimensional vector (`normalize_embeddings=True`, where dot product equals cosine similarity) using two contrastively trained biomedical SentenceTransformer checkpoints:
* **Model 1 (`biobert_nli_stsb`)**: [`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`](https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb) — BioBERT backbone fine-tuned with `CosineSimilarityLoss` across `SNLI`, `MNLI`, `SciNLI`, `SciTail`, `MedNLI`, and `STS-B` (`max_seq_length` expanded from `100` to `256`).
* **Model 2 (`pubmedbert_embeddings`)**: [`NeuML/pubmedbert-base-embeddings`](https://huggingface.co/NeuML/pubmedbert-base-embeddings) — Uncased BiomedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) fine-tuned with `MultipleNegativesRankingLoss` (`max_seq_length=512`).

### 3.4. Manifold Projection, HDBSCAN & Support Vector Clustering (`visualize_results.py`)
* **k-NN Spectral/Attraction-Repulsion Manifold**: Constructs a symmetric `k=15` nearest-neighbor cosine graph from the `1,033 × 1,033` similarity matrix, initializes 2D coordinates via orthogonalized power iteration on the normalized graph Laplacian, and refines coordinates via UMAP-style attractive/repulsive stochastic gradient descent.
* **HDBSCAN Mutual-Reachability Condensation**: Computes core distances ($k_{\text{core}}=7$) and pairwise mutual-reachability distances $d_{\text{mreach}}(i,j) = \max\{\text{core}(i), \text{core}(j), d(i,j)\}$, constructs the Minimum Spanning Tree (MST), removes high-reachability noise edges, and recursively condenses oversized components into stable symptom clusters ($n_{\min}=16$).
* **Support Vector Clustering (SVC) RBF Contours**: Fits a Gaussian radial basis function (RBF) kernel density support function $K(x, x_i) = \exp(-\gamma \|x - x_i\|^2)$ ($\gamma = 0.45$) over each cluster's support vectors to trace closed non-linear decision boundaries in 2D space.

---

## 4. Key Empirical Findings

### 4.1. Inter-Model Comparison (`528,672` Cross-Disorder Pairs)

| Metric | [`BioBERT NLI/STS`](https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb) | [`PubMedBERT Embeddings`](https://huggingface.co/NeuML/pubmedbert-base-embeddings) |
| :--- | :--- | :--- |
| **Cross-Disorder Cosine Mean ± SD** | `0.1972 ± 0.1276` | `0.2435 ± 0.1308` |
| **Inter-Model Agreement (528,672 pairs)** | `Pearson r = 0.5727` | `Spearman ρ = 0.5070` |
| **HDBSCAN Condensed Clusters** | `7` clusters | `11` clusters |
| **Clustered Symptoms (`%`)** | `524 / 1,033` (`50.7%`) | `532 / 1,033` (`51.5%`) |
| **Unclustered Interstitial / Bridge Symptoms (`%`)** | `509 / 1,033` (`49.3%`) | `501 / 1,033` (`48.5%`) |
| **2D Cluster Silhouette Score** | `0.3902` | `0.3659` |

### 4.2. Top Cross-Chapter Semantic Overlaps (Mean Top-10 Cross-Disorder Cosine)

| DSM-5-TR Chapter 1 | DSM-5-TR Chapter 2 | BioBERT Top-10 Mean | PubMedBERT Top-10 Mean | Consensus Mean |
| :--- | :--- | ---: | ---: | ---: |
| **Bipolar & Related** | **Depressive** | `1.0000` | `1.0000` | `1.0000` |
| **Bipolar & Related** | **Section III (Emerging)** | `1.0000` | `1.0000` | `1.0000` |
| **Depressive** | **Section III (Emerging)** | `0.9902` | `0.9808` | `0.9855` |
| **Substance & Addictive** | **Section III (Emerging)** | `0.8267` | `0.9081` | `0.8674` |
| **Depressive** | **Substance & Addictive** | `0.8828` | `0.8437` | `0.8632` |
| **Anxiety** | **Substance & Addictive** | `0.8435` | `0.7513` | `0.7974` |
| **Depressive** | **Sleep-Wake** | `0.8029` | `0.7831` | `0.7930` |
| **Bipolar & Related** | **Substance & Addictive** | `0.8284` | `0.7420` | `0.7852` |
| **Bipolar & Related** | **Sleep-Wake** | `0.7805` | `0.7867` | `0.7836` |
| **Anxiety** | **Sleep-Wake** | `0.8135` | `0.7526` | `0.7831` |
| **Anxiety** | **Trauma & Stressor** | `0.7844` | `0.7780` | `0.7812` |

### 4.3. Primary Transdiagnostic Bridge Symptom Constructs

* **Sleep-Wake Dyssomnia Hub**: Bridges **`12` disorders across `4` chapters** (`Depressive`, `Bipolar & Related`, `Substance & Addictive`, `Section III`) with mean cross-disorder cosine similarity of `0.8968–0.9510` (BioBERT) and `0.8389–0.9027` (PubMedBERT).
* **Sleep-Wake Continuity Disturbance**: Bridges **`5` disorders across `3` chapters** (`Anxiety`, `Trauma & Stressor`, `Substance & Addictive`; BioBERT `0.9476`, PubMedBERT `0.9314`).
* **Affective Elevation / Irritable Mood Disturbance**: Links **`8` disorders across `3` chapters** (`Bipolar & Related`, `Depressive`, `Section III`; BioBERT `0.8193`, PubMedBERT `0.8500`).
* **Externalizing Irritability & Aggression**: Bridges **`4` disorders across `3` chapters** (`Substance & Addictive`, `Depressive`, `Personality`; BioBERT `0.8331`, PubMedBERT `0.8418`).
* **Compulsive Use & Craving Axis**: Impaired control and craving bridge **`11` disorders across `Substance & Addictive` and `Section III`** (`Caffeine Use Disorder`, `Internet Gaming Disorder`), with higher cross-substance semantic invariance in PubMedBERT (`0.8898–0.8956`) than BioBERT (`0.8155–0.8231`).

---

## 5. Reproducibility & Quickstart Instructions

### Prerequisites
* **Python 3.12** (recommended on macOS Apple Silicon so `torch`, `umap-learn`/`numba`, and `hdbscan` install via prebuilt binary wheels).
* [`uv`](https://docs.astral.sh/uv/) or `pip` for environment management.

### Running Downstream Analyses from Public Embeddings (No PDF Required)
```bash
uv venv --python 3.12 --clear .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python visualize_results.py
```

### Full Local Rehydration from Licensed DSM-5-TR PDF
```bash
# Place a licensed copy of "DSM 5 TR-APA (2022).pdf" in the root directory, then run:
.venv/bin/python parse.py
.venv/bin/python check.py
.venv/bin/python embed_symptoms.py
.venv/bin/python visualize_results.py
```
