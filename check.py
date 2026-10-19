"""
DSM-5-TR Dataset Verification & Integrity Audit Script
======================================================
Audits `dsm_symptoms_cleaned_full.json` for schema completeness, second-half disorder
continuity (Gender Dysphoria through Neurocognitive, Personality, Paraphilic, and
Section III conditions), and verifies zero coding-note leaks, missing Criterion A
preambles, or mid-sentence truncations.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def run_audit(json_path: Path = Path("dsm_symptoms_cleaned_full.json")) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    disorders = []
    for r in records:
        if r["disorder"] not in disorders:
            disorders.append(r["disorder"])

    level_counts = Counter(r["criterion_level"] for r in records)
    print(f"Total rows extracted: {len(records)}")
    print(f"Total unique disorders: {len(disorders)}")
    print(f"Criterion levels: {dict(level_counts)}\n")

    # 1. Print all unique disorder titles from Gender Dysphoria onward
    anchor = "Substance/Medication-Induced Sexual Dysfunction"
    start_idx = disorders.index(anchor) if anchor in disorders else len(disorders) // 2
    print("--- Disorders in the Second Half of the JSON ---")
    for d in disorders[start_idx + 1 :]:
        print(f"- {d}")

    # 2. Check Alzheimer's Disease entries specifically
    print("\n--- Alzheimer's Disease Criteria Check ---")
    for r in records:
        if "alzheimer" in r["disorder"].lower():
            print(f"{r['disorder']} | {r['criterion_id']} ({r['criterion_level']}): {r['raw_text'][:100]}...")

    # 3. Structural integrity assertions
    coding_leaks = [r for r in records if "coding note" in r["cleaned_text"]]
    missing_a = [
        r["disorder"]
        for i, r in enumerate(records)
        if (i == 0 or records[i - 1]["disorder"] != r["disorder"]) and r["criterion_id"] != "A"
    ]
    truncations = [
        (r["disorder"], r["criterion_id"])
        for r in records
        if not re.search(r"[\.:\)\"”\]\?]$", r["raw_text"].strip())
        and not r["raw_text"].strip().endswith("; and")
    ]

    print("\n--- Structural Integrity Checks ---")
    print(f"Remaining Coding note leaks: {len(coding_leaks)}")
    print(f"Disorders not starting with Criterion A: {len(missing_a)}")
    print(f"Mid-sentence truncations: {len(truncations)}")

    assert len(coding_leaks) == 0, f"Found Coding note leaks: {coding_leaks}"
    assert len(missing_a) == 0, f"Found disorders missing Criterion A: {missing_a}"
    assert len(truncations) == 0, f"Found truncated criteria: {truncations}"
    print("All integrity assertions passed.")


if __name__ == "__main__":
    run_audit()
