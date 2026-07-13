# ============================================================
# Evaluate generated reports against ground-truth reports
# using BLEU and ROUGE.
#
# NOTE: these are text-overlap metrics, not clinical-factuality
# metrics. They are a Week-1 sanity check only ("did the pipeline
# round-trip end to end") -- NOT the headline metric for this
# project. RadGraph-F1 / CheXbert-F1 / GREEN come later.
#
# Setup:
#   !pip install -q rouge-score nltk
# ============================================================

import csv
import json

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

GENERATED_PATH = "final.json"          # one JSON object per line: {"uid":..., "report":...}
GROUND_TRUTH_PATH = "data/indiana_reports.csv"  # has a 'uid' column + findings/impression columns
OUTPUT_PATH = "eval_results.csv"

smoothing = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def load_generated(path):
    """uid -> generated report text"""
    reports = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            reports[entry["uid"]] = entry["report"]
    return reports


def load_ground_truth(path):
    """
    uid -> reference report text.
    indiana_reports.csv typically has 'findings' and 'impression' columns --
    concatenate them to match what generate_report() produces (a full report).
    Adjust the column names below if your CSV header differs.
    """
    refs = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            findings = (row.get("findings") or "").strip()
            impression = (row.get("impression") or "").strip()
            text = (findings + " " + impression).strip()
            if text:
                refs[row["uid"]] = text
    return refs


def compute_scores(generated, reference):
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    gen_tokens = nltk.word_tokenize(generated.lower())

    bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothing)
    rouge_scores = rouge.score(reference, generated)

    return {
        "bleu": bleu,
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rouge2": rouge_scores["rouge2"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
    }


def main():
    generated = load_generated(GENERATED_PATH)
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    rows = []
    missing = 0

    for uid, gen_text in generated.items():
        ref_text = ground_truth.get(uid)
        if ref_text is None:
            missing += 1
            continue
        scores = compute_scores(gen_text, ref_text)
        rows.append({"uid": uid, **scores})

    if not rows:
        print("No matching uids between generated and ground-truth reports. "
              "Check that uid formats match and column names in load_ground_truth() are correct.")
        return

    # Write per-study scores
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "bleu", "rouge1", "rouge2", "rougeL"])
        writer.writeheader()
        writer.writerows(rows)

    # Print averages
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in ["bleu", "rouge1", "rouge2", "rougeL"]}
    print(f"Evaluated {len(rows)} studies ({missing} skipped, no matching ground truth).")
    print("Average scores:")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nPer-study scores written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()