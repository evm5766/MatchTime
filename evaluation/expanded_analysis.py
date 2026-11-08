import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from sentence_transformers import SentenceTransformer

# Reuse the custom BLEU implementation from the original script
from score_single import Bleu

# Config

CSV_PATH = "./inference_result/sample.csv"
OUTPUT_DIR = "./evaluation/expanded_results"
MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data

df = pd.read_csv(CSV_PATH)

required_columns = ["anonymized", "predicted_res_0"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df.dropna(subset=required_columns).copy()

df["anonymized"] = df["anonymized"].astype(str).str.strip()
df["predicted_res_0"] = df["predicted_res_0"].astype(str).str.strip()

df = df[
    (df["anonymized"] != "") &
    (df["predicted_res_0"] != "")
].reset_index(drop=True)

references = {
    i: [text] for i, text in enumerate(df["anonymized"])
}

hypotheses = {
    i: [text] for i, text in enumerate(df["predicted_res_0"])
}

print(f"Usable commentary pairs: {len(df)}")

# Calculate per-commentary evaluation metrics

print("\nCalculating BLEU...")
bleu_scorer = Bleu(4)
_, bleu_scores = bleu_scorer.compute_score(references, hypotheses)

bleu1 = np.asarray(bleu_scores[0]).flatten() * 100
bleu4 = np.asarray(bleu_scores[3]).flatten() * 100

print("Calculating METEOR...")
meteor_scorer = Meteor()
_, meteor_scores = meteor_scorer.compute_score(references, hypotheses)

meteor_scores = np.asarray(meteor_scores).flatten() * 100

print("Calculating ROUGE-L...")
rouge_scorer = Rouge()
_, rouge_scores = rouge_scorer.compute_score(references, hypotheses)

rouge_scores = np.asarray(rouge_scores).flatten() * 100

print("Calculating CIDEr...")
cider_scorer = Cider()
_, cider_scores = cider_scorer.compute_score(references, hypotheses)

cider_scores = np.asarray(cider_scores).flatten() * 100

print("Calculating sBERT semantic similarity...")
sbert_model = SentenceTransformer(MODEL_NAME, device="cpu")

reference_text = df["anonymized"].tolist()
generated_text = df["predicted_res_0"].tolist()

reference_embeddings = sbert_model.encode(
    reference_text,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=32,
    show_progress_bar=True
)

generated_embeddings = sbert_model.encode(
    generated_text,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=32,
    show_progress_bar=True
)

# Corresponding reference/prediction cosine similarity
sbert_scores = np.sum(
    reference_embeddings * generated_embeddings,
    axis=1
) * 100

# Save scores

df["BLEU-1"] = bleu1
df["BLEU-4"] = bleu4
df["METEOR"] = meteor_scores
df["ROUGE-L"] = rouge_scores
df["CIDEr"] = cider_scores
df["sBERT"] = sbert_scores

# Save all original columns and the six metric scores
individual_path = os.path.join(
    OUTPUT_DIR, "individual_commentary_scores.csv"
)
df.to_csv(individual_path, index=False)

# Correlation analysis

metric_columns = [
    "BLEU-1",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "CIDEr",
    "sBERT"
]

pearson_corr = df[metric_columns].corr(method="pearson")
spearman_corr = df[metric_columns].corr(method="spearman")

pearson_corr.to_csv(
    os.path.join(OUTPUT_DIR, "pearson_correlations.csv")
)

spearman_corr.to_csv(
    os.path.join(OUTPUT_DIR, "spearman_correlations.csv")
)

print("\nPearson correlation:")
print(pearson_corr.round(3))

print("\nSpearman correlation:")
print(spearman_corr.round(3))

# Compare the metric rankings

lexical_columns = [
    "BLEU-1",
    "BLEU-4",
    "METEOR",
    "ROUGE-L",
    "CIDEr"
]

# Average percentile rank across metrics
lexical_ranks = pd.DataFrame({
    col: df[col].rank(pct=True)
    for col in lexical_columns
})

df["lexical_rank"] = lexical_ranks.mean(axis=1)
df["sBERT_rank"] = df["sBERT"].rank(pct=True)

# Positive: sBERT ranks the caption higher than lexical metrics
# Negative: lexical metrics rank the caption higher than sBERT
df["rank_difference"] = (
    df["sBERT_rank"] - df["lexical_rank"]
)

comparison_columns = [
    col for col in [
        "league",
        "game",
        "half",
        "timestamp",
        "type",
        "anonymized",
        "predicted_res_0"
    ] if col in df.columns
]

comparison_columns += metric_columns + [
    "lexical_rank",
    "sBERT_rank",
    "rank_difference"
]

# Ten largest disagreements in each direction
semantic_higher = df.nlargest(
    10, "rank_difference"
)[comparison_columns]

lexical_higher = df.nsmallest(
    10, "rank_difference"
)[comparison_columns]

semantic_higher.to_csv(
    os.path.join(OUTPUT_DIR, "semantic_higher_examples.csv"),
    index=False
)

lexical_higher.to_csv(
    os.path.join(OUTPUT_DIR, "lexical_higher_examples.csv"),
    index=False
)

# Commentary length analysis

df["reference_word_count"] = (
    df["anonymized"].str.split().str.len()
)

df["generated_word_count"] = (
    df["predicted_res_0"].str.split().str.len()
)

df["word_count_difference"] = (
    df["generated_word_count"] - df["reference_word_count"]
)

length_columns = [
    "reference_word_count",
    "generated_word_count",
    "word_count_difference"
] + metric_columns

length_corr = df[length_columns].corr(method="spearman")

length_corr.to_csv(
    os.path.join(OUTPUT_DIR, "length_metric_correlations.csv")
)

# Game-level descriptive summary

if "game" in df.columns:
    game_summary = (
        df.groupby("game")[metric_columns]
        .agg(["mean", "median", "count"])
    )

    game_summary.to_csv(
        os.path.join(OUTPUT_DIR, "game_level_summary.csv")
    )

# Visualizations

# Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(
    spearman_corr.values,
    vmin=-1,
    vmax=1,
    aspect="auto"
)

ax.set_xticks(range(len(metric_columns)))
ax.set_yticks(range(len(metric_columns)))

ax.set_xticklabels(metric_columns, rotation=45, ha="right")
ax.set_yticklabels(metric_columns)

for i in range(len(metric_columns)):
    for j in range(len(metric_columns)):
        ax.text(
            j, i,
            f"{spearman_corr.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

ax.set_title("Spearman Correlation Between Evaluation Metrics")
fig.colorbar(image, ax=ax, label="Correlation")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
    dpi=300
)
plt.close()


# Lexical vs semantic ranking comparison
fig, ax = plt.subplots(figsize=(8, 7))

ax.scatter(
    df["lexical_rank"],
    df["sBERT_rank"],
    alpha=0.6
)

ax.plot([0, 1], [0, 1], linestyle="--")

ax.set_xlabel("Average Lexical Metric Percentile Rank")
ax.set_ylabel("sBERT Percentile Rank")
ax.set_title("Lexical vs Semantic Evaluation Rankings")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "lexical_vs_semantic_ranks.png"),
    dpi=300
)
plt.close()