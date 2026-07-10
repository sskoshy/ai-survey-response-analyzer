from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Project setup
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]

outputs_path = project_root / "outputs"

evaluation_file = (
    outputs_path
    / "ai_evaluation_results.csv"
)


# ---------------------------------------------------------
# 2. Load evaluation results
# ---------------------------------------------------------

df = pd.read_csv(
    evaluation_file
)


# ---------------------------------------------------------
# 3. Keep only rows with disagreements
# ---------------------------------------------------------

disagreements = df[
    ~df["all_labels_match"]
].copy()


# ---------------------------------------------------------
# 4. Select useful comparison columns
# ---------------------------------------------------------

columns_to_keep = [
    "response_id",
    "response_text",
    "manual_theme",
    "theme",
    "theme_match",
    "manual_sentiment",
    "sentiment",
    "sentiment_match",
    "manual_urgency",
    "urgency",
    "urgency_match",
    "explanation"
]

disagreements = disagreements[
    columns_to_keep
]


# ---------------------------------------------------------
# 5. Save disagreement review file
# ---------------------------------------------------------

output_file = (
    outputs_path
    / "ai_disagreement_review.csv"
)

disagreements.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# 6. Print summary
# ---------------------------------------------------------

print(
    "\nAI Disagreement Review"
)

print(
    "----------------------"
)

print(
    f"\nTotal evaluated responses: "
    f"{len(df)}"
)

print(
    f"Responses with disagreements: "
    f"{len(disagreements)}"
)

print(
    "\nTheme disagreements:"
)

print(
    (
        ~df["theme_match"]
    ).sum()
)

print(
    "\nSentiment disagreements:"
)

print(
    (
        ~df["sentiment_match"]
    ).sum()
)

print(
    "\nUrgency disagreements:"
)

print(
    (
        ~df["urgency_match"]
    ).sum()
)

print(
    f"\nReview file saved to:\n"
    f"{output_file}"
)