from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Project setup
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]

outputs_path = project_root / "outputs"

evaluation_file = (
    outputs_path
    / "manual_evaluation_sample.csv"
)


# ---------------------------------------------------------
# 2. Load the manually reviewed file
# ---------------------------------------------------------

df = pd.read_csv(evaluation_file)

print(
    f"\nLoaded {len(df)} responses "
    "for AI evaluation."
)


# ---------------------------------------------------------
# 3. Check that manual labels are completed
# ---------------------------------------------------------

manual_columns = [
    "manual_theme",
    "manual_sentiment",
    "manual_urgency"
]

missing_columns = [
    column
    for column in manual_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "These manual columns are missing: "
        f"{missing_columns}"
    )


# Treat empty text as missing values
for column in manual_columns:

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )


if df[manual_columns].isna().any().any():

    raise ValueError(
        "Some manual labels are still blank. "
        "Complete all manual label columns "
        "before running this file."
    )


# ---------------------------------------------------------
# 4. Clean labels before comparing
# ---------------------------------------------------------

comparison_columns = [
    "theme",
    "sentiment",
    "urgency",
    "manual_theme",
    "manual_sentiment",
    "manual_urgency"
]

for column in comparison_columns:

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )


# ---------------------------------------------------------
# 5. Compare Gemini labels with manual labels
# ---------------------------------------------------------

df["theme_match"] = (
    df["theme"]
    == df["manual_theme"]
)

df["sentiment_match"] = (
    df["sentiment"]
    == df["manual_sentiment"]
)

df["urgency_match"] = (
    df["urgency"]
    == df["manual_urgency"]
)

df["all_labels_match"] = (
    df["theme_match"]
    & df["sentiment_match"]
    & df["urgency_match"]
)


# ---------------------------------------------------------
# 6. Calculate agreement percentages
# ---------------------------------------------------------

theme_agreement = (
    df["theme_match"]
    .mean()
    * 100
)

sentiment_agreement = (
    df["sentiment_match"]
    .mean()
    * 100
)

urgency_agreement = (
    df["urgency_match"]
    .mean()
    * 100
)

complete_agreement = (
    df["all_labels_match"]
    .mean()
    * 100
)


# ---------------------------------------------------------
# 7. Print evaluation results
# ---------------------------------------------------------

print(
    "\nAI Evaluation Results"
)

print(
    "---------------------"
)

print(
    f"\nTheme agreement: "
    f"{theme_agreement:.1f}%"
)

print(
    f"Sentiment agreement: "
    f"{sentiment_agreement:.1f}%"
)

print(
    f"Urgency agreement: "
    f"{urgency_agreement:.1f}%"
)

print(
    f"Complete label agreement: "
    f"{complete_agreement:.1f}%"
)


# ---------------------------------------------------------
# 8. Show disagreements
# ---------------------------------------------------------

disagreements = df[
    ~df["all_labels_match"]
].copy()

print(
    "\nResponses with at least "
    f"one disagreement: "
    f"{len(disagreements)}"
)

if len(disagreements) > 0:

    columns_to_show = [
        "response_id",
        "response_text",
        "manual_theme",
        "theme",
        "manual_sentiment",
        "sentiment",
        "manual_urgency",
        "urgency"
    ]

    print(
        "\nDisagreements:"
    )

    print(
        disagreements[
            columns_to_show
        ]
        .to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# 9. Create agreement chart
# ---------------------------------------------------------

agreement_scores = pd.Series(
    {
        "Theme": theme_agreement,
        "Sentiment": sentiment_agreement,
        "Urgency": urgency_agreement,
        "Complete Match": complete_agreement
    }
)

plt.figure(
    figsize=(8, 5)
)

agreement_scores.plot(
    kind="bar"
)

plt.title(
    "Agreement Between Manual and AI Labels"
)

plt.xlabel(
    "Classification Category"
)

plt.ylabel(
    "Agreement Percentage"
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

chart_file = (
    outputs_path
    / "ai_evaluation_agreement.png"
)

plt.savefig(
    chart_file
)

plt.show()


# ---------------------------------------------------------
# 10. Save detailed results
# ---------------------------------------------------------

results_file = (
    outputs_path
    / "ai_evaluation_results.csv"
)

df.to_csv(
    results_file,
    index=False
)


# ---------------------------------------------------------
# 11. Save evaluation summary
# ---------------------------------------------------------

summary = f"""
AI Classification Evaluation

Number of manually reviewed responses:
{len(df)}

Agreement Scores:
- Theme agreement: {theme_agreement:.1f}%
- Sentiment agreement: {sentiment_agreement:.1f}%
- Urgency agreement: {urgency_agreement:.1f}%
- Complete label agreement: {complete_agreement:.1f}%

Responses with at least one disagreement:
{len(disagreements)}

Interpretation:
These results compare Gemini-generated classifications
with manually assigned labels.

Any disagreements can be reviewed to identify unclear
categories and improve the AI classification prompt.
"""

summary_file = (
    outputs_path
    / "ai_evaluation_summary.txt"
)

with open(
    summary_file,
    "w"
) as file:

    file.write(
        summary
    )


# ---------------------------------------------------------
# 12. Final confirmation
# ---------------------------------------------------------

print(
    "\nEvaluation files saved."
)

print(
    f"\nDetailed results:\n"
    f"{results_file}"
)

print(
    f"\nSummary:\n"
    f"{summary_file}"
)

print(
    f"\nChart:\n"
    f"{chart_file}"
)