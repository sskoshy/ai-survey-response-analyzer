from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Project setup
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]
outputs_path = project_root / "outputs"

classified_file = outputs_path / "ai_classified_responses_all.csv"

# Load all AI-classified responses
df = pd.read_csv(classified_file)


# ---------------------------------------------------------
# 2. Randomly select 10 responses
# ---------------------------------------------------------

evaluation_sample = df.sample(
    n=10,
    random_state=42
).copy()


# ---------------------------------------------------------
# 3. Add blank columns for manual labels
# ---------------------------------------------------------

evaluation_sample["manual_theme"] = ""
evaluation_sample["manual_sentiment"] = ""
evaluation_sample["manual_urgency"] = ""


# Put the columns in an easy-to-review order
evaluation_sample = evaluation_sample[
    [
        "response_id",
        "response_text",
        "manual_theme",
        "manual_sentiment",
        "manual_urgency",
        "theme",
        "sentiment",
        "urgency",
        "explanation"
    ]
]


# ---------------------------------------------------------
# 4. Save evaluation sample
# ---------------------------------------------------------

output_file = outputs_path / "manual_evaluation_sample.csv"

evaluation_sample.to_csv(output_file, index=False)

print("\nManual evaluation sample created.")
print(f"Saved to: {output_file}")

print(
    "\nOpen the CSV and fill in the three manual label columns "
    "before comparing them with the AI labels."
)