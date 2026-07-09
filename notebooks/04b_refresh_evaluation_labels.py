from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Project setup
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]

outputs_path = project_root / "outputs"

manual_file = (
    outputs_path
    / "manual_evaluation_sample.csv"
)

ai_file = (
    outputs_path
    / "ai_classified_responses_all.csv"
)


# ---------------------------------------------------------
# 2. Load both files
# ---------------------------------------------------------

manual_df = pd.read_csv(
    manual_file
)

ai_df = pd.read_csv(
    ai_file
)


# ---------------------------------------------------------
# 3. Check the current AI results for failures
# ---------------------------------------------------------

failure_pattern = (
    "Classification failed"
    "|RESOURCE_EXHAUSTED"
    "|quota"
)

failed_rows = ai_df[
    ai_df["explanation"]
    .astype(str)
    .str.contains(
        failure_pattern,
        case=False,
        na=False
    )
]


if len(failed_rows) > 0:

    raise ValueError(
        f"The full AI results still contain "
        f"{len(failed_rows)} failed classifications. "
        "Rerun 02_ai_classification.py before continuing."
    )


# ---------------------------------------------------------
# 4. Keep the newest AI classifications
# ---------------------------------------------------------

new_ai_labels = ai_df[
    [
        "response_id",
        "theme",
        "sentiment",
        "urgency",
        "explanation"
    ]
].copy()


# ---------------------------------------------------------
# 5. Remove the old AI columns
# ---------------------------------------------------------

manual_df = manual_df.drop(
    columns=[
        "theme",
        "sentiment",
        "urgency",
        "explanation"
    ],
    errors="ignore"
)


# ---------------------------------------------------------
# 6. Add the corrected Gemini labels
# ---------------------------------------------------------

updated_df = manual_df.merge(
    new_ai_labels,
    on="response_id",
    how="left",
    validate="one_to_one"
)


# ---------------------------------------------------------
# 7. Check that every row received new AI labels
# ---------------------------------------------------------

ai_columns = [
    "theme",
    "sentiment",
    "urgency",
    "explanation"
]

if updated_df[
    ai_columns
].isna().any().any():

    raise ValueError(
        "At least one evaluation response "
        "did not receive updated AI labels."
    )


# ---------------------------------------------------------
# 8. Reorder the columns
# ---------------------------------------------------------

updated_df = updated_df[
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
# 9. Save the corrected evaluation file
# ---------------------------------------------------------

updated_df.to_csv(
    manual_file,
    index=False
)


print(
    "\nEvaluation file successfully refreshed."
)

print(
    f"\nUpdated responses: "
    f"{len(updated_df)}"
)

print(
    "\nYour manual labels were preserved."
)

print(
    "\nThe AI columns now contain the "
    "latest Gemini classifications."
)

print(
    f"\nSaved to:\n"
    f"{manual_file}"
)