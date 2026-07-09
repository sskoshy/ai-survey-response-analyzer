from pathlib import Path
from typing import Literal
import json
import os

import pandas as pd
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel


# ---------------------------------------------------------
# 1. Define the allowed output categories
# ---------------------------------------------------------

Theme = Literal[
    "Time Management",
    "Exam Stress",
    "Study Strategy",
    "Course Resources",
    "Scheduling",
    "Motivation",
    "Other"
]

Sentiment = Literal[
    "Positive",
    "Neutral",
    "Negative",
    "Mixed"
]

Urgency = Literal[
    "Low",
    "Medium",
    "High"
]


# ---------------------------------------------------------
# 2. Define the required AI output structure
# ---------------------------------------------------------

class SurveyClassification(BaseModel):
    response_id: int
    theme: Theme
    sentiment: Sentiment
    urgency: Urgency
    explanation: str


class ClassificationBatch(BaseModel):
    classifications: list[SurveyClassification]


# ---------------------------------------------------------
# 3. Project setup
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]

load_dotenv(
    project_root / ".env"
)

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Check your .env file."
    )

client = genai.Client(
    api_key=api_key
)


# ---------------------------------------------------------
# 4. Load the survey data
# ---------------------------------------------------------

csv_path = (
    project_root
    / "data"
    / "sample_survey.csv"
)

df = pd.read_csv(
    csv_path
)

outputs_path = (
    project_root
    / "outputs"
)

outputs_path.mkdir(
    exist_ok=True
)


# ---------------------------------------------------------
# 5. Prepare all responses for one request
# ---------------------------------------------------------

survey_responses = (
    df[
        [
            "response_id",
            "response_text"
        ]
    ]
    .to_dict(
        orient="records"
    )
)

survey_text = json.dumps(
    survey_responses,
    indent=2
)


# ---------------------------------------------------------
# 6. Create the classification prompt
# ---------------------------------------------------------

prompt = f"""
You are analyzing open-ended responses from a student
study experience survey.

Classify every response exactly once.

Use only the categories defined below.

THEMES

Time Management:
Planning, prioritizing tasks, procrastination,
deadlines, workload balance, or managing time.

Exam Stress:
Anxiety, stress, pressure, burnout, or feeling
overwhelmed because of exams or grades.

Study Strategy:
Practice methods, note-taking, remembering concepts,
learning techniques, understanding material,
or study habits.

Course Resources:
Study guides, practice exams, examples, rubrics,
office hours, review sessions, reminders,
or instructor support.

Scheduling:
Difficulty coordinating times, availability,
or conflicting schedules.

Motivation:
Focus, confidence, accountability, encouragement,
or motivation from studying with other people.

Other:
Use only when no listed category clearly applies.

SENTIMENT

Positive:
The response mainly describes something helpful,
successful, encouraging, or beneficial.

Neutral:
The response is mainly a suggestion, observation,
or request without a strong emotional tone.

Negative:
The response mainly describes stress, difficulty,
frustration, anxiety, burnout, or dissatisfaction.

Mixed:
The response contains both meaningful positive
and negative elements.

URGENCY

Low:
A positive observation, preference,
or ordinary suggestion.

Medium:
A meaningful difficulty affecting the student,
but without severe distress.

High:
Strong anxiety, burnout, major distress,
or feeling significantly overwhelmed.

Return one classification for every response.

Survey responses:

{survey_text}
"""


# ---------------------------------------------------------
# 7. Classify all responses in ONE Gemini request
# ---------------------------------------------------------

print(
    "\nSending all survey responses "
    "to Gemini in one request..."
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_schema": ClassificationBatch
    }
)


# ---------------------------------------------------------
# 8. Validate the structured AI response
# ---------------------------------------------------------

if response.parsed is not None:

    classification_batch = (
        response.parsed
    )

else:

    classification_batch = (
        ClassificationBatch
        .model_validate_json(
            response.text
        )
    )


classification_records = [

    result.model_dump()

    for result

    in classification_batch.classifications

]


results_df = pd.DataFrame(
    classification_records
)


# ---------------------------------------------------------
# 9. Verify every response was classified
# ---------------------------------------------------------

expected_ids = set(
    df["response_id"]
)

returned_ids = set(
    results_df["response_id"]
)


missing_ids = (
    expected_ids
    - returned_ids
)

unexpected_ids = (
    returned_ids
    - expected_ids
)


if missing_ids:

    raise ValueError(
        "Gemini did not classify "
        "these response IDs: "
        f"{sorted(missing_ids)}"
    )


if unexpected_ids:

    raise ValueError(
        "Gemini returned unexpected "
        "response IDs: "
        f"{sorted(unexpected_ids)}"
    )


if len(results_df) != len(df):

    raise ValueError(
        "The number of AI results "
        "does not match the number "
        "of survey responses."
    )


# ---------------------------------------------------------
# 10. Merge AI results with the original data
# ---------------------------------------------------------

classified_df = df.merge(
    results_df,
    on="response_id",
    how="left",
    validate="one_to_one"
)


# ---------------------------------------------------------
# 11. Save the completed results
# ---------------------------------------------------------

output_file = (
    outputs_path
    / "ai_classified_responses_all.csv"
)

classified_df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# 12. Print a summary
# ---------------------------------------------------------

print(
    "\nAI classification complete."
)

print(
    f"\nResponses classified: "
    f"{len(classified_df)}"
)

print(
    "\nTheme counts:"
)

print(
    classified_df[
        "theme"
    ]
    .value_counts()
)

print(
    f"\nResults saved to:\n"
    f"{output_file}"
)