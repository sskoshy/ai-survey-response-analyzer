from pathlib import Path
from typing import Literal
import json
import os
import random
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Survey Response Analyzer",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# 2. PROJECT SETTINGS
# =========================================================

project_root = Path(__file__).resolve().parent

sample_file = (
    project_root
    / "data"
    / "sample_survey.csv"
)

load_dotenv(
    project_root
    / ".env"
)

# Protect the Gemini API quota by limiting
# the number of responses analyzed at once.
MAX_RESPONSES = 50


# =========================================================
# 3. ALLOWED AI CLASSIFICATIONS
# =========================================================

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


# =========================================================
# 4. GEMINI STRUCTURED OUTPUT
# =========================================================

class SurveyClassification(BaseModel):
    """
    Required classification for one survey response.
    """

    response_id: int

    theme: Theme

    sentiment: Sentiment

    urgency: Urgency

    explanation: str


class ClassificationBatch(BaseModel):
    """
    Required structure for the complete Gemini response.
    """

    classifications: list[
        SurveyClassification
    ]


# =========================================================
# 5. LOAD GEMINI API KEY
# =========================================================

def get_api_key():
    """
    Loads the Gemini API key.

    Local app:
    Reads the key from the .env file.

    Deployed app:
    Reads the key from Streamlit secrets.
    """

    local_api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if local_api_key:

        return local_api_key


    try:

        cloud_api_key = (

            st.secrets[
                "GEMINI_API_KEY"
            ]

        )

        return cloud_api_key


    except Exception:

        return None


# =========================================================
# 6. PREPARE SURVEY DATA
# =========================================================

def prepare_data(
    dataframe
):
    """
    Validates and cleans uploaded survey data.

    The uploaded CSV must include a column
    named response_text.
    """

    df = dataframe.copy()


    # -----------------------------------------------------
    # Check for the required response column
    # -----------------------------------------------------

    if (
        "response_text"
        not in df.columns
    ):

        raise ValueError(

            "The CSV must contain a column "
            "named 'response_text'."

        )


    # -----------------------------------------------------
    # Remove missing responses
    # -----------------------------------------------------

    df = df.dropna(

        subset=[
            "response_text"
        ]

    )


    # -----------------------------------------------------
    # Clean response text
    # -----------------------------------------------------

    df[
        "response_text"
    ] = (

        df[
            "response_text"
        ]

        .astype(
            str
        )

        .str.strip()

    )


    # -----------------------------------------------------
    # Remove empty responses
    # -----------------------------------------------------

    df = (

        df[

            df[
                "response_text"
            ]

            != ""

        ]

        .copy()

    )


    # -----------------------------------------------------
    # Reset row numbers
    # -----------------------------------------------------

    df = df.reset_index(

        drop=True

    )


    # -----------------------------------------------------
    # Check that usable responses remain
    # -----------------------------------------------------

    if len(df) == 0:

        raise ValueError(

            "No usable survey responses "
            "were found."

        )


    # -----------------------------------------------------
    # Limit API usage
    # -----------------------------------------------------

    if (
        len(df)
        > MAX_RESPONSES
    ):

        raise ValueError(

            f"Please upload no more than "
            f"{MAX_RESPONSES} survey responses "
            "at one time."

        )


    # -----------------------------------------------------
    # Create consistent response IDs
    # -----------------------------------------------------

    df[
        "response_id"
    ] = range(

        1,

        len(df) + 1

    )


    # -----------------------------------------------------
    # Put response ID first
    # -----------------------------------------------------

    remaining_columns = [

        column

        for column

        in df.columns

        if column
        != "response_id"

    ]


    df = df[

        [

            "response_id",

            *remaining_columns

        ]

    ]


    return df


# =========================================================
# 7. CREATE GEMINI PROMPT
# =========================================================

def create_prompt(
    dataframe
):
    """
    Creates a structured classification prompt.

    The prompt includes category definitions,
    sentiment guidance, urgency definitions,
    and tie-breaking rules.
    """

    survey_responses = (

        dataframe[

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


TIE-BREAKING RULES

1. Choose the category based on the main cause or
topic of the response, not only emotional words.

2. If deadlines, workload, prioritization,
procrastination, or task planning are the main
topic, choose Time Management, even when the
response also mentions stress or burnout.

3. Choose Exam Stress when exams, tests, grades,
or exam pressure are the main topic.

4. Use High urgency only when the response
describes strong distress, burnout, severe
anxiety, panic, or feeling significantly
overwhelmed.

5. Ordinary stress, concern, or difficulty
without signs of severe distress should
usually be Medium urgency.

6. Use Low urgency for positive observations,
preferences, and ordinary suggestions.


Return one classification for every response.

Survey responses:

{survey_text}
"""


    return prompt


# =========================================================
# 8. RUN GEMINI CLASSIFICATION
# =========================================================

def classify_responses(
    dataframe,
    api_key
):
    """
    Classifies all survey responses in one request.

    The function:

    1. Uses structured Gemini output.
    2. Retries temporary service errors.
    3. Uses Gemini Flash-Lite as a backup.
    4. Verifies that every response is classified.
    """

    client = genai.Client(

        api_key=api_key

    )


    prompt = create_prompt(

        dataframe

    )


    # Try the main model first.
    # Use Flash-Lite as a backup.
    model_options = [

        "gemini-3.5-flash",

        "gemini-3.1-flash-lite"

    ]


    last_error = None


    # -----------------------------------------------------
    # Try each Gemini model
    # -----------------------------------------------------

    for model_name in model_options:


        # -------------------------------------------------
        # Try each model up to three times
        # -------------------------------------------------

        for attempt in range(3):


            try:


                # -----------------------------------------
                # Send classification request
                # -----------------------------------------

                response = (

                    client

                    .models

                    .generate_content(

                        model=model_name,

                        contents=prompt,

                        config={

                            "response_mime_type":
                                "application/json",

                            "response_schema":
                                ClassificationBatch

                        }

                    )

                )


                # -----------------------------------------
                # Read structured Gemini response
                # -----------------------------------------

                if (

                    response.parsed
                    is not None

                ):


                    if isinstance(

                        response.parsed,

                        ClassificationBatch

                    ):


                        classification_batch = (

                            response.parsed

                        )


                    else:


                        classification_batch = (

                            ClassificationBatch

                            .model_validate(

                                response.parsed

                            )

                        )


                else:


                    classification_batch = (

                        ClassificationBatch

                        .model_validate_json(

                            response.text

                        )

                    )


                # -----------------------------------------
                # Convert Gemini output to records
                # -----------------------------------------

                classification_records = [

                    classification.model_dump()

                    for classification

                    in (

                        classification_batch

                        .classifications

                    )

                ]


                # -----------------------------------------
                # Convert records to dataframe
                # -----------------------------------------

                results_df = pd.DataFrame(

                    classification_records

                )


                # -----------------------------------------
                # Validate returned response IDs
                # -----------------------------------------

                expected_ids = set(

                    dataframe[
                        "response_id"
                    ]

                )


                returned_ids = set(

                    results_df[
                        "response_id"
                    ]

                )


                missing_ids = (

                    expected_ids

                    - returned_ids

                )


                unexpected_ids = (

                    returned_ids

                    - expected_ids

                )


                # -----------------------------------------
                # Check for missing classifications
                # -----------------------------------------

                if missing_ids:


                    raise ValueError(

                        "Gemini did not classify "
                        "these response IDs: "

                        f"{sorted(missing_ids)}"

                    )


                # -----------------------------------------
                # Check for unexpected IDs
                # -----------------------------------------

                if unexpected_ids:


                    raise ValueError(

                        "Gemini returned unexpected "
                        "response IDs: "

                        f"{sorted(unexpected_ids)}"

                    )


                # -----------------------------------------
                # Check result count
                # -----------------------------------------

                if (

                    len(results_df)

                    !=

                    len(dataframe)

                ):


                    raise ValueError(

                        "The number of AI results "
                        "does not match the number "
                        "of survey responses."

                    )


                # -----------------------------------------
                # Check for duplicate IDs
                # -----------------------------------------

                duplicate_ids = (

                    results_df[
                        "response_id"
                    ]

                    .duplicated()

                    .any()

                )


                if duplicate_ids:


                    raise ValueError(

                        "Gemini returned duplicate "
                        "response IDs."

                    )


                # -----------------------------------------
                # Merge original data with Gemini output
                # -----------------------------------------

                classified_df = (

                    dataframe

                    .merge(

                        results_df,

                        on="response_id",

                        how="left",

                        validate="one_to_one"

                    )

                )


                # -----------------------------------------
                # Return completed analysis
                # -----------------------------------------

                return classified_df


            # ------------------------------------------------
            # Handle Gemini errors
            # ------------------------------------------------

            except Exception as error:


                error_text = str(

                    error

                )


                error_text_lower = (

                    error_text

                    .lower()

                )


                temporary_error = (


                    "429"
                    in error_text


                    or


                    "500"
                    in error_text


                    or


                    "502"
                    in error_text


                    or


                    "503"
                    in error_text


                    or


                    "504"
                    in error_text


                    or


                    "resource_exhausted"
                    in error_text_lower


                    or


                    "unavailable"
                    in error_text_lower


                    or


                    "high demand"
                    in error_text_lower


                    or


                    "temporarily overloaded"
                    in error_text_lower


                    or


                    "deadline_exceeded"
                    in error_text_lower

                )


                # -----------------------------------------
                # Stop immediately for unrelated errors
                # -----------------------------------------

                if not temporary_error:

                    raise


                last_error = error


                # -----------------------------------------
                # Wait before retrying
                # -----------------------------------------

                if attempt < 2:


                    wait_time = (

                        2 ** attempt

                        + random.uniform(

                            0,

                            1

                        )

                    )


                    time.sleep(

                        wait_time

                    )


                # -----------------------------------------
                # Switch model after three failed attempts
                # -----------------------------------------

                else:

                    break


    # -----------------------------------------------------
    # Both Gemini models failed
    # -----------------------------------------------------

    raise RuntimeError(

        "Gemini request failed after "
        "automatic retries. "

        f"Original error: {last_error}"

    ) from last_error


# =========================================================
# 9. RECOMMENDATIONS
# =========================================================

recommendation_map = {


    "Time Management":

        "Provide planning templates, deadline reminders, "
        "and resources for prioritizing assignments.",


    "Exam Stress":

        "Expand exam preparation support, review sessions, "
        "and stress-management resources.",


    "Study Strategy":

        "Provide practice problems, study-method guidance, "
        "and resources for retaining course material.",


    "Course Resources":

        "Increase access to study guides, examples, "
        "practice materials, and instructor support.",


    "Scheduling":

        "Offer more flexible meeting times and improve "
        "coordination of academic support resources.",


    "Motivation":

        "Encourage study groups, accountability programs, "
        "and collaborative learning opportunities.",


    "Other":

        "Review individual responses to identify needs "
        "that do not fit the current categories."

}


# =========================================================
# 10. APP HEADER
# =========================================================

st.title(

    "📊 AI-Powered Survey Response Analyzer"

)


st.write(

    """
    Transform open-ended student survey responses
    into structured themes, sentiment insights,
    urgency levels, and actionable recommendations.
    """

)


st.caption(

    "Built with Python, Streamlit, Gemini, "
    "Pandas, Pydantic, and structured AI outputs."

)


# =========================================================
# 11. SIDEBAR
# =========================================================

with st.sidebar:


    st.header(

        "How to Use"

    )


    st.write(

        """
        1. Choose the included sample dataset
        or upload your own CSV.

        2. Make sure the data contains a column
        named `response_text`.

        3. Click **Analyze Responses**.

        4. Review the insights and download
        the classified results.
        """

    )


    st.divider()


    st.subheader(

        "Required Column"

    )


    st.code(

        "response_text"

    )


    st.subheader(

        "Analysis Limit"

    )


    st.write(

        f"Maximum responses per analysis: "
        f"**{MAX_RESPONSES}**"

    )


    st.info(

        "AI classifications should be reviewed "
        "by a human before high-impact decisions "
        "are made."

    )


# =========================================================
# 12. SELECT DATA SOURCE
# =========================================================

st.header(

    "1. Select Survey Data"

)


data_source = st.radio(

    "Choose a data source:",

    [

        "Use sample survey",

        "Upload my own CSV"

    ],

    horizontal=True

)


# =========================================================
# 13. LOAD DATA
# =========================================================

raw_df = None


# ---------------------------------------------------------
# Load sample survey
# ---------------------------------------------------------

if (

    data_source

    ==

    "Use sample survey"

):


    try:


        raw_df = pd.read_csv(

            sample_file

        )


        st.success(

            "Sample student survey loaded."

        )


    except FileNotFoundError:


        st.error(

            "The sample survey file "
            "could not be found."

        )


# ---------------------------------------------------------
# Load uploaded CSV
# ---------------------------------------------------------

else:


    uploaded_file = st.file_uploader(

        "Upload a CSV file",

        type=[

            "csv"

        ]

    )


    if (

        uploaded_file

        is not None

    ):


        try:


            raw_df = pd.read_csv(

                uploaded_file

            )


            st.success(

                "CSV uploaded successfully."

            )


        except Exception as error:


            st.error(

                "The uploaded CSV "
                "could not be read."

            )


            st.write(

                str(error)

            )


# =========================================================
# 14. PREVIEW DATA
# =========================================================

if (

    raw_df

    is not None

):


    try:


        prepared_df = prepare_data(

            raw_df

        )


        st.subheader(

            "Data Preview"

        )


        st.dataframe(

            prepared_df.head(

                10

            ),

            use_container_width=True,

            hide_index=True

        )


        st.write(

            f"**Responses ready for analysis:** "
            f"{len(prepared_df)}"

        )


        # =================================================
        # 15. RUN ANALYSIS
        # =================================================

        st.header(

            "2. Run AI Analysis"

        )


        analyze_button = st.button(

            "✨ Analyze Responses",

            type="primary",

            use_container_width=True

        )


        if analyze_button:


            api_key = get_api_key()


            # ---------------------------------------------
            # Check API key
            # ---------------------------------------------

            if not api_key:


                st.error(

                    "Gemini API key was not found. "
                    "Check the local .env file or "
                    "the deployed Streamlit secrets."

                )


            # ---------------------------------------------
            # Run Gemini analysis
            # ---------------------------------------------

            else:


                try:


                    with st.spinner(

                        "Gemini is analyzing "
                        "the survey responses..."

                    ):


                        classified_df = (

                            classify_responses(

                                prepared_df,

                                api_key

                            )

                        )


                    st.session_state[

                        "classified_results"

                    ] = classified_df


                    st.success(

                        "AI analysis completed."

                    )


                # -----------------------------------------
                # Display friendly error messages
                # -----------------------------------------

                except Exception as error:


                    error_text = str(

                        error

                    )


                    error_text_lower = (

                        error_text

                        .lower()

                    )


                    if (

                        "429"
                        in error_text

                        or

                        "resource_exhausted"
                        in error_text_lower

                        or

                        "quota"
                        in error_text_lower

                    ):


                        st.error(

                            "The Gemini API usage limit "
                            "was reached. Please wait "
                            "for the quota to reset, "
                            "then try again."

                        )


                    elif (

                        "503"
                        in error_text

                        or

                        "unavailable"
                        in error_text_lower

                        or

                        "high demand"
                        in error_text_lower

                        or

                        "temporarily"
                        in error_text_lower

                    ):


                        st.error(

                            "Gemini is experiencing "
                            "temporary high demand. "
                            "The app already retried "
                            "and tested a backup model. "
                            "Please wait briefly, "
                            "then try again."

                        )


                    else:


                        st.error(

                            "The AI analysis could "
                            "not be completed."

                        )


                        st.write(

                            str(error)

                        )


    except ValueError as error:


        st.error(

            str(error)

        )


# =========================================================
# 16. DISPLAY ANALYSIS RESULTS
# =========================================================

if (

    "classified_results"

    in st.session_state

):


    results_df = (

        st.session_state[

            "classified_results"

        ]

    )


    st.divider()


    st.header(

        "3. Analysis Results"

    )


    # -----------------------------------------------------
    # Calculate summaries
    # -----------------------------------------------------

    theme_counts = (

        results_df[

            "theme"

        ]

        .value_counts()

    )


    sentiment_counts = (

        results_df[

            "sentiment"

        ]

        .value_counts()

    )


    urgency_counts = (

        results_df[

            "urgency"

        ]

        .value_counts()

    )


    top_theme = (

        theme_counts

        .index[0]

    )


    top_sentiment = (

        sentiment_counts

        .index[0]

    )


    high_urgency_count = (

        results_df[

            "urgency"

        ]

        .eq(

            "High"

        )

        .sum()

    )


    # -----------------------------------------------------
    # Display summary metrics
    # -----------------------------------------------------

    (

        metric_1,

        metric_2,

        metric_3,

        metric_4

    ) = st.columns(4)


    metric_1.metric(

        "Responses",

        len(results_df)

    )


    metric_2.metric(

        "Top Theme",

        top_theme

    )


    metric_3.metric(

        "Top Sentiment",

        top_sentiment

    )


    metric_4.metric(

        "High Urgency",

        int(

            high_urgency_count

        )

    )


    # -----------------------------------------------------
    # Create result tabs
    # -----------------------------------------------------

    (

        overview_tab,

        responses_tab,

        recommendations_tab

    ) = st.tabs(

        [

            "Overview",

            "Classified Responses",

            "Recommendations"

        ]

    )


    # =====================================================
    # OVERVIEW TAB
    # =====================================================

    with overview_tab:


        (

            chart_1,

            chart_2

        ) = st.columns(2)


        with chart_1:


            st.subheader(

                "Theme Distribution"

            )


            st.bar_chart(

                theme_counts

            )


        with chart_2:


            st.subheader(

                "Sentiment Distribution"

            )


            st.bar_chart(

                sentiment_counts

            )


        st.subheader(

            "Urgency Distribution"

        )


        st.bar_chart(

            urgency_counts

        )


    # =====================================================
    # CLASSIFIED RESPONSES TAB
    # =====================================================

    with responses_tab:


        st.dataframe(

            results_df,

            use_container_width=True,

            hide_index=True

        )


        csv_data = (

            results_df

            .to_csv(

                index=False

            )

            .encode(

                "utf-8"

            )

        )


        st.download_button(

            label=(

                "⬇️ Download Classified Results"

            ),

            data=csv_data,

            file_name=(

                "ai_classified_survey_results.csv"

            ),

            mime="text/csv",

            use_container_width=True

        )


    # =====================================================
    # RECOMMENDATIONS TAB
    # =====================================================

    with recommendations_tab:


        st.subheader(

            "Recommended Actions"

        )


        top_three_themes = (

            theme_counts

            .head(3)

            .index

        )


        for (

            rank,

            theme

        ) in enumerate(

            top_three_themes,

            start=1

        ):


            st.markdown(

                f"""
                **{rank}. {theme}**

                {
                    recommendation_map[
                        theme
                    ]
                }
                """

            )


        st.info(

            "Recommendations are based on the "
            "most frequently identified themes "
            "and should be interpreted alongside "
            "the original survey responses."

        )


# =========================================================
# 17. FOOTER
# =========================================================

st.divider()


st.caption(

    "Portfolio project by Sara Koshy • "
    "Statistics • AI-Assisted Data Analysis"

)