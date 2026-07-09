from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Project setup
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]
outputs_path = project_root / "outputs"

classified_file = outputs_path / "ai_classified_responses_all.csv"

df = pd.read_csv(classified_file)


# ---------------------------------------------------------
# 2. Basic summary of AI classifications
# ---------------------------------------------------------

print("\nAI Classification Results Summary")
print("--------------------------------")

print("\nNumber of classified responses:", len(df))

print("\nTheme counts:")
theme_counts = df["theme"].value_counts()
print(theme_counts)

print("\nSentiment counts:")
sentiment_counts = df["sentiment"].value_counts()
print(sentiment_counts)

print("\nUrgency counts:")
urgency_counts = df["urgency"].value_counts()
print(urgency_counts)


# ---------------------------------------------------------
# 3. Save charts
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
theme_counts.plot(kind="bar")
plt.title("Most Common Themes in Survey Responses")
plt.xlabel("Theme")
plt.ylabel("Number of Responses")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outputs_path / "theme_counts.png")
plt.show()

plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Breakdown of Survey Responses")
plt.xlabel("Sentiment")
plt.ylabel("Number of Responses")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outputs_path / "sentiment_counts.png")
plt.show()

plt.figure(figsize=(8, 5))
urgency_counts.plot(kind="bar")
plt.title("Urgency Breakdown of Survey Responses")
plt.xlabel("Urgency")
plt.ylabel("Number of Responses")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outputs_path / "urgency_counts.png")
plt.show()


# ---------------------------------------------------------
# 4. Create a simple written insight report
# ---------------------------------------------------------

top_theme = theme_counts.index[0]
top_theme_count = theme_counts.iloc[0]

top_sentiment = sentiment_counts.index[0]
top_sentiment_count = sentiment_counts.iloc[0]

top_urgency = urgency_counts.index[0]
top_urgency_count = urgency_counts.iloc[0]

report = f"""
AI Survey Response Analyzer: Summary Report

Total responses analyzed: {len(df)}

Key Findings:
1. The most common theme was "{top_theme}", appearing in {top_theme_count} responses.
2. The most common sentiment was "{top_sentiment}", appearing in {top_sentiment_count} responses.
3. The most common urgency level was "{top_urgency}", appearing in {top_urgency_count} responses.

Theme Breakdown:
{theme_counts.to_string()}

Sentiment Breakdown:
{sentiment_counts.to_string()}

Urgency Breakdown:
{urgency_counts.to_string()}

Early Interpretation:
The AI classification results suggest that students are mainly struggling with patterns related to {top_theme.lower()}. These results can help identify where support resources, study planning, or course improvements may be most useful.

Recommended Next Steps:
- Review the most common theme and connect it to specific student needs.
- Check a sample of AI labels manually to evaluate accuracy.
- Use the final results to build a Streamlit dashboard.
"""

report_file = outputs_path / "summary_report.txt"

with open(report_file, "w") as f:
    f.write(report)

print("\nSummary report saved to:")
print(report_file)

print("\nCharts saved in the outputs folder.")