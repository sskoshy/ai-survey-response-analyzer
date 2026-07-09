from pathlib import Path
from collections import Counter
import string

import pandas as pd
import matplotlib.pyplot as plt

# Find the main project folder
project_root = Path(__file__).resolve().parents[1]

# Load the survey data
csv_path = project_root / "data" / "sample_survey.csv"
df = pd.read_csv(csv_path)

# Create outputs folder if it does not already exist
outputs_path = project_root / "outputs"
outputs_path.mkdir(exist_ok=True)

# Preview the data
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

# Basic summary
print("\nNumber of responses:", len(df))
print("Average rating:", round(df["rating"].mean(), 2))

# Average rating by student year
print("\nAverage rating by student year:")
avg_by_year = df.groupby("student_year")["rating"].mean().round(2)
print(avg_by_year)

# Average rating by major group
print("\nAverage rating by major group:")
avg_by_major = df.groupby("major_group")["rating"].mean().round(2)
print(avg_by_major)

# Count responses by rating
print("\nRating counts:")
rating_counts = df["rating"].value_counts().sort_index()
print(rating_counts)

# Bar chart: average rating by student year
plt.figure(figsize=(8, 5))
avg_by_year.sort_values().plot(kind="bar")
plt.title("Average Survey Rating by Student Year")
plt.xlabel("Student Year")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outputs_path / "average_rating_by_student_year.png")
plt.show()

# Basic text cleaning
stop_words = {
    "i", "me", "my", "we", "our", "the", "a", "an", "and", "or", "but",
    "to", "of", "in", "on", "for", "with", "when", "it", "is", "are",
    "am", "be", "have", "has", "do", "does", "not", "too", "more",
    "they", "them", "this", "that", "as", "at", "by"
}

all_text = " ".join(df["response_text"].astype(str)).lower()

# Remove punctuation
clean_text = all_text.translate(str.maketrans("", "", string.punctuation))

# Split into words and remove common stop words
words = [
    word for word in clean_text.split()
    if word not in stop_words and len(word) > 2
]

word_counts = Counter(words)
top_words = word_counts.most_common(10)

print("\nTop 10 most common words:")
for word, count in top_words:
    print(f"{word}: {count}")

# Convert top words to dataframe
top_words_df = pd.DataFrame(top_words, columns=["word", "count"])

# Bar chart: top words
plt.figure(figsize=(8, 5))
plt.bar(top_words_df["word"], top_words_df["count"])
plt.title("Most Common Words in Survey Responses")
plt.xlabel("Word")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outputs_path / "top_words.png")
plt.show()

print("\nCharts saved in the outputs folder.")