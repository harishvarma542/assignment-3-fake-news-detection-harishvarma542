# Fake News Classification using Spark MLlib

This project builds a simple machine learning pipeline using Apache Spark to classify news articles as **FAKE** or **REAL** based on their textual content.

---

## 📁 Dataset Used

**`fake_news_sample.csv`**

Columns:
- `id` — Unique identifier for each article.
- `title` — Title of the news article.
- `text` — Main content of the news article.
- `label` — Ground truth label: `FAKE` or `REAL`.

---

## ⚙️ Project Structure

```plaintext
.
├── fake_news_sample.csv        # Input dataset
├── fake_news_pipeline.py       # Main pipeline script
├── output/
│   ├── task1_output.csv/       # Task 1 result: Loaded & explored data
│   ├── task2_output.csv/       # Task 2 result: Cleaned & tokenized text
│   ├── task3_output.csv/       # Task 3 result: TF-IDF features and labels
│   ├── task4_output.csv/       # Task 4 result: Predictions
│   └── task5_output.csv/       # Task 5 result: Evaluation metrics
└── README.md                   # Project 

```

## ✅ Tasks Completed

### Task 1: Load & Basic Exploration
- Loaded CSV using Spark.
- Created a temporary view (news_data).
- Queried for:
    -First 5 rows
    -Total number of articles
    -Distinct label
- 🔽 Output: output/task1_output.csv

### Task 2: Text Preprocessing
- Converted text to lowercase.
- Tokenized the text.
- Removed stopwords using StopWordsRemover.
- 🔽 Output: output/task2_output.csv

### Task 3: Feature Extraction
- Applied HashingTF and IDF to generate TF-IDF vectors.
- Converted label to numeric with StringIndexer.
- 🔽 Output: output/task3_output.csv

### Task 4: Model Training
- Split the data (80% train / 20% test).
- Trained a LogisticRegression classifier.
Generated predictions.
- 🔽 Output: output/task4_output.csv

### Task 5: Model Evaluation
- Used MulticlassClassificationEvaluator to compute:
    -Accuracy
    - F1 Score
- 🔽 Output: output/task5_output.csv (as a markdown-style table)

## 🚀 How to Run

### Prerequisites
- Apache Spark installed and configured
- Python 3.8+
- Dependencies: pyspark

Run the Pipeline

``` bash
spark-submit fake_news_pipeline.py
``` 

This will:

- Process the dataset end-to-end
- Save outputs from each task to the output/ folder

