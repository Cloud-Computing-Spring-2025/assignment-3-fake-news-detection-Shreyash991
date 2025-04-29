# Fake News Detection Project

## Overview
For this assignment, I built a machine learning pipeline using Spark MLlib to detect fake news articles. The system analyzes news content and classifies articles as either FAKE or REAL based on their text features.

## Dataset
I worked with the `fake_news_sample.csv` dataset which contains:
- News article IDs
- Article titles
- Article text content
- Labels indicating whether each article is FAKE or REAL

## Project Tasks

### Task 1: Data Loading & Exploration
First, I loaded the CSV data and did some basic exploration:
- Loaded the dataset with schema inference
- Created a temp view for SQL queries
- Checked the first few rows to understand the data structure
- Counted total articles in the dataset
- Identified the different label categories
- Saved results to task1_output.csv

### Task 2: Text Preprocessing
Next, I cleaned up the text data:
- Combined title and text fields
- Converted everything to lowercase
- Tokenized the text into individual words
- Removed common stopwords that don't add meaning
- Limited to first 10 tokens to avoid overfitting
- Saved the processed text to task2_output.csv

### Task 3: Feature Engineering
For the ML model to work with text, I needed numerical features:
- Applied TF-IDF vectorization (using HashingTF with 200 features)
- Converted text labels to numerical indices
- Saved the feature vectors to task3_output.csv

### Task 4: Model Building
I split the data and trained a classifier:
- Created 70/30 train/test split
- Built a Logistic Regression model with custom parameters
- Generated predictions on the test set
- Saved prediction results to task4_output.csv

### Task 5: Model Evaluation
Finally, I checked how well the model performed:
- Calculated accuracy score
- Measured F1 score for better evaluation with imbalanced data
- Saved evaluation metrics to task5_output.csv

## Running the Code
To run this project:

1. Make sure you have Spark installed
2. Place the `fake_news_sample.csv` file in the project folder
3. Run the script:
   ```
   spark-submit fake_news_detection.py
   ```
4. Check the output CSV files to see results from each task

## Challenges & Learnings
Working with text data in Spark was interesting - balancing between using enough text for good predictions while avoiding performance issues was tricky. The TF-IDF approach worked well for this classification task, and I learned a lot about tuning Logistic Regression parameters for text classification problems.