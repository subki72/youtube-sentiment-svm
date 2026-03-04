# YouTube Comment Sentiment Analysis Using Support Vector Machine (SVM)

## Project Overview
This repository provides an end-to-end Machine Learning pipeline to analyze and classify the sentiment of Indonesian YouTube comments. The project addresses the challenges of informal language, class imbalance, and feature extraction in social media text data. The classification is performed using a Support Vector Machine (SVM) model, optimized for high precision and recall in detecting both positive and negative public opinions.

## Key Technical Features
1. **Automated Data Acquisition:** Integrated scraping mechanism using `youtube-comment-downloader` to fetch real-time public feedback without requiring a YouTube API Key.
2. **Specialized Indonesian Preprocessing:** - Normalization of colloquial/slang Indonesian words using a custom dictionary.
   - Advanced stemming via the Nazief & Adriani algorithm (Sastrawi library).
   - Noise reduction through regex-based filtering (URLs, hashtags, mentions, and non-alphabetic characters).
3. **Feature Engineering:** Implementation of `TfidfVectorizer` with an `ngram_range` of (1, 2) to capture both individual words and meaningful phrases.
4. **Optimized Classification:**
   - **Imbalance Handling:** Utilizes `class_weight='balanced'` to ensure the model remains sensitive to the minority class (Positive sentiment).
   - **Hyperparameter Tuning:** Systematic search for optimal parameters (C, kernel, and gamma) via `GridSearchCV` with 5-Fold Cross-Validation.
5. **Model Serialization:** Exports the trained model and vectorizer using `joblib` for immediate deployment into production environments.

## Directory Structure
- `Sentiment.ipynb`: Main Jupyter Notebook containing the full execution pipeline.
- `requirements.txt`: List of required Python dependencies.
- `.gitignore`: Configuration for excluding large datasets, environment files, and serialized models from the version control.

## Installation and Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/subki72/youtube-sentiment-svm.git](https://github.com/subki72/youtube-sentiment-svm.git)
   cd youtube-sentiment-svm

```

2. Install dependencies:
```bash
pip install -r requirements.txt

```


3. Run the analysis:
Execute all cells in `Sentiment.ipynb` to scrape data, train the model, and generate evaluation reports.

## Evaluation Results

The model is evaluated using a comprehensive classification report and confusion matrix on unseen test data. By prioritizing the F1-Macro score, the model demonstrates a robust ability to distinguish sentiment in highly imbalanced and informal datasets, reaching a strong balance between Precision and Recall.

## Visualization

The project includes WordCloud generation to provide qualitative insights, allowing stakeholders to identify the most dominant terms driving public sentiment in both categories.

