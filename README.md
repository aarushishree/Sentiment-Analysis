ML-Based Sentiment Analysis of Reviews | LG Electronics Internship

This repository contains a comprehensive sentiment analysis pipeline for web-scraped product reviews. The project uses classical machine learning, NLP techniques, and transformer-based deep learning models to classify reviews as Positive, Negative, or Neutral.

Project Overview:

Collected review data using Selenium-based web scraping from product review sources.

Implemented text preprocessing and cleaning including stopword removal, lowercasing, punctuation removal, and domain-specific word filtering.

Performed exploratory data analysis (EDA):

Review length distribution

Word counts and top words

Wordcloud visualization

N-gram analysis

Labeled sentiments using:

VADER

TextBlob

Built classical ML models:
Naive Bayes

Logistic Regression

SVM (LinearSVC)

Random Forest

Gradient Boosting

XGBoost

Developed a BERT-based transformer model for binary sentiment classification (Positive vs Negative).

Evaluated models using accuracy, F1 score, classification report, and confusion matrices.

Visualized model comparison across different approaches.


Tech Stack:

Languages: Python

Libraries: Pandas, Numpy, NLTK, Seaborn, Matplotlib, WordCloud, Scikit-learn, Transformers, Torch, TextBlob, VaderSentiment

Data Handling: Excel, CSV

ML Models: Naive Bayes, Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost, BERT

Web Scraping: Selenium


Workflow:

Data Collection: Selenium-based web scraping of reviews from product pages.

Data Cleaning & Preprocessing:

Lowercasing, removing punctuation, numbers, URLs, and stopwords

Tokenization and n-gram analysis

EDA & Visualization:

Review length and word count distributions

Top words and wordclouds

N-gram bar plots

Sentiment Labeling: Using VADER and TextBlob to generate sentiment labels.

Modeling:

Train/test split, TF-IDF vectorization

Train and evaluate multiple classical ML models

Hyperparameter tuning with GridSearchCV for Random Forest and XGBoost

Transformer-based BERT model for advanced classification

Evaluation: Accuracy, F1 score, classification report, and confusion matrix for all models.

Visualization: Bar plot comparing model accuracies.


Results:

BERT achieved the highest accuracy: ~76.9% on binary sentiment classification.

Classical ML models ranged from ~65% to 69% accuracy.

Confusion matrices and classification reports provided for all models.


Key Features:

End-to-end sentiment analysis pipeline

Cleaned and preprocessed text data for ML

Multiple classical and deep learning models

Visualization of insights (wordclouds, top n-grams, sentiment distributions)

Comparison of model performances


Files in the Repo:

lgdataset1.xlsx – Raw review dataset

vader_labeled_reviews1.csv – Reviews labeled with VADER sentiment

textblob_sentiment_reviews.xlsx – Reviews labeled with TextBlob sentiment

sentiment_analysis_models.ipynb – Full code and analysis notebook

accuracy_plot.png – Model accuracy comparison


References:

[NLTK Documentation](https://www.nltk.org/)

[VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)

[TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)

[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

Future Work:

Fine-tune BERT and transformer models for multi-class sentiment analysis (Positive, Negative, Neutral).

Deploy the pipeline as a web app or API for real-time sentiment prediction.

Integrate multi-source review scraping to increase dataset size and diversity.

