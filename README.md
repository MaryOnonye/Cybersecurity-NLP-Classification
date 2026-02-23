Cybersecurity NLP Classification

Classical Machine Learning vs Transformer Models for Threat Categorization

Project Overview

This project builds and compares classical machine learning models and transformer-based deep learning models for classifying cybersecurity news articles into threat categories.

The goal is to evaluate whether domain-aware feature engineering with traditional machine learning can outperform a fine-tuned transformer model when working with a relatively small, imbalanced, and semantically dense corpus.

Dataset size: 3,742 cybersecurity news articles
Classes:

Vulnerability (36%)

Malware (35%)

Cyber_Attack (19%)

Data_Breach (10%)

This is a four-class supervised text classification problem.

Research Question

Can domain-engineered classical machine learning models outperform a fine-tuned transformer model on a small, imbalanced cybersecurity dataset?

Data Processing Pipeline
Text Preprocessing

Merged title and article body

Lowercasing

URL and punctuation removal

Stopword removal

Lemmatization

Bigram and trigram analysis

Part-of-speech tagging

Indicator-of-Compromise (IOC) regex detection

Domain Feature Engineering

Custom cybersecurity-aware signals were engineered to strengthen model performance:

CVE pattern detection

IP address detection

Email detection

File hash detection

Threat keyword flags (ransomware, phishing, malware, etc.)

Structural document features

POS-based document characterization

This hybrid approach captures both linguistic structure and technical threat indicators.

Classical Machine Learning Models

Feature representation:

Word-level TF-IDF (1–3 grams)

Character-level TF-IDF (3–6 grams)

Domain-engineered features appended to TF-IDF matrix

Models evaluated:

Naïve Bayes

Decision Tree

Maximum Entropy

Logistic Regression

Random Forest

Gradient Boosting

K-Nearest Neighbors

Best Model: Tuned Logistic Regression

Test Accuracy: approximately 83%

Strong macro and weighted F1 scores

Most balanced confusion matrix across classes

Strong minority-class stability

Result: Classical ML significantly outperformed baseline NLTK models (70–73% accuracy).

Deep Learning Model: DeBERTa-v3-base

A transformer model (DeBERTa-v3-base) was fine-tuned using HuggingFace.

Training strategy:

Stratified 80/10/10 train-validation-test split

Class weights

Custom focal loss

Hyperparameter tuning via Optuna

GPU training

Context window: 512 tokens

Test Performance

Accuracy: approximately 70.7%

Macro F1: 0.6718

Weighted F1: 0.6998

The model struggled primarily with semantic overlap between Cyber_Attack and Malware.

Key Insight

Despite transformer models typically outperforming classical approaches, the tuned Logistic Regression model achieved superior performance on this dataset.

Primary reasons:

Dataset size (~3,700 articles) is relatively small for deep learning

Heavy semantic overlap between threat categories

Domain-engineered lexical features captured threat-specific signals more effectively

This project demonstrates that for small, imbalanced, domain-specific corpora, well-designed feature engineering combined with classical machine learning can outperform deep learning models.

Practical Applications

This system can:

Automate cybersecurity news labeling

Reduce analyst workload

Improve threat triage speed

Detect shifts in threat distributions

Support real-time threat dashboards

Project Structure

Cybersecurity-NLP-Classification/

data/
TheHackerNews_Dataset.xlsx

notebooks/
01_eda.ipynb
02_feature_engineering.ipynb
03_classical_ml.ipynb
04_deberta_modeling.ipynb
05_deberta_hyperparam_tuning.ipynb

slides/
presentation.pdf

requirements.txt
README.md
LICENSE

Installation

git clone <your_repo_url>
cd Cybersecurity-NLP-Classification
pip install -r requirements.txt

Technologies Used

Python

scikit-learn

NLTK

HuggingFace Transformers

PyTorch

Optuna

Pandas

NumPy

Matplotlib

Seaborn
