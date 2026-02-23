# Cybersecurity NLP Classification  
## Classical ML vs Transformer Models for Threat Categorization

---

Predictive NLP system for classifying cybersecurity news articles into threat categories.

This project compares **domain-engineered classical machine learning models** with a **fine-tuned transformer (DeBERTa)** to evaluate performance on a small, imbalanced, domain-specific corpus.

---

## Problem Framing

Four-class supervised text classification problem:

- **Vulnerability** (36%)
- **Malware** (35%)
- **Cyber_Attack** (19%)
- **Data_Breach** (10%)

Dataset size: **3,742** cybersecurity news articles

Research Question:

> Can domain-engineered classical ML outperform a fine-tuned transformer on a small, imbalanced cybersecurity dataset?

---

## System Pipeline

### 1. Text Processing Layer

- Merge title + article body  
- Lowercasing  
- URL and punctuation removal  
- Stopword removal  
- Lemmatization  
- Bigram / trigram analysis  
- Part-of-speech tagging  

---

### 2. Domain Feature Engineering

Cybersecurity-aware signals engineered into the feature matrix:

- CVE pattern detection  
- IP address detection  
- Email detection  
- File hash detection  
- Threat keyword flags (ransomware, phishing, malware)  
- Structural document features  
- POS-based document characterization  

This hybrid representation captures both linguistic structure and technical threat indicators.

---

## Modeling Layer

### Classical Machine Learning

Feature Representation:
- Word-level TF-IDF (1–3 grams)  
- Character-level TF-IDF (3–6 grams)  
- Domain-engineered features appended to sparse matrix  

Models Evaluated:
- Naive Bayes  
- Decision Tree  
- Maximum Entropy  
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- K-Nearest Neighbors  

**Best Performing Model:** Tuned Logistic Regression  
Test Accuracy: ~83%

---

### Transformer Model

Fine-tuned **DeBERTa-v3-base** using HuggingFace.

Training Strategy:
- Stratified 80/10/10 split  
- Class weights  
- Focal loss  
- Hyperparameter tuning (Optuna)  
- 512-token context window  

Test Accuracy: ~70.7%

Primary confusion observed between **Cyber_Attack** and **Malware** classes.

---

## Key Insight

Despite transformer dominance in large-scale NLP tasks, a tuned classical model outperformed DeBERTa on this dataset.

Contributing factors:

- Limited dataset size (~3.7K samples)  
- High semantic overlap across threat categories  
- Domain-engineered lexical signals captured class distinctions effectively  

This highlights that **feature engineering + classical ML remains competitive for small, domain-specific corpora**.

---

## Practical Applications

- Automated cybersecurity news labeling  
- Threat triage support  
- Analyst workload reduction  
- Threat distribution monitoring  
- Security intelligence dashboards  

---

## Repository Structure

Cybersecurity-NLP-Classification/
│
├── data/
│   └── TheHackerNews_Dataset.xlsx
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classical_ml.ipynb
│   ├── 04_deberta_modeling.ipynb
│   └── 05_deberta_hyperparam_tuning.ipynb
│
├── slides/
│   └── presentation.pdf
│
├── requirements.txt
├── README.md
└── LICENSE

---

## Installation

git clone <your_repo_url>  
cd Cybersecurity-NLP-Classification  
pip install -r requirements.txt  

---

## Technologies Used

- Python  
- scikit-learn  
- NLTK  
- HuggingFace Transformers  
- PyTorch  
- Optuna  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  

---

## License

MIT
