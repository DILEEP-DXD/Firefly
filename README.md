# 🔥 Firefly Spam Classification

## Overview

This project implements a spam classification system on the SMS Spam Collection dataset using a Firefly Algorithm for feature selection. Text messages are preprocessed and converted into TF-IDF vectors, after which Chi-squared filtering narrows down the top 500 features. The Firefly Algorithm then optimizes this feature subset by mimicking the attraction behavior of fireflies to maximize F1 score on a validation split. A Multinomial Naive Bayes classifier is trained on both the baseline and the optimized feature set, and their performances are compared through accuracy, F1 score, confusion matrix, and convergence plots.

---

## Requirements

### Python Dependencies
Install all required packages via pip:

```bash
pip install numpy pandas nltk scikit-learn matplotlib
```

### NLTK Datasets
The following NLTK datasets are **downloaded automatically** when the script runs:
- `punkt`
- `stopwords`

---

## Dataset
Place the `spam.csv` file (SMS Spam Collection dataset) in the same directory as the script before running.

---

## Usage

```bash
python firefly_spam.py
```

---

## Output
- Printed metrics: Accuracy and F1 Score for both Baseline and Optimized models
- Plots: Performance comparison, Confusion Matrix, Feature Reduction graph, Convergence curve
- Saved figures: `accuracy_plot.png`, `f1_plot.png`
