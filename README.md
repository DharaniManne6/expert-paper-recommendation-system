# Expert Paper Recommendation System with Adversarial Robustness

## Overview

This project is an AI-powered recommendation system designed to recommend the most relevant research papers to domain experts using semantic similarity and embedding-based retrieval techniques.

The system uses NLP and transformer-based embeddings to match experts with research papers from domains such as:

* Artificial Intelligence
* Security
* Hardware
* Medicine

The project also explores future extensions in adversarial machine learning and data poisoning attacks on recommendation systems.

---

# Objectives

* Build a semantic recommendation system for expert-paper matching
* Compare traditional and transformer-based embedding models
* Evaluate recommendation quality using ranking metrics
* Analyze robustness of recommendation systems against poisoning attacks
* Prepare the foundation for future adversarial ML research

---

# Dataset

## Papers Dataset

Collected around 7000+ research papers using automated scraping from arXiv and IEEE sources.

### Dataset Fields

* Paper_ID
* Title
* Authors
* Abstract
* Source
* URL
* Domain_Tag
* arXiv_ID

## Expert Dataset

Created a curated expert dataset with domain-specialized experts across:

* AI
* Security
* Hardware
* Medicine

---

# Technologies Used

## Programming

* Python

## NLP & ML Libraries

* Scikit-learn
* TensorFlow
* PyTorch
* Hugging Face Transformers
* Sentence-Transformers

## Models

* TF-IDF
* Sentence-BERT
* SciBERT

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

---

# Project Pipeline

## Step 1: Data Collection

* Built custom scraping scripts using APIs and web scraping
* Collected real research papers from academic platforms
* Removed duplicates and cleaned missing values

## Step 2: Data Preprocessing

* Text cleaning
* Lowercasing
* Stopword removal
* Tokenization
* Domain tagging using keyword matching

## Step 3: Embedding Generation

Implemented three approaches:

### TF-IDF

Traditional vectorization baseline.

### Sentence-BERT

Semantic sentence embeddings for contextual similarity.

### SciBERT

Scientific-domain transformer model specialized for research papers.

---

# Recommendation System

The system:

1. Converts expert interests into embeddings
2. Converts paper abstracts into embeddings
3. Computes cosine similarity
4. Retrieves Top-K relevant papers

---

# Evaluation Metrics

## NDCG@5 (Normalized Discounted Cumulative Gain)

Measures ranking quality by checking whether highly relevant papers appear at top positions.

### Why NDCG?

Because recommendation systems are ranking problems, not classification problems.

NDCG rewards:

* Correct recommendations
* Better ranking positions

---

# Model Performance

| Model         | NDCG@5 |
| ------------- | ------ |
| TF-IDF        | 0.979  |
| Sentence-BERT | 0.856  |
| SciBERT       | 0.989  |

## Best Model

SciBERT performed best because it is pretrained on scientific text and captures research semantics better than general embeddings.

---

# Visualizations

* Paper embedding visualization
* Expert-paper clustering
* Domain similarity clusters

---

# Challenges Faced

## Large Dataset Handling

Similarity matrices became extremely large (>500MB).

### Solution

* Used embedding caching
* Optimized similarity computations
* Removed large generated outputs from Git tracking

---

## Domain Overlap

AI overlapped heavily with medicine and security papers.

### Solution

Implemented rule-based domain tagging and filtering logic.

---

## Evaluation Challenges

No interaction/ground truth dataset available.

### Solution

Used:

* NDCG ranking evaluation
* Manual recommendation validation

---

# Future Work

## Adversarial Data Poisoning

Future implementation will include:

* Targeted poisoning attacks
* Untargeted poisoning attacks
* Semantic-preserving word substitution
* Optimization-based attacks
* Context-aware BERT poisoning

Goal:
Analyze how malicious text modifications can manipulate recommendation outputs.

---

# Research Direction

Current research focus:

* Recommendation system robustness
* Adversarial machine learning
* Poisoning attacks on semantic retrieval systems

Future goal:
Develop defense mechanisms against adversarial recommendation manipulation.

---

# Repository Structure

```text
Dataset_Initial/
Dataset_later/
Initial_Stage/
Initial_results/
cache_embeddings/
similarity_outputs/
```

---

# How to Run

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Main Pipeline

```bash
python main.py
```

---

# Author

Dharani Manne
M.S. Data Science — Wright State University

Research Interests:

* NLP
* Recommendation Systems
* Adversarial Machine Learning
* LLM Security
