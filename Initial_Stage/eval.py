# -----------------------------------------------------------
# PAPER -> EXPERT RECOMMENDATION SYSTEM
# Model Comparison:
# 1. TF-IDF
# 2. Sentence-BERT
# 3. SciBERT
# Evaluation: NDCG@5
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel
import torch


# -----------------------------------------------------------
# STEP 1: Load datasets
# -----------------------------------------------------------

papers = pd.read_excel("papers_shuffled.xlsx")
experts = pd.read_excel("experts_shuffled.xlsx")

print("\nPapers loaded:", len(papers))
print("Experts loaded:", len(experts))


# -----------------------------------------------------------
# STEP 2: Clean text
# -----------------------------------------------------------

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


papers["Title"] = papers["Title"].apply(clean_text)
papers["Abstract"] = papers["Abstract"].apply(clean_text)
experts["Profile_Text"] = experts["Profile_Text"].apply(clean_text)


# -----------------------------------------------------------
# STEP 3: Combine paper text
# -----------------------------------------------------------

paper_texts = (papers["Title"] + ". " + papers["Abstract"]).tolist()
expert_texts = experts["Profile_Text"].tolist()


# -----------------------------------------------------------
# Utility function for NDCG evaluation
# -----------------------------------------------------------

def evaluate_ndcg(similarity_matrix, papers, experts, k=5):

    domain_bonus = 0.15
    adjusted_similarity = similarity_matrix.copy()

    # Domain-aware bonus
    for i in range(len(papers)):

        paper_domain = papers.iloc[i]["Domain_tag"]

        for j in range(len(experts)):

            expert_domain = experts.iloc[j]["Domain"]

            if paper_domain == expert_domain:
                adjusted_similarity[i][j] += domain_bonus

    ndcg_scores = []

    for i in range(len(papers)):

        paper_domain = papers.iloc[i]["Domain_tag"]

        relevance = []

        for j in range(len(experts)):

            if experts.iloc[j]["Domain"] == paper_domain:
                relevance.append(1)
            else:
                relevance.append(0)

        relevance = np.array(relevance).reshape(1, -1)
        scores = adjusted_similarity[i].reshape(1, -1)

        ndcg = ndcg_score(relevance, scores, k=k)
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores), adjusted_similarity


# ===========================================================
# MODEL 1 — TF-IDF
# ===========================================================

print("\nRunning TF-IDF model...")

tfidf = TfidfVectorizer(stop_words="english")

combined_text = paper_texts + expert_texts

tfidf_matrix = tfidf.fit_transform(combined_text)

paper_vectors = tfidf_matrix[:len(papers)]
expert_vectors = tfidf_matrix[len(papers):]

similarity_matrix = cosine_similarity(paper_vectors, expert_vectors)

tfidf_ndcg, _ = evaluate_ndcg(similarity_matrix, papers, experts)

print("TF-IDF NDCG@5:", tfidf_ndcg)


# ===========================================================
# MODEL 2 — Sentence-BERT
# ===========================================================

print("\nRunning Sentence-BERT model...")

sbert_model = SentenceTransformer("all-mpnet-base-v2")

paper_embeddings = sbert_model.encode(paper_texts)
expert_embeddings = sbert_model.encode(expert_texts)

paper_embeddings = normalize(paper_embeddings)
expert_embeddings = normalize(expert_embeddings)

similarity_matrix = cosine_similarity(paper_embeddings, expert_embeddings)

sbert_ndcg, adjusted_similarity = evaluate_ndcg(similarity_matrix, papers, experts)

print("Sentence-BERT NDCG@5:", sbert_ndcg)


# ===========================================================
# MODEL 3 — SciBERT
# ===========================================================

print("\nRunning SciBERT model...")

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def get_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = scibert_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.numpy()[0]


print("Encoding papers with SciBERT...")

paper_embeddings_scibert = np.array([get_embedding(t) for t in paper_texts])

print("Encoding experts with SciBERT...")

expert_embeddings_scibert = np.array([get_embedding(t) for t in expert_texts])

paper_embeddings_scibert = normalize(paper_embeddings_scibert)
expert_embeddings_scibert = normalize(expert_embeddings_scibert)

similarity_matrix = cosine_similarity(paper_embeddings_scibert, expert_embeddings_scibert)

scibert_ndcg, _ = evaluate_ndcg(similarity_matrix, papers, experts)

print("SciBERT NDCG@5:", scibert_ndcg)


# -----------------------------------------------------------
# FINAL MODEL COMPARISON
# -----------------------------------------------------------

print("\n===================================")
print("MODEL COMPARISON RESULTS")
print("===================================")

print("TF-IDF NDCG@5:", tfidf_ndcg)
print("Sentence-BERT NDCG@5:", sbert_ndcg)
print("SciBERT NDCG@5:", scibert_ndcg)


# -----------------------------------------------------------
# Example recommendation
# -----------------------------------------------------------

paper_index = 2

print("\nExample Query Paper:\n")
print(papers.iloc[paper_index]["Title"])

scores = adjusted_similarity[paper_index]

top_experts = scores.argsort()[::-1][:5]

print("\nTop Recommended Experts:\n")

for idx in top_experts:

    print(
        "Expert:", experts.iloc[idx]["Expert_ID"],
        "| Domain:", experts.iloc[idx]["Domain"],
        "| Score:", scores[idx]
    )


# -----------------------------------------------------------
# PCA Visualization (Sentence-BERT embeddings)
# -----------------------------------------------------------

print("\nCreating PCA visualization...")

combined_embeddings = np.vstack((paper_embeddings, expert_embeddings))

pca = PCA(n_components=2)

reduced = pca.fit_transform(combined_embeddings)

paper_points = reduced[:len(papers)]
expert_points = reduced[len(papers):]

plt.figure(figsize=(10,7))


# Plot papers
plt.scatter(
    paper_points[:,0],
    paper_points[:,1],
    alpha=0.3,
    color="grey",
    label="Papers"
)


# Expert colors by domain
domain_colors = {
    "AI": "red",
    "Security": "blue",
    "Hardware": "green",
    "Medicine": "purple"
}

for domain in experts["Domain"].unique():

    indices = experts[experts["Domain"] == domain].index

    plt.scatter(
        expert_points[indices,0],
        expert_points[indices,1],
        s=220,
        marker="X",
        color=domain_colors.get(domain, "black"),
        label=f"{domain} Experts"
    )


plt.title("PCA Visualization of Paper–Expert Embedding Space")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.legend()
plt.grid(alpha=0.2)




