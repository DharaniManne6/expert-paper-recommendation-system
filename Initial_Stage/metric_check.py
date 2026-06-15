# -----------------------------------------------------------
# PAPER -> EXPERT RECOMMENDATION SYSTEM
# Research Pipeline Implementation
# Models Compared:
# 1. TF-IDF
# 2. Sentence-BERT
# 3. SciBERT
#
# Evaluation:
# NDCG@5 with graded relevance
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
# STEP 1: LOAD DATASETS
# -----------------------------------------------------------

papers = pd.read_excel("papers_shuffled.xlsx")
experts = pd.read_excel("experts_shuffled.xlsx")

print("Papers:", len(papers))
print("Experts:", len(experts))


# -----------------------------------------------------------
# STEP 2: TEXT PREPROCESSING
# -----------------------------------------------------------

def clean_text(text):

    text = str(text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


papers["Title"] = papers["Title"].apply(clean_text)
papers["Abstract"] = papers["Abstract"].apply(clean_text)

experts["Profile_Text"] = experts["Profile_Text"].apply(clean_text)


# -----------------------------------------------------------
# STEP 3: CREATE DOCUMENT TEXT
# -----------------------------------------------------------

paper_texts = (papers["Title"] + ". " + papers["Abstract"]).tolist()
expert_texts = experts["Profile_Text"].tolist()


# -----------------------------------------------------------
# STEP 4: DEFINE GRADED RELEVANCE
# -----------------------------------------------------------

def compute_relevance(paper_domain, expert_domain):

    if paper_domain == expert_domain:
        return 3

    # related domains (example relationship)
    if (paper_domain == "AI" and expert_domain == "Security") or \
       (paper_domain == "Security" and expert_domain == "AI"):
        return 1

    return 0


# -----------------------------------------------------------
# STEP 5: EVALUATION FUNCTION
# -----------------------------------------------------------

def evaluate_ndcg(similarity_matrix, papers, experts, k=5):

    domain_bonus = 0.10
    adjusted_similarity = similarity_matrix.copy()

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

            expert_domain = experts.iloc[j]["Domain"]

            relevance.append(
                compute_relevance(paper_domain, expert_domain)
            )

        relevance = np.array(relevance).reshape(1, -1)

        scores = adjusted_similarity[i].reshape(1, -1)

        ndcg = ndcg_score(relevance, scores, k=k)

        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores), adjusted_similarity


# ===========================================================
# MODEL 1 — TF-IDF
# ===========================================================

print("\nRunning TF-IDF model")

tfidf = TfidfVectorizer(stop_words="english")

combined = paper_texts + expert_texts

tfidf_matrix = tfidf.fit_transform(combined)

paper_vectors = tfidf_matrix[:len(papers)]
expert_vectors = tfidf_matrix[len(papers):]

similarity_matrix = cosine_similarity(paper_vectors, expert_vectors)

tfidf_ndcg, _ = evaluate_ndcg(similarity_matrix, papers, experts)

print("TF-IDF NDCG@5:", tfidf_ndcg)


# ===========================================================
# MODEL 2 — Sentence-BERT
# ===========================================================

print("\nRunning Sentence-BERT")

sbert = SentenceTransformer("all-mpnet-base-v2")

paper_emb = sbert.encode(paper_texts)
expert_emb = sbert.encode(expert_texts)

paper_emb = normalize(paper_emb)
expert_emb = normalize(expert_emb)

similarity_matrix = cosine_similarity(paper_emb, expert_emb)

sbert_ndcg, adjusted_similarity = evaluate_ndcg(similarity_matrix, papers, experts)

print("Sentence-BERT NDCG@5:", sbert_ndcg)


# ===========================================================
# MODEL 3 — SciBERT
# ===========================================================

print("\nRunning SciBERT")

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def get_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = scibert(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.numpy()[0]


print("Encoding papers")

paper_scibert = np.array([get_embedding(t) for t in paper_texts])

print("Encoding experts")

expert_scibert = np.array([get_embedding(t) for t in expert_texts])

paper_scibert = normalize(paper_scibert)
expert_scibert = normalize(expert_scibert)

similarity_matrix = cosine_similarity(paper_scibert, expert_scibert)

scibert_ndcg, _ = evaluate_ndcg(similarity_matrix, papers, experts)

print("SciBERT NDCG@5:", scibert_ndcg)


# -----------------------------------------------------------
# MODEL COMPARISON RESULTS
# -----------------------------------------------------------

print("\n========== FINAL RESULTS ==========")

print("TF-IDF:", tfidf_ndcg)
print("Sentence-BERT:", sbert_ndcg)
print("SciBERT:", scibert_ndcg)


# -----------------------------------------------------------
# PCA VISUALIZATION
# -----------------------------------------------------------

print("\nGenerating PCA visualization")

combined_embeddings = np.vstack((paper_emb, expert_emb))

pca = PCA(n_components=2)

reduced = pca.fit_transform(combined_embeddings)

paper_points = reduced[:len(papers)]
expert_points = reduced[len(papers):]

plt.figure(figsize=(10,7))

plt.scatter(
    paper_points[:,0],
    paper_points[:,1],
    alpha=0.3,
    color="gray",
    label="Papers"
)

colors = {
    "AI":"red",
    "Security":"blue",
    "Hardware":"green",
    "Medicine":"purple"
}

for domain in experts["Domain"].unique():

    idx = experts[experts["Domain"] == domain].index

    plt.scatter(
        expert_points[idx,0],
        expert_points[idx,1],
        s=200,
        marker="X",
        color=colors.get(domain,"black"),
        label=domain+" Experts"
    )

plt.title("Embedding Space Visualization (PCA)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.legend()
plt.grid(alpha=0.2)
plt.savefig("paper_expert_cluster.png", dpi=300)

for i in range(5):

    print("\n-----------------------------------")
    print("Paper:", papers.iloc[i]["Title"])
    print("Domain:", papers.iloc[i]["Domain_tag"])

    scores = adjusted_similarity[i]
    top_experts = scores.argsort()[::-1][:5]

    print("Top Experts:")

    for idx in top_experts:

        print(
            experts.iloc[idx]["Expert_ID"],
            "| Domain:", experts.iloc[idx]["Domain"]
        )



# -----------------------------------------------------------
# EXPORT RECOMMENDATIONS FOR MANUAL VALIDATION
# -----------------------------------------------------------

top_k = 5
results = []

for i in range(len(papers)):

    paper_title = papers.iloc[i]["Title"]
    paper_domain = papers.iloc[i]["Domain_tag"]

    scores = adjusted_similarity[i]

    top_experts = scores.argsort()[::-1][:top_k]

    for rank, idx in enumerate(top_experts):

        results.append({
            "Paper_Title": paper_title,
            "Paper_Domain": paper_domain,
            "Rank": rank+1,
            "Expert_ID": experts.iloc[idx]["Expert_ID"],
            "Expert_Domain": experts.iloc[idx]["Domain"],
            "Score": scores[idx]
        })


results_df = pd.DataFrame(results)

results_df.to_excel("manual_recommendation_check.xlsx", index=False)

print("Manual evaluation file saved.")

# -----------------------------------------------------------
# EXPERT -> TOP 5 PAPERS RECOMMENDATION
# -----------------------------------------------------------

print("\n==============================")
print("EXPERT -> TOP 5 PAPERS EXAMPLE")
print("==============================")

top_k = 5

# Use the BEST model (Sentence-BERT recommended)
# similarity_matrix already computed for SBERT above

expert_paper_similarity = similarity_matrix.T   # transpose

for i in range(3):  # show for first 3 experts

    print("\n-----------------------------------")
    print("Expert:", experts.iloc[i]["Expert_ID"])
    print("Domain:", experts.iloc[i]["Domain"])

    scores = expert_paper_similarity[i]

    top_papers = scores.argsort()[::-1][:top_k]

    print("Top 5 Recommended Papers:")

    for rank, idx in enumerate(top_papers):

        print(
            f"Rank {rank+1}:",
            papers.iloc[idx]["Title"],
            "| Domain:", papers.iloc[idx]["Domain_tag"]
        )

# -----------------------------------------------------------
# EXPERT -> TOP 5 PAPERS RECOMMENDATION
# -----------------------------------------------------------

print("\n==============================")
print("EXPERT -> TOP 5 PAPERS EXAMPLE")
print("==============================")

top_k = 5

# Use the BEST model (Sentence-BERT recommended)
# similarity_matrix already computed for SBERT above

expert_paper_similarity = similarity_matrix.T   # transpose

for i in range(3):  # show for first 3 experts

    print("\n-----------------------------------")
    print("Expert:", experts.iloc[i]["Expert_ID"])
    print("Domain:", experts.iloc[i]["Domain"])

    scores = expert_paper_similarity[i]

    top_papers = scores.argsort()[::-1][:top_k]

    print("Top 5 Recommended Papers:")

    for rank, idx in enumerate(top_papers):

        print(
            f"Rank {rank+1}:",
            papers.iloc[idx]["Title"],
            "| Domain:", papers.iloc[idx]["Domain_tag"]
        )