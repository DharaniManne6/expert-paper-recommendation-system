# -----------------------------------------------------------
# Paper -> Expert Recommendation System
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# -----------------------------------------------------------
# STEP 1: Load datasets
# -----------------------------------------------------------

papers = pd.read_excel("papers_shuffled.xlsx")
experts = pd.read_excel("experts_shuffled.xlsx")

print("\nFirst few papers:\n")
print(papers.head())

print("\nFirst few experts:\n")
print(experts.head())


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
# STEP 3: Combine title + abstract
# -----------------------------------------------------------

paper_texts = (
    papers["Title"] + ". " + papers["Abstract"]
).tolist()

expert_texts = experts["Profile_Text"].tolist()


# -----------------------------------------------------------
# STEP 4: Load embedding model
# -----------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-mpnet-base-v2")


# -----------------------------------------------------------
# STEP 5: Generate embeddings
# -----------------------------------------------------------

print("\nGenerating paper embeddings...")

paper_embeddings = model.encode(paper_texts)

print("Paper embeddings shape:", paper_embeddings.shape)


print("\nGenerating expert embeddings...")

expert_embeddings = model.encode(expert_texts)

print("Expert embeddings shape:", expert_embeddings.shape)


# -----------------------------------------------------------
# STEP 6: Normalize embeddings
# -----------------------------------------------------------

paper_embeddings = normalize(paper_embeddings)
expert_embeddings = normalize(expert_embeddings)


# -----------------------------------------------------------
# STEP 7: Compute similarity matrix
# -----------------------------------------------------------

print("\nComputing paper-expert similarity...")

similarity_matrix = cosine_similarity(paper_embeddings, expert_embeddings)

print("Similarity matrix shape:", similarity_matrix.shape)


# -----------------------------------------------------------
# STEP 8: Domain-aware scoring
# -----------------------------------------------------------

print("\nApplying domain-aware ranking...")

domain_bonus = 0.15

adjusted_similarity = similarity_matrix.copy()

for i in range(len(papers)):

    paper_domain = papers.iloc[i]["Domain_tag"]

    for j in range(len(experts)):

        expert_domain = experts.iloc[j]["Domain"]

        if paper_domain == expert_domain:
            adjusted_similarity[i][j] += domain_bonus


# -----------------------------------------------------------
# STEP 9: Example paper query
# -----------------------------------------------------------

paper_index = 2

print("\nQuery Paper:\n")
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
# STEP 10: Top-K evaluation
# -----------------------------------------------------------

# print("\nEvaluating Top-K accuracy...")

# top_k = 3
# correct = 0

# for i in range(len(papers)):

#     scores = adjusted_similarity[i]

#     top_experts = scores.argsort()[::-1][:top_k]

#     paper_domain = papers.iloc[i]["Domain_tag"]

#     expert_domains = experts.iloc[top_experts]["Domain"].values

#     if paper_domain in expert_domains:
#         correct += 1

# accuracy = correct / len(papers)

# print(f"Top-{top_k} Domain Accuracy:", accuracy)

# -----------------------------------------------------------
# STEP 10: NDCG Evaluation
# -----------------------------------------------------------

from sklearn.metrics import ndcg_score

print("\nCalculating NDCG@5...")

k = 5
ndcg_scores = [] 

for i in range(len(papers)):

    paper_domain = papers.iloc[i]["Domain_tag"]

    # relevance vector (1 if expert domain matches)
    relevance = []

    for j in range(len(experts)):

        if experts.iloc[j]["Domain"] == paper_domain:
            relevance.append(1)
        else:
            relevance.append(0)

    relevance = np.array(relevance).reshape(1, -1)

    # predicted ranking scores
    scores = adjusted_similarity[i].reshape(1, -1)

    ndcg = ndcg_score(relevance, scores, k=k)

    ndcg_scores.append(ndcg)

print("Average NDCG@5:", np.mean(ndcg_scores))
# -----------------------------------------------------------
# STEP 11: PCA Visualization with Domain Clustering
# -----------------------------------------------------------

print("\nCreating PCA visualization with domain clustering...")

combined_embeddings = np.vstack((paper_embeddings, expert_embeddings))

pca = PCA(n_components=2)

reduced = pca.fit_transform(combined_embeddings)

paper_points = reduced[:len(papers)]
expert_points = reduced[len(papers):]

plt.figure(figsize=(10,7))


# -----------------------------------------------------------
# Plot papers (light grey for background context)
# -----------------------------------------------------------

plt.scatter(
    paper_points[:,0],
    paper_points[:,1],
    alpha=0.3,
    color="grey",
    label="Papers"
)


# -----------------------------------------------------------
# Plot experts by domain
# -----------------------------------------------------------

domain_colors = {
    "AI": "red",
    "Security": "blue",
    "Hardware": "green",
    "Medicine": "purple"
}

for domain in experts["Domain"].unique():

    domain_indices = experts[experts["Domain"] == domain].index

    plt.scatter(
        expert_points[domain_indices,0],
        expert_points[domain_indices,1],
        s=220,
        marker="X",
        color=domain_colors.get(domain, "black"),
        label=f"{domain} Experts"
    )


# -----------------------------------------------------------
# Plot settings
# -----------------------------------------------------------

plt.title("PCA Visualization of Paper–Expert Embedding Space")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.legend()

plt.grid(alpha=0.2)

plt.show()