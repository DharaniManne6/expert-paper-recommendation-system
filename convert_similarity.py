# ===========================================================
# STEP 1: Import required libraries
# ===========================================================

import warnings
warnings.filterwarnings("ignore")  

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import ndcg_score



# ===========================================================
# STEP 2: Load the dataset
# ===========================================================

# Load the papers dataset from Excel file
papers = pd.read_excel("Dataset_later\Paperss.xlsx")

print("\nFirst few rows of dataset:\n")
print(papers.head())

print("\nDataset Columns:\n")
print(papers.columns)



# ===========================================================
# STEP 3: Generate embeddings from paper abstracts
# ===========================================================

print("\nGenerating embeddings for paper abstracts...\n")

# pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract abstract text
texts = papers["Abstract"].astype(str).tolist()

# Convert abstracts into embeddings
paper_embeddings = model.encode(texts)

print("Embedding matrix shape:", paper_embeddings.shape)



# ===========================================================
# STEP 4: Compute similarity between papers
# ===========================================================

print("\nComputing similarity matrix...\n")

# Cosine similarity measures semantic similarity
similarity_matrix = cosine_similarity(paper_embeddings)

print("Similarity matrix shape:", similarity_matrix.shape)



# ===========================================================
# STEP 5: Example recommendation (Top similar papers)
# ===========================================================

print("\nExample Recommendation (Top 5 Similar Papers):\n")

# Choose the first paper as query
paper_index = 2

# Get similarity scores
similar_scores = list(enumerate(similarity_matrix[paper_index]))

# Sort papers based on similarity
similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)

# Print top 5 similar papers
for i in similar_scores[1:11]:
    print("-", papers.iloc[i[0]]["Title"])


# ===========================================================
# STEP 6: Dimensionality reduction using PCA
# ===========================================================

print("\nReducing embedding dimensions for visualization...\n")

pca = PCA(n_components=2)

reduced_embeddings = pca.fit_transform(paper_embeddings)



# ===========================================================
# STEP 7: Basic embedding visualization
# ===========================================================

print("Saving PCA visualization...")

plt.figure(figsize=(8,6))

# Scatter plot of all paper embeddings
plt.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1])

plt.title("Paper Embedding Visualization (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Save plot to file
plt.savefig("paper_embedding_visualization.png", dpi=300)

plt.show()



# ===========================================================
# STEP 8: Visualization with domain clustering
# ===========================================================

print("Saving domain clustering visualization...")

plt.figure(figsize=(8,6))

# Color points by research domain
sns.scatterplot(
    x=reduced_embeddings[:,0],
    y=reduced_embeddings[:,1],
    hue=papers["Domain_tag"],
    palette="tab10"
)

plt.title("Paper Embedding Clusters by Domain")

# Save plot
plt.savefig("paper_domain_clusters.png", dpi=300)

plt.show()



# ===========================================================
# STEP 9: Evaluate ranking quality using NDCG
# ===========================================================

print("\nCalculating NDCG score...\n")

# Lists for storing ground truth relevance and predicted scores
y_true = []
y_scores = []

for i in range(len(papers)):

    # Ground truth relevance:
    # Papers from the same domain are considered relevant
    true_relevance = (papers["Domain_tag"] == papers.iloc[i]["Domain_tag"]).astype(int)

    # Predicted relevance = similarity scores
    scores = similarity_matrix[i]

    y_true.append(true_relevance)
    y_scores.append(scores)

# Compute NDCG score
ndcg = ndcg_score(y_true, y_scores)

print("Final NDCG Score:", ndcg)



# ===========================================================
# STEP 10: Save similarity matrix (optional analysis)
# ===========================================================

print("\nSaving similarity matrix to CSV...")

similarity_df = pd.DataFrame(similarity_matrix)

similarity_df.to_csv("paper_similarity_matrix.csv", index=False)

print("Process completed successfully.")