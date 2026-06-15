# # -----------------------------------------------------------
# # PAPER -> EXPERT RECOMMENDATION SYSTEM
# # Research Pipeline Implementation
# # Models Compared:
# # 1. TF-IDF
# # 2. Sentence-BERT
# # 3. SciBERT
# #
# # Evaluation:
# # NDCG@5 with graded relevance
# # -----------------------------------------------------------

# import pandas as pd
# import numpy as np
# import re
# import warnings
# warnings.filterwarnings("ignore")

# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import ndcg_score
# from sklearn.preprocessing import normalize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA

# import matplotlib.pyplot as plt

# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModel
# import torch


# # -----------------------------------------------------------
# # STEP 1: LOAD DATASETS
# # -----------------------------------------------------------

# DATA_PATH = "Dataset_later"

# papers = pd.read_excel(f"{DATA_PATH}/Papers.xlsx")
# experts = pd.read_excel(f"{DATA_PATH}/Experts_Shuffled.xlsx")

# print("Papers:", len(papers))
# print("Experts:", len(experts))


# # -----------------------------------------------------------
# # STEP 2: TEXT PREPROCESSING
# # -----------------------------------------------------------

# def clean_text(text):

#     text = str(text)
#     text = re.sub(r"\s+", " ", text)

#     return text.strip()


# papers["Title"] = papers["Title"].apply(clean_text)
# papers["Abstract"] = papers["Abstract"].apply(clean_text)

# experts["Profile_Text"] = experts["Profile_Text"].apply(clean_text)


# # -----------------------------------------------------------
# # STEP 3: CREATE DOCUMENT TEXT
# # -----------------------------------------------------------

# paper_texts = (papers["Title"] + ". " + papers["Abstract"]).tolist()
# expert_texts = experts["Profile_Text"].tolist()


# # -----------------------------------------------------------
# # STEP 4: DEFINE GRADED RELEVANCE
# # -----------------------------------------------------------

# def compute_relevance(paper_domain, expert_domain):

#     if paper_domain == expert_domain:
#         return 3

#     # related domains (example relationship)
#     if (paper_domain == "AI" and expert_domain == "Security") or \
#        (paper_domain == "Security" and expert_domain == "AI"):
#         return 1

#     return 0


# # -----------------------------------------------------------
# # STEP 5: EVALUATION FUNCTION
# # -----------------------------------------------------------

# def evaluate_ndcg(similarity_matrix, papers, experts, k=5):

#     domain_bonus = 0.10
#     adjusted_similarity = similarity_matrix.copy()

#     for i in range(len(papers)):

#         paper_domain = papers.iloc[i]["Domain_tag"]

#         for j in range(len(experts)):

#             expert_domain = experts.iloc[j]["Domain"]

#             if paper_domain == expert_domain:
#                 adjusted_similarity[i][j] += domain_bonus

#     ndcg_scores = []

#     for i in range(len(papers)):

#         paper_domain = papers.iloc[i]["Domain_tag"]

#         relevance = []

#         for j in range(len(experts)):

#             expert_domain = experts.iloc[j]["Domain"]

#             relevance.append(
#                 compute_relevance(paper_domain, expert_domain)
#             )

#         relevance = np.array(relevance).reshape(1, -1)

#         scores = adjusted_similarity[i].reshape(1, -1)

#         ndcg = ndcg_score(relevance, scores, k=k)

#         ndcg_scores.append(ndcg)

#     return np.mean(ndcg_scores), adjusted_similarity


# # ===========================================================
# # MODEL 1 — TF-IDF
# # ===========================================================

# print("\nRunning TF-IDF model")

# tfidf = TfidfVectorizer(stop_words="english")

# combined = paper_texts + expert_texts

# tfidf_matrix = tfidf.fit_transform(combined)

# paper_vectors = tfidf_matrix[:len(papers)]
# expert_vectors = tfidf_matrix[len(papers):]

# similarity_matrix = cosine_similarity(paper_vectors, expert_vectors)

# tfidf_ndcg, _ = evaluate_ndcg(similarity_matrix, papers, experts)

# print("TF-IDF NDCG@5:", tfidf_ndcg)


# # ===========================================================
# # MODEL 2 — Sentence-BERT
# # ===========================================================

# print("\nRunning Sentence-BERT")

# # sbert = SentenceTransformer("all-mpnet-base-v2")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")

# paper_emb = sbert.encode(paper_texts)
# expert_emb = sbert.encode(expert_texts)

# paper_emb = normalize(paper_emb)
# expert_emb = normalize(expert_emb)

# similarity_matrix = cosine_similarity(paper_emb, expert_emb)

# sbert_ndcg, adjusted_similarity = evaluate_ndcg(similarity_matrix, papers, experts)

# print("Sentence-BERT NDCG@5:", sbert_ndcg)


# # ===========================================================
# # MODEL 3 — SciBERT
# # ===========================================================

# print("\nRunning SciBERT")

# tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
# scibert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


# def get_embedding(text):

#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=512
#     )

#     with torch.no_grad():

#         outputs = scibert(**inputs)

#     embedding = outputs.last_hidden_state.mean(dim=1)

#     return embedding.numpy()[0]


# print("Encoding papers")

# paper_scibert = np.array([get_embedding(t) for t in paper_texts])

# print("Encoding experts")

# expert_scibert = np.array([get_embedding(t) for t in expert_texts])

# paper_scibert = normalize(paper_scibert)
# expert_scibert = normalize(expert_scibert)

# similarity_matrix = cosine_similarity(paper_scibert, expert_scibert)

# scibert_ndcg, _ = evaluate_ndcg(similarity_matrix, papers, experts)

# print("SciBERT NDCG@5:", scibert_ndcg)


# # -----------------------------------------------------------
# # MODEL COMPARISON RESULTS
# # -----------------------------------------------------------

# print("\n========== FINAL RESULTS ==========")

# print("TF-IDF:", tfidf_ndcg)
# print("Sentence-BERT:", sbert_ndcg)
# print("SciBERT:", scibert_ndcg)


# # -----------------------------------------------------------
# # PCA VISUALIZATION
# # -----------------------------------------------------------

# print("\nGenerating PCA visualization")

# combined_embeddings = np.vstack((paper_emb, expert_emb))

# pca = PCA(n_components=2)

# reduced = pca.fit_transform(combined_embeddings)

# paper_points = reduced[:len(papers)]
# expert_points = reduced[len(papers):]

# plt.figure(figsize=(10,7))

# plt.scatter(
#     paper_points[:,0],
#     paper_points[:,1],
#     alpha=0.3,
#     color="gray",
#     label="Papers"
# )

# colors = {
#     "AI":"red",
#     "Security":"blue",
#     "Hardware":"green",
#     "Medicine":"purple"
# }

# for domain in experts["Domain"].unique():

#     idx = experts[experts["Domain"] == domain].index

#     plt.scatter(
#         expert_points[idx,0],
#         expert_points[idx,1],
#         s=200,
#         marker="X",
#         color=colors.get(domain,"black"),
#         label=domain+" Experts"
#     )

# plt.title("Embedding Space Visualization (PCA)")
# plt.xlabel("Component 1")
# plt.ylabel("Component 2")

# plt.legend()
# plt.grid(alpha=0.2)
# plt.savefig("paper_expert_cluster.png", dpi=300)

# for i in range(5):

#     print("\n-----------------------------------")
#     print("Paper:", papers.iloc[i]["Title"])
#     print("Domain:", papers.iloc[i]["Domain_tag"])

#     scores = adjusted_similarity[i]
#     top_experts = scores.argsort()[::-1][:5]

#     print("Top Experts:")

#     for idx in top_experts:

#         print(
#             experts.iloc[idx]["Expert_ID"],
#             "| Domain:", experts.iloc[idx]["Domain"]
#         )



# # -----------------------------------------------------------
# # EXPORT RECOMMENDATIONS FOR MANUAL VALIDATION
# # -----------------------------------------------------------

# top_k = 5
# results = []

# for i in range(len(papers)):

#     paper_title = papers.iloc[i]["Title"]
#     paper_domain = papers.iloc[i]["Domain_tag"]

#     scores = adjusted_similarity[i]

#     top_experts = scores.argsort()[::-1][:top_k]

#     for rank, idx in enumerate(top_experts):

#         results.append({
#             "Paper_Title": paper_title,
#             "Paper_Domain": paper_domain,
#             "Rank": rank+1,
#             "Expert_ID": experts.iloc[idx]["Expert_ID"],
#             "Expert_Domain": experts.iloc[idx]["Domain"],
#             "Score": scores[idx]
#         })


# results_df = pd.DataFrame(results)

# results_df.to_excel("manual_recommendation_check.xlsx", index=False)

# print("Manual evaluation file saved.")

# # -----------------------------------------------------------
# # EXPERT -> TOP 5 PAPERS RECOMMENDATION
# # -----------------------------------------------------------

# print("\n==============================")
# print("EXPERT -> TOP 5 PAPERS EXAMPLE")
# print("==============================")

# top_k = 5

# # Use the BEST model (Sentence-BERT recommended)
# # similarity_matrix already computed for SBERT above

# expert_paper_similarity = similarity_matrix.T   # transpose

# for i in range(3):  # show for first 3 experts

#     print("\n-----------------------------------")
#     print("Expert:", experts.iloc[i]["Expert_ID"])
#     print("Domain:", experts.iloc[i]["Domain"])

#     scores = expert_paper_similarity[i]

#     top_papers = scores.argsort()[::-1][:top_k]

#     print("Top 5 Recommended Papers:")

#     for rank, idx in enumerate(top_papers):

#         print(
#             f"Rank {rank+1}:",
#             papers.iloc[idx]["Title"],
#             "| Domain:", papers.iloc[idx]["Domain_tag"]
#         )

# # -----------------------------------------------------------
# # EXPERT -> TOP 5 PAPERS RECOMMENDATION
# # -----------------------------------------------------------

# print("\n==============================")
# print("EXPERT -> TOP 5 PAPERS EXAMPLE")
# print("==============================")

# top_k = 5

# # Use the BEST model (Sentence-BERT recommended)
# # similarity_matrix already computed for SBERT above

# expert_paper_similarity = similarity_matrix.T   # transpose

# for i in range(3):  # show for first 3 experts

#     print("\n-----------------------------------")
#     print("Expert:", experts.iloc[i]["Expert_ID"])
#     print("Domain:", experts.iloc[i]["Domain"])

#     scores = expert_paper_similarity[i]

#     top_papers = scores.argsort()[::-1][:top_k]

#     print("Top 5 Recommended Papers:")

#     for rank, idx in enumerate(top_papers):

#         print(
#             f"Rank {rank+1}:",
#             papers.iloc[idx]["Title"],
#             "| Domain:", papers.iloc[idx]["Domain_tag"]
#         )


# -----------------------------------------------------------
# PAPER -> EXPERT RECOMMENDATION SYSTEM
# Faster + safer version
# Models Compared:
# 1. TF-IDF
# 2. Sentence-BERT
# 3. SciBERT
# Evaluation: NDCG@5
# -----------------------------------------------------------

# import os
# import re
# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch

# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import ndcg_score
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModel

# # -----------------------------------------------------------
# # CONFIG
# # -----------------------------------------------------------

# DATA_PATH = "Dataset_later"
# PAPERS_FILE = f"{DATA_PATH}/Papers.xlsx"
# EXPERTS_FILE = f"{DATA_PATH}/Experts_Shuffled.xlsx"

# TOP_K = 5
# RUN_PCA = False              # set True later if needed
# RUN_SCIBERT = True           # set False if you want to skip SciBERT
# SBERT_BATCH_SIZE = 32
# SCIBERT_BATCH_SIZE = 8
# SCIBERT_MAX_LENGTH = 256

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -----------------------------------------------------------
# # STEP 1: LOAD DATASETS
# # -----------------------------------------------------------

# papers = pd.read_excel(PAPERS_FILE)
# experts = pd.read_excel(EXPERTS_FILE)

# print("Papers:", len(papers))
# print("Experts:", len(experts))
# print("Using device:", DEVICE)

# # -----------------------------------------------------------
# # STEP 2: TEXT PREPROCESSING
# # -----------------------------------------------------------

# def clean_text(text):
#     text = "" if pd.isna(text) else str(text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

# papers["Title"] = papers["Title"].apply(clean_text)
# papers["Abstract"] = papers["Abstract"].apply(clean_text)
# experts["Profile_Text"] = experts["Profile_Text"].apply(clean_text)

# # -----------------------------------------------------------
# # STEP 3: CREATE DOCUMENT TEXT
# # -----------------------------------------------------------

# paper_texts = (papers["Title"] + ". " + papers["Abstract"]).tolist()
# expert_texts = experts["Profile_Text"].tolist()

# # -----------------------------------------------------------
# # STEP 4: DEFINE GRADED RELEVANCE
# # -----------------------------------------------------------

# def compute_relevance(paper_domain, expert_domain):
#     if paper_domain == expert_domain:
#         return 3

#     if (paper_domain == "AI" and expert_domain == "Security") or \
#        (paper_domain == "Security" and expert_domain == "AI"):
#         return 1

#     return 0

# # -----------------------------------------------------------
# # STEP 5: EVALUATION FUNCTION
# # -----------------------------------------------------------

# def evaluate_ndcg(similarity_matrix, papers, experts, k=5):
#     domain_bonus = 0.10
#     adjusted_similarity = similarity_matrix.copy()

#     for i in range(len(papers)):
#         paper_domain = papers.iloc[i]["Domain_tag"]

#         for j in range(len(experts)):
#             expert_domain = experts.iloc[j]["Domain"]

#             if paper_domain == expert_domain:
#                 adjusted_similarity[i][j] += domain_bonus

#     ndcg_scores = []

#     for i in range(len(papers)):
#         paper_domain = papers.iloc[i]["Domain_tag"]
#         relevance = []

#         for j in range(len(experts)):
#             expert_domain = experts.iloc[j]["Domain"]
#             relevance.append(compute_relevance(paper_domain, expert_domain))

#         relevance = np.array(relevance).reshape(1, -1)
#         scores = adjusted_similarity[i].reshape(1, -1)

#         ndcg = ndcg_score(relevance, scores, k=k)
#         ndcg_scores.append(ndcg)

#     return float(np.mean(ndcg_scores)), adjusted_similarity

# # -----------------------------------------------------------
# # STEP 6: SAVE MANUAL CHECK FILE
# # -----------------------------------------------------------

# def export_manual_check(adjusted_similarity, model_name):
#     results = []

#     for i in range(len(papers)):
#         paper_title = papers.iloc[i]["Title"]
#         paper_domain = papers.iloc[i]["Domain_tag"]

#         scores = adjusted_similarity[i]
#         top_experts = scores.argsort()[::-1][:TOP_K]

#         for rank, idx in enumerate(top_experts, start=1):
#             results.append({
#                 "Model": model_name,
#                 "Paper_Title": paper_title,
#                 "Paper_Domain": paper_domain,
#                 "Rank": rank,
#                 "Expert_ID": experts.iloc[idx]["Expert_ID"],
#                 "Expert_Domain": experts.iloc[idx]["Domain"],
#                 "Score": float(scores[idx])
#             })

#     out_file = f"manual_recommendation_check_{model_name}.xlsx"
#     pd.DataFrame(results).to_excel(out_file, index=False)
#     print(f"Saved: {out_file}")

# # -----------------------------------------------------------
# # STEP 7: PRINT EXPERT -> TOP PAPERS
# # -----------------------------------------------------------

# def print_expert_top_papers(similarity_matrix, model_name):
#     print(f"\n==============================")
#     print(f"EXPERT -> TOP {TOP_K} PAPERS ({model_name})")
#     print("==============================")

#     expert_paper_similarity = similarity_matrix.T

#     for i in range(min(3, len(experts))):
#         print("\n-----------------------------------")
#         print("Expert:", experts.iloc[i]["Expert_ID"])
#         print("Domain:", experts.iloc[i]["Domain"])

#         scores = expert_paper_similarity[i]
#         top_papers = scores.argsort()[::-1][:TOP_K]

#         print(f"Top {TOP_K} Recommended Papers:")
#         for rank, idx in enumerate(top_papers, start=1):
#             print(
#                 f"Rank {rank}:",
#                 papers.iloc[idx]["Title"],
#                 "| Domain:", papers.iloc[idx]["Domain_tag"]
#             )

# # -----------------------------------------------------------
# # MODEL 1 — TF-IDF
# # -----------------------------------------------------------

# print("\nRunning TF-IDF model...")

# tfidf = TfidfVectorizer(
#     stop_words="english",
#     max_features=30000
# )

# combined = paper_texts + expert_texts
# tfidf_matrix = tfidf.fit_transform(combined)

# paper_vectors = tfidf_matrix[:len(papers)]
# expert_vectors = tfidf_matrix[len(papers):]

# similarity_matrix_tfidf = cosine_similarity(paper_vectors, expert_vectors)

# tfidf_ndcg, adjusted_similarity_tfidf = evaluate_ndcg(
#     similarity_matrix_tfidf, papers, experts, k=TOP_K
# )

# print(f"TF-IDF NDCG@{TOP_K}: {tfidf_ndcg}")

# export_manual_check(adjusted_similarity_tfidf, "TFIDF")

# # -----------------------------------------------------------
# # MODEL 2 — Sentence-BERT
# # -----------------------------------------------------------

# print("\nRunning Sentence-BERT...")

# try:
#     sbert = SentenceTransformer("all-MiniLM-L6-v2")

#     print("Encoding papers with Sentence-BERT...")
#     paper_emb_sbert = sbert.encode(
#         paper_texts,
#         batch_size=SBERT_BATCH_SIZE,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         normalize_embeddings=True
#     )

#     print("Encoding experts with Sentence-BERT...")
#     expert_emb_sbert = sbert.encode(
#         expert_texts,
#         batch_size=16,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         normalize_embeddings=True
#     )

#     similarity_matrix_sbert = cosine_similarity(paper_emb_sbert, expert_emb_sbert)

#     sbert_ndcg, adjusted_similarity_sbert = evaluate_ndcg(
#         similarity_matrix_sbert, papers, experts, k=TOP_K
#     )

#     print(f"Sentence-BERT NDCG@{TOP_K}: {sbert_ndcg}")

#     export_manual_check(adjusted_similarity_sbert, "SBERT")
#     print_expert_top_papers(similarity_matrix_sbert, "SBERT")

# except Exception as e:
#     sbert_ndcg = None
#     similarity_matrix_sbert = None
#     adjusted_similarity_sbert = None
#     print("\nSentence-BERT failed.")
#     print("Reason:", e)

# # -----------------------------------------------------------
# # MODEL 3 — SciBERT
# # -----------------------------------------------------------

# scibert_ndcg = None
# similarity_matrix_scibert = None
# adjusted_similarity_scibert = None

# def mean_pooling(last_hidden_state, attention_mask):
#     mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#     masked_embeddings = last_hidden_state * mask
#     summed = masked_embeddings.sum(dim=1)
#     counts = mask.sum(dim=1).clamp(min=1e-9)
#     return summed / counts

# def encode_scibert_batch(texts, tokenizer, model, batch_size=8, max_length=256, label="texts"):
#     all_embeddings = []

#     with torch.no_grad():
#         for start in range(0, len(texts), batch_size):
#             batch = texts[start:start + batch_size]

#             inputs = tokenizer(
#                 batch,
#                 return_tensors="pt",
#                 truncation=True,
#                 padding=True,
#                 max_length=max_length
#             )

#             inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
#             outputs = model(**inputs)

#             embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
#             embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

#             all_embeddings.append(embeddings.cpu().numpy())

#             print(f"SciBERT encoded {min(start + batch_size, len(texts))}/{len(texts)} {label}")

#     return np.vstack(all_embeddings)

# if RUN_SCIBERT:
#     print("\nRunning SciBERT...")

#     try:
#         # First try normal online loading
#         tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
#         scibert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(DEVICE)
#         scibert.eval()

#         print("Encoding papers with SciBERT...")
#         paper_emb_scibert = encode_scibert_batch(
#             paper_texts,
#             tokenizer,
#             scibert,
#             batch_size=SCIBERT_BATCH_SIZE,
#             max_length=SCIBERT_MAX_LENGTH,
#             label="papers"
#         )

#         print("Encoding experts with SciBERT...")
#         expert_emb_scibert = encode_scibert_batch(
#             expert_texts,
#             tokenizer,
#             scibert,
#             batch_size=SCIBERT_BATCH_SIZE,
#             max_length=SCIBERT_MAX_LENGTH,
#             label="experts"
#         )

#         similarity_matrix_scibert = cosine_similarity(paper_emb_scibert, expert_emb_scibert)

#         scibert_ndcg, adjusted_similarity_scibert = evaluate_ndcg(
#             similarity_matrix_scibert, papers, experts, k=TOP_K
#         )

#         print(f"SciBERT NDCG@{TOP_K}: {scibert_ndcg}")

#         export_manual_check(adjusted_similarity_scibert, "SCIBERT")
#         print_expert_top_papers(similarity_matrix_scibert, "SCIBERT")

#     except Exception as e:
#         print("\nSciBERT failed.")
#         print("Reason:", e)
#         print("Skipping SciBERT and continuing.")

# # -----------------------------------------------------------
# # FINAL RESULTS
# # -----------------------------------------------------------

# print("\n========== FINAL RESULTS ==========")
# print("TF-IDF:", tfidf_ndcg)
# print("Sentence-BERT:", sbert_ndcg if sbert_ndcg is not None else "FAILED")
# print("SciBERT:", scibert_ndcg if scibert_ndcg is not None else "FAILED")

# summary_rows = [
#     {"Model": "TF-IDF", "NDCG@5": tfidf_ndcg},
#     {"Model": "Sentence-BERT", "NDCG@5": sbert_ndcg},
#     {"Model": "SciBERT", "NDCG@5": scibert_ndcg},
# ]

# summary_df = pd.DataFrame(summary_rows)
# summary_df.to_excel("model_ndcg_summary.xlsx", index=False)
# print("Saved: model_ndcg_summary.xlsx")

# # -----------------------------------------------------------
# # OPTIONAL PCA
# # -----------------------------------------------------------

# if RUN_PCA and similarity_matrix_sbert is not None:
#     print("\nGenerating PCA visualization...")

#     combined_embeddings = np.vstack((paper_emb_sbert, expert_emb_sbert))
#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(combined_embeddings)

#     paper_points = reduced[:len(papers)]
#     expert_points = reduced[len(papers):]

#     plt.figure(figsize=(10, 7))

#     plt.scatter(
#         paper_points[:, 0],
#         paper_points[:, 1],
#         alpha=0.3,
#         color="gray",
#         label="Papers"
#     )

#     colors = {
#         "AI": "red",
#         "Security": "blue",
#         "Hardware": "green",
#         "Medicine": "purple"
#     }

#     for domain in experts["Domain"].unique():
#         idx = experts[experts["Domain"] == domain].index

#         plt.scatter(
#             expert_points[idx, 0],
#             expert_points[idx, 1],
#             s=200,
#             marker="X",
#             color=colors.get(domain, "black"),
#             label=domain + " Experts"
#         )

#     plt.title("Embedding Space Visualization (PCA)")
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.legend()
#     plt.grid(alpha=0.2)
#     plt.savefig("paper_expert_cluster.png", dpi=300)
#     plt.show()




# -----------------------------------------------------------
# PAPER -> EXPERT RECOMMENDATION SYSTEM
# Organized + faster workflow with caching
# Models:
# 1. TF-IDF
# 2. Sentence-BERT
# 3. SciBERT
# Evaluation: NDCG@5
# -----------------------------------------------------------

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

DATA_PATH = "Dataset_later"
PAPERS_FILE = os.path.join(DATA_PATH, "Papers.xlsx")
EXPERTS_FILE = os.path.join(DATA_PATH, "Experts_Shuffled.xlsx")

CACHE_DIR = "cache_embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)

TOP_K = 5
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"

SBERT_PAPER_BATCH = 32
SBERT_EXPERT_BATCH = 16

SCIBERT_PAPER_BATCH = 8
SCIBERT_EXPERT_BATCH = 8
SCIBERT_MAX_LENGTH = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set to False if you want to skip a model
RUN_TFIDF = True
RUN_SBERT = True
RUN_SCIBERT = True

# If True, load cached embeddings when available
USE_CACHE = True

# -----------------------------------------------------------
# UTILS
# -----------------------------------------------------------

def clean_text(text):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def compute_relevance(paper_domain, expert_domain):
    if paper_domain == expert_domain:
        return 3

    if (paper_domain == "AI" and expert_domain == "Security") or \
       (paper_domain == "Security" and expert_domain == "AI"):
        return 1

    return 0

def evaluate_ndcg(similarity_matrix, papers_df, experts_df, k=5):
    adjusted_similarity = similarity_matrix.copy()
    domain_bonus = 0.10

    for i in range(len(papers_df)):
        p_domain = papers_df.iloc[i]["Domain_tag"]

        for j in range(len(experts_df)):
            e_domain = experts_df.iloc[j]["Domain"]
            if p_domain == e_domain:
                adjusted_similarity[i][j] += domain_bonus

    ndcg_scores = []

    for i in range(len(papers_df)):
        p_domain = papers_df.iloc[i]["Domain_tag"]

        relevance = [
            compute_relevance(p_domain, experts_df.iloc[j]["Domain"])
            for j in range(len(experts_df))
        ]

        y_true = np.array(relevance).reshape(1, -1)
        y_score = adjusted_similarity[i].reshape(1, -1)

        ndcg_scores.append(ndcg_score(y_true, y_score, k=k))

    return float(np.mean(ndcg_scores)), adjusted_similarity

def export_manual_check(adjusted_similarity, papers_df, experts_df, model_name, top_k=5):
    rows = []

    for i in range(len(papers_df)):
        scores = adjusted_similarity[i]
        top_experts = scores.argsort()[::-1][:top_k]

        for rank, idx in enumerate(top_experts, start=1):
            rows.append({
                "Model": model_name,
                "Paper_Title": papers_df.iloc[i]["Title"],
                "Paper_Domain": papers_df.iloc[i]["Domain_tag"],
                "Rank": rank,
                "Expert_ID": experts_df.iloc[idx]["Expert_ID"],
                "Expert_Domain": experts_df.iloc[idx]["Domain"],
                "Score": float(scores[idx])
            })

    out_file = f"manual_recommendation_check_{model_name}.xlsx"
    pd.DataFrame(rows).to_excel(out_file, index=False)
    print(f"Saved: {out_file}")

def print_expert_top_papers(similarity_matrix, papers_df, experts_df, model_name, top_k=5):
    print(f"\n==============================")
    print(f"EXPERT -> TOP {top_k} PAPERS ({model_name})")
    print("==============================")

    expert_paper_similarity = similarity_matrix.T

    for i in range(min(3, len(experts_df))):
        print("\n-----------------------------------")
        print("Expert:", experts_df.iloc[i]["Expert_ID"])
        print("Domain:", experts_df.iloc[i]["Domain"])

        scores = expert_paper_similarity[i]
        top_papers = scores.argsort()[::-1][:top_k]

        print(f"Top {top_k} Recommended Papers:")
        for rank, idx in enumerate(top_papers, start=1):
            print(
                f"Rank {rank}:",
                papers_df.iloc[idx]["Title"],
                "| Domain:", papers_df.iloc[idx]["Domain_tag"]
            )

def save_npy(array, path):
    np.save(path, array)

def load_npy(path):
    return np.load(path)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------

papers = pd.read_excel(PAPERS_FILE)
experts = pd.read_excel(EXPERTS_FILE)

papers["Title"] = papers["Title"].apply(clean_text)
papers["Abstract"] = papers["Abstract"].apply(clean_text)
experts["Profile_Text"] = experts["Profile_Text"].apply(clean_text)

paper_texts = (papers["Title"] + ". " + papers["Abstract"]).tolist()
expert_texts = experts["Profile_Text"].tolist()

print("Papers:", len(papers))
print("Experts:", len(experts))
print("Using device:", DEVICE)

# -----------------------------------------------------------
# MODEL 1: TF-IDF
# -----------------------------------------------------------

tfidf_ndcg = None
if RUN_TFIDF:
    print("\nRunning TF-IDF model...")

    tfidf = TfidfVectorizer(stop_words="english", max_features=30000)
    combined = paper_texts + expert_texts
    tfidf_matrix = tfidf.fit_transform(combined)

    paper_vectors = tfidf_matrix[:len(papers)]
    expert_vectors = tfidf_matrix[len(papers):]

    similarity_matrix_tfidf = cosine_similarity(paper_vectors, expert_vectors)
    tfidf_ndcg, adjusted_tfidf = evaluate_ndcg(similarity_matrix_tfidf, papers, experts, k=TOP_K)

    print(f"TF-IDF NDCG@{TOP_K}: {tfidf_ndcg}")
    export_manual_check(adjusted_tfidf, papers, experts, "TFIDF", TOP_K)

# -----------------------------------------------------------
# MODEL 2: Sentence-BERT
# -----------------------------------------------------------

sbert_ndcg = None
if RUN_SBERT:
    print("\nRunning Sentence-BERT...")

    sbert_paper_cache = os.path.join(CACHE_DIR, "paper_emb_sbert.npy")
    sbert_expert_cache = os.path.join(CACHE_DIR, "expert_emb_sbert.npy")

    if USE_CACHE and os.path.exists(sbert_paper_cache) and os.path.exists(sbert_expert_cache):
        print("Loading cached Sentence-BERT embeddings...")
        paper_emb_sbert = load_npy(sbert_paper_cache)
        expert_emb_sbert = load_npy(sbert_expert_cache)
    else:
        model = SentenceTransformer(SBERT_MODEL_NAME)

        print("Encoding papers with Sentence-BERT...")
        paper_emb_sbert = model.encode(
            paper_texts,
            batch_size=SBERT_PAPER_BATCH,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print("Encoding experts with Sentence-BERT...")
        expert_emb_sbert = model.encode(
            expert_texts,
            batch_size=SBERT_EXPERT_BATCH,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        save_npy(paper_emb_sbert, sbert_paper_cache)
        save_npy(expert_emb_sbert, sbert_expert_cache)
        print("Saved Sentence-BERT embeddings to cache.")

    similarity_matrix_sbert = cosine_similarity(paper_emb_sbert, expert_emb_sbert)
    sbert_ndcg, adjusted_sbert = evaluate_ndcg(similarity_matrix_sbert, papers, experts, k=TOP_K)

    print(f"Sentence-BERT NDCG@{TOP_K}: {sbert_ndcg}")
    export_manual_check(adjusted_sbert, papers, experts, "SBERT", TOP_K)
    print_expert_top_papers(similarity_matrix_sbert, papers, experts, "SBERT", TOP_K)

# -----------------------------------------------------------
# MODEL 3: SciBERT
# -----------------------------------------------------------

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def encode_scibert_batch(texts, tokenizer, model, batch_size=8, max_length=256, label="texts", print_every=25):
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_num, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[start:start + batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)

            embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

            if batch_num % print_every == 0 or batch_num == total_batches:
                done = min(start + batch_size, len(texts))
                print(f"SciBERT progress: {done}/{len(texts)} {label}")

    return np.vstack(all_embeddings)

scibert_ndcg = None
if RUN_SCIBERT:
    print("\nRunning SciBERT...")

    scibert_paper_cache = os.path.join(CACHE_DIR, "paper_emb_scibert.npy")
    scibert_expert_cache = os.path.join(CACHE_DIR, "expert_emb_scibert.npy")

    if USE_CACHE and os.path.exists(scibert_paper_cache) and os.path.exists(scibert_expert_cache):
        print("Loading cached SciBERT embeddings...")
        paper_emb_scibert = load_npy(scibert_paper_cache)
        expert_emb_scibert = load_npy(scibert_expert_cache)
    else:
        tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME, local_files_only=False)
        scibert_model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME, local_files_only=False).to(DEVICE)
        scibert_model.eval()

        print("Encoding papers with SciBERT...")
        paper_emb_scibert = encode_scibert_batch(
            paper_texts,
            tokenizer,
            scibert_model,
            batch_size=SCIBERT_PAPER_BATCH,
            max_length=SCIBERT_MAX_LENGTH,
            label="papers",
            print_every=25
        )

        print("Encoding experts with SciBERT...")
        expert_emb_scibert = encode_scibert_batch(
            expert_texts,
            tokenizer,
            scibert_model,
            batch_size=SCIBERT_EXPERT_BATCH,
            max_length=SCIBERT_MAX_LENGTH,
            label="experts",
            print_every=5
        )

        save_npy(paper_emb_scibert, scibert_paper_cache)
        save_npy(expert_emb_scibert, scibert_expert_cache)
        print("Saved SciBERT embeddings to cache.")

    similarity_matrix_scibert = cosine_similarity(paper_emb_scibert, expert_emb_scibert)
    scibert_ndcg, adjusted_scibert = evaluate_ndcg(similarity_matrix_scibert, papers, experts, k=TOP_K)

    print(f"SciBERT NDCG@{TOP_K}: {scibert_ndcg}")
    export_manual_check(adjusted_scibert, papers, experts, "SCIBERT", TOP_K)
    print_expert_top_papers(similarity_matrix_scibert, papers, experts, "SCIBERT", TOP_K)

# -----------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------

summary_df = pd.DataFrame([
    {"Model": "TF-IDF", "NDCG@5": tfidf_ndcg},
    {"Model": "Sentence-BERT", "NDCG@5": sbert_ndcg},
    {"Model": "SciBERT", "NDCG@5": scibert_ndcg},
])

summary_df.to_excel("model_ndcg_summary.xlsx", index=False)

print("\n========== FINAL RESULTS ==========")
print(summary_df)
print("Saved: model_ndcg_summary.xlsx")