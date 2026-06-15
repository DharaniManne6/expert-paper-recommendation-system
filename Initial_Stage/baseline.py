# -----------------------------------------------------------
# PAPER → EXPERT RECOMMENDATION EVALUATION
# Random Baseline + Perfect Ranking (Upper Bound)
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score


# -----------------------------------------------------------
# STEP 1: Load datasets
# -----------------------------------------------------------

papers = pd.read_excel("papers_shuffled.xlsx")
experts = pd.read_excel("experts_shuffled.xlsx")

print("Papers loaded:", len(papers))
print("Experts loaded:", len(experts))


# -----------------------------------------------------------
# STEP 2: Define evaluation parameters
# -----------------------------------------------------------

K = 5
num_trials = 50


# -----------------------------------------------------------
# STEP 3: RANDOM BASELINE EVALUATION
# -----------------------------------------------------------

all_trial_scores = []

for trial in range(num_trials):

    ndcg_scores = []

    for i in range(len(papers)):

        paper_domain = papers.iloc[i]["Domain_tag"]

        # Build ground truth relevance
        relevance = []

        for j in range(len(experts)):

            expert_domain = experts.iloc[j]["Domain"]

            if expert_domain == paper_domain:
                relevance.append(1)
            else:
                relevance.append(0)

        relevance = np.array(relevance).reshape(1, -1)

        # Random scores
        random_scores = np.random.rand(len(experts)).reshape(1, -1)

        # Compute NDCG
        ndcg = ndcg_score(relevance, random_scores, k=K)
        ndcg_scores.append(ndcg)

        # ---------------------------------------------------
        # SHOW ONE RANDOM EXAMPLE (first trial, first paper)
        # ---------------------------------------------------

        if trial == 0 and i == 0:

            print("\n==============================")
            print("RANDOM BASELINE EXAMPLE")
            print("==============================")

            print("\nPaper Title:")
            print(papers.iloc[i]["Title"])

            print("\nPaper Domain:", paper_domain)

            ranking = np.argsort(random_scores[0])[::-1]

            print("\nTop 5 Random Experts:")

            for idx in ranking[:5]:
                print(
                    experts.iloc[idx]["Expert_ID"],
                    "| Domain:", experts.iloc[idx]["Domain"],
                    "| Score:", round(random_scores[0][idx], 3)
                )

    trial_avg = np.mean(ndcg_scores)
    all_trial_scores.append(trial_avg)


# -----------------------------------------------------------
# STEP 4: Final Random Baseline Score
# -----------------------------------------------------------

random_baseline_ndcg = np.mean(all_trial_scores)

print("\n-----------------------------------")
print("Random Baseline Evaluation")
print("-----------------------------------")

print("Trials:", num_trials)
print("Top-K:", K)

print("\nRandom Baseline NDCG@5:", random_baseline_ndcg)


# -----------------------------------------------------------
# STEP 5: PERFECT RANKING EVALUATION
# -----------------------------------------------------------

perfect_ndcg_scores = []

for i, paper in papers.iterrows():

    paper_domain = paper["Domain_tag"]

    true_relevance = []

    for _, expert in experts.iterrows():

        if expert["Domain"] == paper_domain:
            true_relevance.append(1)
        else:
            true_relevance.append(0)

    true_relevance = np.array(true_relevance).reshape(1, -1)

    # Perfect prediction
    predicted_scores = true_relevance.copy()

    ndcg = ndcg_score(true_relevance, predicted_scores, k=K)
    perfect_ndcg_scores.append(ndcg)

    # ---------------------------------------------------
    # SHOW PERFECT RANKING EXAMPLE (first paper)
    # ---------------------------------------------------

    if i == 0:

        print("\n==============================")
        print("PERFECT RANKING EXAMPLE")
        print("==============================")

        print("\nPaper Title:")
        print(paper["Title"])

        print("\nPaper Domain:", paper_domain)

        ranking = np.argsort(predicted_scores[0])[::-1]

        print("\nTop 5 Perfect Experts:")

        for idx in ranking[:5]:
            print(
                experts.iloc[idx]["Expert_ID"],
                "| Domain:", experts.iloc[idx]["Domain"],
                "| Relevance:", predicted_scores[0][idx]
            )


# -----------------------------------------------------------
# STEP 6: Final Perfect Ranking Score
# -----------------------------------------------------------

perfect_ndcg = np.mean(perfect_ndcg_scores)

print("\n-----------------------------------")
print("Perfect Ranking Evaluation")
print("-----------------------------------")

print("Perfect Ranking NDCG@5:", perfect_ndcg)