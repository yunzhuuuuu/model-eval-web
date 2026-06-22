import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd

EMBEDDING_METHODS = [
    "gemini_3072",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-dot-v1"
]

# load data
def load_datasets():
    datasets = {}

    d = np.load("datasets/squad_1_1.npz")
    datasets["squad_1_1"] = {
        "questions": d["questions"], 
        "contexts": d["contexts"],
        "most_relevant": d["most_relevant_context"]
    } # old name questions_squad, contexts_squad, most_relevant_squad

    d = np.load("datasets/assistive_technology_320.npz")
    datasets["assistive_technology_320"] = {
        "questions": d["questions"],
        "contexts": d["contexts"],
        "most_relevant": d["most_relevant_context"]
    } # old name questions_at, contexts_at, most_relevant_at 

    return datasets

# generate similarity rankings
def calculate_similarities(question_embeddings, context_embeddings):
    similarities = cosine_similarity(question_embeddings,context_embeddings)
    rankings = np.zeros(similarities.shape, np.int32)
    for i in range(similarities.shape[0]):
        rankings[i,:] = np.argsort(-similarities[i,:])
    return rankings

def load_rankings():
    all_rankings = defaultdict(dict)
    for dataset_name in ["squad_1_1", "assistive_technology_320"]:
        for embedding_method in EMBEDDING_METHODS:
            question_embeddings = np.load(f"embeddings/{dataset_name}_questions_{embedding_method}.npz")["embeddings"]
            context_embeddings = np.load(f"embeddings/{dataset_name}_contexts_{embedding_method}.npz")["embeddings"]
            rankings = calculate_similarities(question_embeddings, context_embeddings)
            all_rankings[dataset_name][embedding_method] = rankings
    return all_rankings

# evaluation methods
def recall_at_k(rankings, most_relevant, k):
    hits = 0
    for q in range(len(most_relevant)):
        if most_relevant[q] in rankings[q, :k]:
            hits += 1
    return hits / len(most_relevant)

def mean_reciprocal_rank(rankings, most_relevant):
    rr_sum = 0
    for q in range(len(most_relevant)):
        rank = np.where(rankings[q] == most_relevant[q])[0][0]
        rr_sum += 1 / (rank + 1)
    return rr_sum / len(most_relevant)

# main
def build_results_dataframe():
    datasets = load_datasets()
    rankings = load_rankings()
    rows = []
    for dataset_name in datasets:
        most_relevant = datasets[dataset_name]["most_relevant"]
        for model in EMBEDDING_METHODS:
            r = rankings[dataset_name][model]
            rows.append({
                "dataset": dataset_name,
                "model": model,
                "Recall@1": recall_at_k(r, most_relevant, 1),
                "Recall@3": recall_at_k(r, most_relevant, 3),
                "MRR": mean_reciprocal_rank(r, most_relevant)
            })
    return pd.DataFrame(rows)