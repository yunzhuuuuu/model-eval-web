import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
import os

MODELS = [
    "gemini_3072",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-dot-v1"
]

# load data
def load_datasets():
    datasets = {}
    # auto discover datasets
    for filename in os.listdir("datasets"):
        if not filename.endswith(".npz"):
            continue
        dataset_name = filename[:-4]

        d = np.load(os.path.join("datasets", filename))
        datasets[dataset_name] = {
            "questions": d["questions"],
            "contexts": d["contexts"],
            "most_relevant": d["most_relevant_context"]
        }
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
    datasets = load_datasets()
    for dataset_name in datasets:
        for model in MODELS:
            q_path = (f"embeddings/"f"{dataset_name}_questions_{model}.npz")
            c_path = (f"embeddings/"f"{dataset_name}_contexts_{model}.npz")
            if not (os.path.exists(q_path) and os.path.exists(c_path)):
                continue

            question_embeddings = np.load(q_path)["embeddings"]
            context_embeddings = np.load(c_path)["embeddings"]
            rankings = calculate_similarities(question_embeddings,context_embeddings)
            all_rankings[dataset_name][model] = rankings
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
def build_results_tables():
    datasets = load_datasets()
    rankings = load_rankings()
    result_tables = {}
    for dataset_name in datasets:
        most_relevant = datasets[dataset_name]["most_relevant"]
        rows = []
        for model in MODELS:
            if model not in rankings[dataset_name]:
                rows.append({
                    "model": model,
                    "Recall@1": np.nan,
                    "Recall@3": np.nan,
                    "MRR": np.nan
                })
                continue

            r = rankings[dataset_name][model]
            rows.append({
                "model": model,
                "Recall@1": round(recall_at_k(r, most_relevant, 1), 4),
                "Recall@3": round(recall_at_k(r, most_relevant, 3), 4),
                "MRR": round(mean_reciprocal_rank(r, most_relevant), 4)
            })
        result_tables[dataset_name] = pd.DataFrame(rows)
    return result_tables