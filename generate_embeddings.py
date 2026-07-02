import numpy as np
from sentence_transformers import SentenceTransformer
import os
from csv import reader
import json

def gemini_embedded(texts, label):
    # get gemini embeddings
    batch_start = 0
    batch_size = 250
    all_embeddings = []
    while batch_start < len(texts):
        with open('request.json','w') as f:
            obj = {'instances': [
                {"content": text}
                for text in texts[batch_start: batch_start+batch_size]],
                'parameters' : {
                    'autoTruncate': True
                }}
            json.dump(obj, f)
        os.system("./get_gemini_embeddings.sh")
        with open('gemini_embeddings.json') as f:
            embeddings = json.load(f)

            batch_embeddings = np.array([x['embeddings']['values'] for x in embeddings['predictions']])
            all_embeddings.append(batch_embeddings)
        batch_start += batch_size
    all_embeddings = np.concatenate(all_embeddings)
    print(f"saving gemini embeddings {label} {all_embeddings.shape}")
    np.savez(f"{label}.npz", embeddings=all_embeddings)

# def embed_assistive_tech(generate_gemini_embeddings=False):
#     with open("assistive_technotes_320.csv") as f:
#         facts = reader(f)
#         contexts = [fact[1] for fact in facts]
#         contexts = contexts[1:]

#     with open("assistive_technotes_qanda.csv") as f:
#         r = reader(f)
#         q_and_as, most_relevant = zip(*[(q_and_a[1], q_and_a[2]) for q_and_a in r])
#         questions = q_and_as[1:]
#         most_relevant = most_relevant[1:]
#         most_relevant_indices = [contexts.index(m) for m in most_relevant]
#     embed_dataset("assistive_technology_320", questions, contexts, most_relevant_indices, generate_gemini_embeddings)

def embed_csv_dataset(dataset_name, facts_csv, qa_csv, generate_gemini_embeddings=False):
    # Facts CSV:
    # col 0 = context
    with open(facts_csv, encoding='utf-8') as f:
        facts = reader(f)
        contexts = [row[0] for row in facts][1:]

    # Q&A CSV:
    # col 0 = question
    # col 1 = matching context
    with open(qa_csv, encoding='utf-8') as f:
        rows = list(reader(f))

        questions = [row[0] for row in rows[1:]]
        most_relevant = [row[1] for row in rows[1:]]

    context_to_index = {
        context: i
        for i, context in enumerate(contexts)
    }

    for context in most_relevant:
        if context not in context_to_index:
            raise ValueError(
                f"Context from Q&A file not found in facts file:\n{context}"
            )
        
    most_relevant_indices = [
        context_to_index[context]
        for context in most_relevant
    ]

    embed_dataset(dataset_name, questions, contexts, most_relevant_indices, generate_gemini_embeddings)

# def embed_squad(generate_gemini_embeddings=False):
#     dataset = load_dataset("squad", split='validation')
#     questions = dataset["question"]
#     contexts = dataset["context"]

#     # Build unique context set (many questions share the same paragraph)
#     unique_contexts = list(dict.fromkeys(contexts))
#     context_to_index = {c: i for i, c in enumerate(unique_contexts)}
#     most_relevant_context = [context_to_index[c] for c in contexts]
#     embed_dataset('squad_1_1', questions, unique_contexts, most_relevant_context, generate_gemini_embeddings)

def embed_dataset(dataset_name, questions, contexts, most_relevant_context, generate_gemini_embeddings=False):
    print(f"Questions: {len(questions)}")
    print(f"Contexts: {len(contexts)}")
    np.savez(f"datasets/{dataset_name}.npz", questions=questions, contexts=contexts, most_relevant_context=most_relevant_context)
    if generate_gemini_embeddings:
        gemini_embedded(questions, f"embeddings/{dataset_name}_questions_gemini_3072")
        gemini_embedded(contexts, f"embeddings/{dataset_name}_contexts_gemini_3072")

    # a faster model
    model = SentenceTransformer("multi-qa-MiniLM-L6-dot-v1")
    embed_sentence_transformer(model, questions, f"embeddings/{dataset_name}_questions_multi-qa-MiniLM-L6-dot-v1.npz")
    embed_sentence_transformer(model, contexts, f"embeddings/{dataset_name}_contexts_multi-qa-MiniLM-L6-dot-v1.npz")

    # the EchoMinds model
    model = SentenceTransformer("all-mpnet-base-v2")
    embed_sentence_transformer(model, questions, f"embeddings/{dataset_name}_questions_all-mpnet-base-v2.npz")
    embed_sentence_transformer(model, contexts, f"embeddings/{dataset_name}_contexts_all-mpnet-base-v2.npz")

def embed_sentence_transformer(model, texts, label):
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    np.savez(label, embeddings=embeddings)

if __name__ == "__main__":
    # embed_squad()
    dataset1 = "assistive_technology_320"
    dataset2 = "cooking"
    # should rewrite to run for all unprocessed datasets automatically
    # embed_csv_dataset(
    #     dataset1,
    #     f"raw_data/{dataset1}/facts.csv",
    #     f"raw_data/{dataset1}/qanda.csv"
    # )
    embed_csv_dataset(
        dataset2,
        f"raw_data/{dataset2}/facts.csv",
        f"raw_data/{dataset2}/qanda.csv"
    )