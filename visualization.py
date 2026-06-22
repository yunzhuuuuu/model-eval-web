import streamlit as st
import pandas as pd

from evaluation import load_datasets, build_results_dataframe

st.set_page_config(
    page_title="Embedding Evaluation Dashboard",
    layout="wide"
)
st.title("Embedding Evaluation Dashboard")

datasets = load_datasets()
results_df = build_results_dataframe()

tab1, tab2 = st.tabs([
    "Dataset Explorer",
    "Evaluation Results"
])

# Dataset Explorer
with tab1:
    st.header("Dataset Selection")

    dataset_name = st.selectbox(
        "Choose a dataset",
        list(datasets.keys())
    )
    dataset = datasets[dataset_name]
    questions = dataset["questions"]
    contexts = dataset["contexts"]
    most_relevant = dataset["most_relevant"]

    # questions table
    st.subheader("Questions")
    questions_df = pd.DataFrame({
        "Question ID": range(len(questions)),
        "Question": questions,
        "Correct Context ID": most_relevant
    })
    st.dataframe(
        questions_df,
        use_container_width=True,
        height=350,
        hide_index=True
    )

    # context table
    st.subheader("Contexts (First 60)")
    contexts_df = pd.DataFrame({
        "Context ID": range(min(60, len(contexts))),
        "Context": contexts[:60]
    })
    st.dataframe(
        contexts_df,
        use_container_width=True,
        height=350,
        hide_index=True
    )

    # question-context pair
    st.header("Question-Context Pair Inspector")
    q_idx = st.number_input(
        "Question ID",
        min_value=0,
        max_value=len(questions) - 1,
        value=0,
        step=1
    )
    st.markdown("### Question")
    st.write(questions[q_idx])

    correct_context_id = most_relevant[q_idx]
    st.markdown(
        f"### Ground Truth Context (ID {correct_context_id})"
    )
    st.write(
        contexts[correct_context_id]
    )

# Evaluation Results

with tab2:
    st.subheader("Evaluation Metrics")
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )
