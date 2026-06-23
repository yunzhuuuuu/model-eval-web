import streamlit as st
import pandas as pd

from evaluation import load_datasets, build_results_tables

st.set_page_config(
    page_title="Embedding Evaluation Dashboard",
    layout="wide"
)
st.title("Retrieval Model Evaluation Dashboard")

datasets = load_datasets()
results_tables = build_results_tables()

tab1, tab2, tab3 = st.tabs([
    "Introduction",
    "Dataset Explorer",
    "Evaluation Results"
])

# Introduction
with tab1:
    st.header("Learning Objectives")
    st.markdown("""
    This dashboard is designed to help students explore how dataset design influences the performance of retrieval models.

    Through this activity, students will:
    - Understand the evaluation metrics for retrieval models and use them to compare the performance of different models
    - Investigate how different dataset can affect retrieval accuracy
    - Design and evaluate their own retrieval datasets

    There won't be any coding involved. Students only need to create their datasets in excel forms and upload it here (TO BE DONE)     
    """)
                
    st.header("Background")
    st.markdown("""
    Olin students developed a note-taking application, where users store information as notes and could later retrieve that information by asking questions. 
    The app implements a retrieval model that can match the user's question to the most relevant note instead of matching exact key words.    

    The retrieval system compares the meaning of a question with the meaning of every available note.
    For example, a user might ask:

    "What ingredients do I need for the pasta recipe?"
                
    A relevant note might contain:

    "Pasta: spaghetti, olive oil, garlic, cherry tomatoes, basil, parmesan cheese."
                
    Even though the question and note do not use exactly the same words, a retrieval model can recognize that they discuss the same topic.
    To do this, the model calculates a similarity score between the question and each available note. Notes with higher similarity scores are ranked closer to the top of the results.

    A strong retrieval model consistently places the correct note among its highest-ranked results, which can also be reflected by some metrics we'll discuss in the Evaluation Results tab.

    The SQuAD and Assistive Technology datasets were used by the Olin students and serve as examples of retrieval datasets.
    """)

    st.header("Your Task")
    st.markdown("""
    You will create and evaluate their own datasets based on the scenario of a note-taking application. A valid dataset should contain:
    The topic of the notes is flexible. For example, a dataset could focus on:
    - Study notes for a class
    - Cooking recipes
    - Travel planning information
    - Any other information that could reasonably be stored as notes 
                
    TODO: write instructions on how to create their datasets, what it should look like and how to upload it        
    """)

# Dataset Explorer
with tab2:
    st.header("Dataset Selection")
    st.markdown("""
    Choose a dataset to explore.

    Each dataset contains:

    - Questions: Information requests that a user might ask.
    - Contexts: Notes (fact statements) that may contain answers.
    - Ground Truth Matches: Human-annotated links between questions and the context that best answers them.
    """)
    dataset_name = st.selectbox(
        "Select a dataset",
        list(datasets.keys())
    )
    dataset = datasets[dataset_name]
    questions = dataset["questions"]
    contexts = dataset["contexts"]
    most_relevant = dataset["most_relevant"]

    # questions table
    st.subheader("Questions")
    st.markdown("""
    The Questions table lists all questions and the ID of the corresponding context pieces that has been labeled as the correct match.
    """)
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
    st.markdown("""
    The Contexts table displays the available passages or notes that can be retrieved.
    Only the first 60 contexts are shown here for easier browsing.
    """)
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
    st.markdown("""
    Select an example question to inspect its corresponding ground-truth context.
    """)
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

with tab3:
    st.subheader("Evaluation Metrics")
    st.markdown("""
    This page summarizes how well different retrieval models perform on each dataset.
    The results table provides a quantitative comparison of retrieval performance across datasets and models.

    Three evaluation metrics are reported:

    - Recall@1 measures how often the correct note is ranked first. For example, a Recall@1 score of 0.80 means the model returned the correct note as its top result for 80% of questions.
    - Recall@3 measures how often the correct note appears within the top three results. This reflects how likely a user is to find the correct note after reviewing a small number of suggestions.
    - Mean Reciprocal Rank (MRR) evaluates not only whether the correct note was retrieved, but also how highly it was ranked. Models receive higher scores when the correct note appears closer to the top of the results list.
    (more: https://www.pinecone.io/learn/offline-evaluation/#Mean-Reciprocal-Rank-(MRR))
    """)

    st.button(
    "Evaluate Your Dataset",
    disabled=True,
    help="TODO: generate a new table of the model performances on the customized dataset"
)

    st.markdown("### SQuAD")

    st.dataframe(
        results_tables["squad_1_1"],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Assistive Technology")

    st.dataframe(
        results_tables["assistive_technology_320"],
        use_container_width=True,
        hide_index=True
    )