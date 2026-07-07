import streamlit as st
import pandas as pd
import os

from evaluation import load_datasets, build_results_tables
from generate_embeddings import embed_csv_dataset

st.set_page_config(
    page_title="Embedding Evaluation Dashboard",
    layout="wide"
)
st.title("Retrieval Model Evaluation Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Introduction",
    "Dataset Explorer",
    "Upload Dataset",
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

    There won't be any coding involved. Students only need to create their datasets in excel forms and upload it here. 
    """)
                
    st.header("Background")
    st.markdown("""
    _Content here can be revised depends on the focus of this project (eg. more about Echominds/HCD/retrieval ML)_

    Students from Olin College of Engineering developed a note-taking application, where users store information as notes and could later retrieve that information by asking questions. 
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
    You will create and evaluate their own datasets based on the scenario of a note-taking application. Read on to explore some example datasets and learn how to create and evaluate your own dataset.   
    """)


# Dataset Explorer
with tab2:
    st.header("Dataset Selection")
    datasets = load_datasets()
    st.markdown("""
    Choose a dataset to explore.

    Each dataset contains:

    - Q&A: Information requests that a user might ask and human-annotated the context that best answers them.
    - Contexts: Notes (fact statements) that the model use to search for the answers.
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
    st.subheader("Q&A")
    st.markdown("""
    The Questions and Answers table lists all questions and the context that has been labeled as the correct match.    """)
    questions_df = pd.DataFrame({
        "Question ID": range(len(questions)),
        "Question": questions,
        "Correct Context": [
            contexts[idx]
            for idx in most_relevant
        ]
    })
    st.dataframe(
        questions_df,
        width="stretch",
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
        width="stretch",
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


# Upload Dataset
with tab3:
    st.header("Create and Upload Your Dataset")

    st.header("How to create your dataset")
    st.markdown("""
    Before you start creating your own dataset and evaluating the model performance on it, you should make sure you've explored the Dataset Explorer Tab and have a sense of what a valid dataset looks like. We will also provide more examples below.

    #### What to include in your dataset

    Your dataset notes can be about any topic that could reasonably be stored in a note-taking application, including:

    - Cooking
    - Travel planning
    - Study notes
    - ......

    You can create the dataset yourself or use LLMs to generate the dataset. Make sure the questions and notes are realistic and relevant to the topic. Here's a potential prompt you can follow:
                
    [PROMPT TO BE DONE]
                
    You are also encouraged to make both by yourself and by using LLMs and explore the differences in the evaluation results.
 
    #### What your dataset should look like
    Your dataset must include 2 .csv files which you could create in Excel or Google Sheets and then export as .csv files. They are:

    1. context.csv
    This file contains your notes stored as sentences. Each row should contain one note. The first row should be a header row with the column name "context". 
    Example:
                
    TODO: example screenshot
                
    2. qanda.csv
    This file contains questions and their correct matching note. Each row should contain one question and its corresponding note. The first row should be a header row with the column names "question" and "context". The "context" column must contain the exact text of the matching note from context.csv.
    Example:
                
    TODO: example screenshot
                
    #### Important notes:
    - Every “Relevant Note” in qanda.csv must exist exactly in context.csv.
    - You can have multiple questions that match the same note.
    - You can have notes in context.csv that are not matched to any question in qanda.csv.                
    - You should manually check that the questions and correct notes pairs are accurate and make sense.

    After uploading, embeddings will be generated automatically and the dataset will become available for evaluation.
    """)

    st.header("Upload Files")
    st.markdown("""Give a word or short phrase to name your dataset.""")
    dataset_name = st.text_input("Dataset Name:")
    st.markdown("""Upload your two .csv files. Once you click "Save Dataset", the system will automatically generate embeddings for your dataset and save it for evaluation.""")
    context_file = st.file_uploader("Upload context", type=["csv"], key="context")
    qanda_file = st.file_uploader("Upload q&a", type=["csv"], key="qanda")

    if st.button("Save Dataset"):
        if not dataset_name:
            st.error("Please provide a dataset name.")
        elif context_file is None or qanda_file is None:
            st.error("Please upload both CSV files.")
        else:
            dataset_dir = os.path.join("raw_data", dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            # save uploads
            context_path = os.path.join(dataset_dir, "context.csv")
            qanda_path = os.path.join(dataset_dir, "qanda.csv")
            with open(context_path, "wb") as f:
                f.write(context_file.getbuffer())
            with open(qanda_path, "wb") as f:
                f.write(qanda_file.getbuffer())

            st.info("Dataset saved. Generating embeddings...")

            # run pipeline
            embed_csv_dataset(
                dataset_name,
                context_path,
                qanda_path,
                generate_gemini_embeddings=False 
            )
            st.success(f"Dataset '{dataset_name}' is ready and embedded.")


# Evaluation Results
with tab4:
    st.subheader("Evaluation Metrics")
    results_tables = build_results_tables()
    st.markdown("""
    This page summarizes how well different retrieval models perform on each dataset.
    The results table provides a quantitative comparison of retrieval performance across datasets and models.

    Three evaluation metrics are reported:

    - Recall@1 measures how often the correct note is ranked first. For example, a Recall@1 score of 0.80 means the model returned the correct note as its top result for 80% of questions.
    - Recall@3 measures how often the correct note appears within the top three results. This reflects how likely a user is to find the correct note after reviewing a small number of suggestions.
    - Mean Reciprocal Rank (MRR) evaluates not only whether the correct note was retrieved, but also how highly it was ranked. Models receive higher scores when the correct note appears closer to the top of the results list.
    (more: https://www.pinecone.io/learn/offline-evaluation/#Mean-Reciprocal-Rank-(MRR))
    
    TODO: users should be able to choose from various metrics to evaluate
    """)

    if st.button("Evaluate Available Datasets"):
        for dataset_name, table in results_tables.items():
            st.markdown(f"### {dataset_name}")
            st.dataframe(
                table,
                width="stretch",
                hide_index=True
            )

    # st.markdown("### SQuAD")

    # st.dataframe(
    #     results_tables["squad_1_1"],
    #     width="stretch",
    #     hide_index=True
    # )

    # st.markdown("### Assistive Technology")

    # st.dataframe(
    #     results_tables["assistive_technology_320"],
    #     width="stretch",
    #     hide_index=True
    # )