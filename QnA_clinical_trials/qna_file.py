import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load sentence transformer model for embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


def load_clinical_trial_data(json_file):
    """Load and embed clinical trial summaries."""
    with open(json_file, "r") as f:
        trials = json.load(f)

    texts = [trial["summary"] for trial in trials]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return trials, index, embeddings


def find_relevant_trial(query, trials, index, embeddings):
    """Find the most relevant clinical trial using FAISS similarity search."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, idx = index.search(query_embedding, 1)  # Find top match
    return trials[idx[0][0]]


def answer_query(query, trial):
    """Use an LLM to answer questions based on the trial summary."""
    return qa_pipeline({"question": query, "context": trial["summary"]})


def main():
    json_file = "structured_clinical_trials.json"
    trials, index, embeddings = load_clinical_trial_data(json_file)

    query = "What are the side effects of Drug X?"
    relevant_trial = find_relevant_trial(query, trials, index, embeddings)
    answer = answer_query(query, relevant_trial)

    print(f"Query: {query}")
    print(f"Relevant Trial: {relevant_trial['title']}")
    print(f"Answer: {answer['answer']}")


if __name__ == "__main__":
    main()
