import os
import openai
import arxiv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

def fetch_arxiv_abstracts(query="hep-ph", max_results=10):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    client = arxiv.Client()
    results = client.results(search)

    papers = []
    for result in results:
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "id": result.entry_id,
            "url": result.pdf_url
        })

    return papers

# EMBED AND STORE

def embed_abstracts(papers, show_progress_bar=False):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    abstracts = [paper["abstract"] for paper in papers]
    embeddings = model.encode(abstracts, show_progress_bar=show_progress_bar)
    return np.array(embeddings)

def store_faiss_index(embeddings, papers):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "arxiv_index.faiss")

    # Save full metadata
    with open("arxiv_metadata.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)

    print(f"Stored {len(embeddings)} embeddings in FAISS.")

# RETRIEVE

def load_faiss_index(index_path="arxiv_index.faiss"):
    return faiss.read_index(index_path)

def load_metadata_abstracts(metadata_path="arxiv_metadata.json"):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve_similar_abstracts(query, k=3, include_abstract=True):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = load_faiss_index()
    metadata = load_metadata_abstracts()

    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)  # distances, indices

    matched_papers = [metadata[idx] for idx in I[0]]

    if include_abstract:    
        print("\nTop Matching Abstracts:\n")
        for rank, paper in enumerate(matched_papers):
            print(f"--- [{rank+1}] {paper['title']} ---")
            print(f" Abstract: {paper['abstract']}\n")
            print(f" arXiv link: {paper['url']}")
            print("—" * 80)
    else:
        print("\nTop Matches:\n")
        for rank, paper in enumerate(matched_papers):
            print(f"--- [{rank+1}] {paper['title']} ---")

    return matched_papers
