import os
from dotenv import load_dotenv
import openai
import arxiv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from openai import OpenAI
import re
from IPython.display import Markdown, display

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def ask_question_about_abstracts(papers, question, model="gpt-4o", max_tokens=500):
    """
    Ask GPT-4o a question based on the list of paper abstracts.
    
    Args:
        papers (list): List of paper dicts with 'title' and 'abstract'.
        question (str): User's question.
        model (str): OpenAI model name (default: "gpt-4o").
        max_tokens (int): Max tokens for response.

    Returns:
        str: GPT-4o's answer.
    """
    context = "\n\n".join(
        [f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in papers]
    )

    system_prompt = "You are an expert research assistant. Use the abstracts provided to answer the user's question as precisely as possible."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here are some research paper abstracts:\n\n{context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

def format_for_markdown(answer_text):
    # Replace inline math: \( ... \) → $...$
    answer_text = re.sub(r'\\\((.*?)\\\)', r'$\1$', answer_text)
    
    # Replace block math: \[ ... \] → $$...$$
    answer_text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', answer_text)

    return answer_text