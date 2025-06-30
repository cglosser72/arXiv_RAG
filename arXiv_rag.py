import os
from dotenv import load_dotenv
import openai
import arxiv
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import faiss
import numpy as np
import json
from openai import OpenAI
import re
from IPython.display import Markdown, display
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

from rank_bm25 import BM25Okapi

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_arxiv_abstracts(query="hep-ph", max_results=10):
    """
    Fetch recent paper metadata and abstracts from the arXiv API.

    Performs a search on arXiv using the specified query and retrieves a list of 
    papers sorted by submission date. Each paper includes title, abstract, authors, 
    publication date, arXiv ID, URL, and comments.

    Args:
        query (str, optional): The arXiv category or search query (e.g., "hep-ph"). 
            Default is "hep-ph".
        max_results (int, optional): Maximum number of papers to fetch. 
            Default is 10.

    Returns:
        List[dict]: A list of paper metadata dictionaries.
    """
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = client.results(search)

    all_papers = []
    for result in results:
        all_papers.append({
            "title": result.title,
            "abstract": result.summary,
            "id": result.entry_id,
            "url": result.pdf_url,
            "published": result.published.strftime("%Y-%m-%d"),
            "authors": [a.name for a in result.authors],
            "comment": result.comment
        })
        if len(all_papers) >= max_results:
            break
    return all_papers

# EMBED AND STORE

def build_bm25_index(papers):
    """
    Build a BM25 index over a list of arXiv paper abstracts.

    Tokenizes each abstract into lowercase word tokens and creates a BM25Okapi object 
    for sparse keyword-based retrieval.

    Args:
        papers (List[dict]): List of paper metadata dictionaries, each containing an 'abstract' field.

    Returns:
        Tuple[BM25Okapi, List[List[str]]]: A tuple containing the BM25 index and the tokenized corpus.
    """    
    tokenized_corpus = [paper['abstract'].lower().split() for paper in papers]
    return BM25Okapi(tokenized_corpus), tokenized_corpus

def embed_abstracts(papers, show_progress_bar=False):
    """
    Generate dense vector embeddings for a list of arXiv paper abstracts.

    Uses the "all-MiniLM-L6-v2" SentenceTransformer model to convert each abstract 
    into a fixed-size embedding suitable for semantic search.

    Args:
        papers (List[dict]): List of paper metadata dictionaries, each containing an 'abstract' field.
        show_progress_bar (bool, optional): Whether to display a progress bar during embedding.
            Default is False.

    Returns:
        np.ndarray: Array of shape (n_papers, embedding_dim) containing the abstract embeddings.
    """
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    abstracts = [paper["abstract"] for paper in papers]
    embeddings = model.encode(abstracts, show_progress_bar=show_progress_bar)
    return np.array(embeddings)

def store_faiss_index(embeddings, papers):
    """
    Store a FAISS index and related metadata for later retrieval.

    Saves the dense vector index of abstract embeddings using FAISS, along with:
    - Full paper metadata in JSON format.
    - A tokenized corpus of abstracts for BM25 indexing.

    Args:
        embeddings (np.ndarray): Matrix of abstract embeddings (shape: [n_papers, embedding_dim]).
        papers (List[dict]): List of paper metadata dictionaries corresponding to the embeddings.

    Side Effects:
        - Writes "arxiv_index.faiss" (FAISS index)
        - Writes "arxiv_metadata.json" (paper metadata)
        - Writes "bm25_tokenized_corpus.json" (tokenized abstracts for BM25)
    """    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "arxiv_index.faiss")

    # Save full metadata
    with open("arxiv_metadata.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)

    # Tokenize for BM25
    tokenized_corpus = [paper['abstract'].lower().split() for paper in papers]
    with open("bm25_tokenized_corpus.json", "w") as f:
        json.dump(tokenized_corpus, f)

    print(f"Stored {len(embeddings)} embeddings in FAISS.")

# RETRIEVE

def load_faiss_index(index_path="arxiv_index.faiss"):
    """
    Load a FAISS index from disk for dense vector similarity search.

    This index is typically built from sentence embeddings of arXiv abstracts and 
    saved during the indexing step.

    Args:
        index_path (str, optional): Path to the saved FAISS index file. 
            Default is "arxiv_index.faiss".

    Returns:
        faiss.Index: A FAISS index object ready for search operations.
    """    
    return faiss.read_index(index_path)

def load_metadata_abstracts(metadata_path="arxiv_metadata.json"):
    """
    Load metadata for arXiv papers from a JSON file.

    The metadata includes fields such as title, abstract, authors, publication date, 
    and arXiv URL. This file is typically generated during the embedding and indexing step.

    Args:
        metadata_path (str, optional): Path to the metadata JSON file. 
            Default is "arxiv_metadata.json".

    Returns:
        List[dict]: A list of paper metadata dictionaries.
    """    
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve_similar_abstracts(query, k=3, include_abstract=True):
    """
    Retrieve the top-k most semantically similar arXiv abstracts using dense vector search.

    This function encodes the input query using a sentence embedding model and searches 
    a pre-built FAISS index of arXiv paper abstracts for the most relevant matches.

    Args:
        query (str): The user’s search query or question.
        k (int, optional): Number of top results to retrieve. Default is 3.
        include_abstract (bool, optional): Whether to print abstracts in the output. 
            If False, only titles are printed. Default is True.

    Returns:
        List[dict]: A list of the top-k matching paper dictionaries, each containing metadata 
        such as title, abstract, authors, and URL.
    """
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
    Ask GPT-4o a question based on the list of paper abstracts and structured metadata.

    Args:
        papers (list): List of paper dicts with 'title', 'abstract', 'authors', etc.
        question (str): User's question.
        model (str): OpenAI model name (default: "gpt-4o").
        max_tokens (int): Max tokens for response.

    Returns:
        str: GPT-4o's answer.
    """
    context = "\n\n".join([
        f"Paper {i+1}:\n"
        f"  Title: {p['title']}\n"
        f"  Authors: {', '.join(p.get('authors', []))}\n"
        f"  Published: {p.get('published', 'N/A')}\n"
        f"  Comment: {p.get('comment', '')}\n"
        f"  Abstract: {p['abstract']}"
        for i, p in enumerate(papers)
    ])

    system_prompt = (
        "You are an expert research assistant. Use the structured data provided "
        "(including authors, publication dates, and abstracts) to answer the user's question as accurately as possible."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here are some research papers:\n\n{context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


def format_for_markdown(answer_text):
    """
    Convert LaTeX-style math expressions in text to Markdown-compatible syntax.

    Replaces inline math \( ... \) with $...$ and block math \[ ... \] with $$...$$.

    Args:
        answer_text (str): Text containing LaTeX-style math.

    Returns:
        str: Text with math expressions formatted for Markdown rendering.
    """    
    # Replace inline math: \( ... \) → $...$
    answer_text = re.sub(r'\\\((.*?)\\\)', r'$\1$', answer_text)
    
    # Replace block math: \[ ... \] → $$...$$
    answer_text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', answer_text)

    return answer_text

def extract_k_from_question(question, default_k=4, max_k=25):
    """
    Use GPT-4 to extract the number of results the user is asking for (e.g., "10 newest preprints").
    If no number is found, return default_k.
    """
    prompt = (
        "Extract the number of results requested in the question below.\n"
        "Only return an integer, no explanation.\n\n"
        f"Question: {question}"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0
    )

    try:
        k = int(response.choices[0].message.content.strip())
        if k>0:
            return min(k, max_k)
        else:
            return default_k
    except:
        return default_k

def retrieve_and_rerank(query, initial_k=10, final_k=3):
    """
    Retrieve top-k abstracts using dense retrieval, then rerank using cross-encoder.

    Args:
        query (str): The user's query.
        initial_k (int): Number of initial candidates from retriever.
        final_k (int): Number of top reranked abstracts to return.

    Returns:
        list: Top reranked paper dicts.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Load retriever data
    index = load_faiss_index()
    metadata = load_metadata_abstracts()

    # Retrieve dense matches
    query_vec = model.encode([query])
    _, I = index.search(np.array(query_vec), initial_k)
    candidate_papers = [metadata[i] for i in I[0]]

    # Prepare for reranking
    rerank_inputs = [(query, paper['abstract']) for paper in candidate_papers]
    scores = cross_encoder.predict(rerank_inputs)

    # Sort and return top reranked
    reranked = sorted(zip(scores, candidate_papers), key=lambda x: x[0], reverse=True)
    top_reranked = [paper for score, paper in reranked[:final_k]]

    print("\nReranked Top Matches:\n")
    for rank, paper in enumerate(top_reranked):
        print(f"--- [{rank+1}] {paper['title']} ---")
        print(f" Abstract: {paper['abstract']}\n")
        print(f" arXiv link: {paper['url']}")
        print("—" * 80)

    return top_reranked

def hybrid_retrieve(query, top_k=5, bm25_weight=0.5):
    """
    Retrieve the top-k most relevant papers using a hybrid strategy that combines dense 
    vector similarity and sparse keyword matching.

    This function blends cosine similarity scores from a dense FAISS index with BM25 
    keyword relevance scores to rank arXiv paper abstracts. The weighting between dense 
    and sparse signals is tunable via the `bm25_weight` parameter.

    Args:
        query (str): The user query or question in natural language.
        top_k (int, optional): Number of top results to return. Default is 5.
        bm25_weight (float, optional): Weight for the BM25 score in the final ranking.
            A value closer to 1 favors sparse (BM25) scoring, while closer to 0 favors
            dense (semantic embedding) scoring. Default is 0.5.

    Returns:
        List[dict]: A list of the top-k paper dictionaries sorted by the hybrid relevance score.
            Each dictionary includes metadata such as title, abstract, authors, and arXiv URL.

    Notes:
        - Embeddings are generated using the "all-MiniLM-L6-v2" model.
        - BM25 uses a pre-tokenized corpus saved during the indexing step.
        - Dense scores are converted to a relevance score via 1 / (1 + L2 distance).
    """    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = load_faiss_index()
    metadata = load_metadata_abstracts()

    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), len(metadata))
    dense_scores = np.zeros(len(metadata))
    for rank, idx in enumerate(I[0]):
        dense_scores[idx] = 1 / (1 + D[0][rank])

    with open("bm25_tokenized_corpus.json", "r") as f:
        tokenized_corpus = json.load(f)
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())

    combined = []
    for i in range(len(metadata)):
        score = bm25_weight * bm25_scores[i] + (1 - bm25_weight) * dense_scores[i]
        combined.append((score, metadata[i]))

    top_combined = sorted(combined, key=lambda x: x[0], reverse=True)[:top_k]
    return [p for s, p in top_combined]



def construct_graph(papers, similarity_threshold=0.7):
    """
    Construct a similarity graph where nodes are papers and edges connect papers 
    with abstract embeddings that exceed a cosine similarity threshold.

    Args:
        papers (list): List of paper dicts with 'abstract' and other metadata.
        similarity_threshold (float): Cosine similarity threshold to create an edge.

    Returns:
        G (networkx.Graph): Graph of papers with similarity-based edges.
    """
    # Get embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    abstracts = [p["abstract"] for p in papers]
    embeddings = model.encode(abstracts)

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Build graph
    G = nx.Graph()

    # Add nodes with metadata
    for i, paper in enumerate(papers):
        G.add_node(i, **paper)

    # Add edges for pairs exceeding the similarity threshold
    num_papers = len(papers)
    for i in range(num_papers):
        for j in range(i + 1, num_papers):
            sim = sim_matrix[i][j]
            if sim >= similarity_threshold:
                G.add_edge(i, j, weight=sim)

    return G

def get_top_similar_pairs(papers, top_n=10):
    """
    Return the top-N most similar paper pairs based on cosine similarity of abstracts.

    Args:
        papers (list): List of paper dicts.
        top_n (int): Number of top similar pairs to return.

    Returns:
        list of tuples: [(i, j, similarity_score), ...]
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    abstracts = [p["abstract"] for p in papers]
    embeddings = model.encode(abstracts)

    sim_matrix = cosine_similarity(embeddings)

    pairs = []
    num_papers = len(papers)
    for i in range(num_papers):
        for j in range(i + 1, num_papers):  # avoid duplicates and self-pairs
            sim = sim_matrix[i][j]
            pairs.append((i, j, sim))

    # Sort by similarity
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Return top-N
    return pairs[:top_n]

def plot_similarity_heatmap(papers, filename=''):
    """
    Plot a cosine similarity heatmap for the abstracts of the given papers.
    """
    abstracts = [p["abstract"] for p in papers]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(abstracts)
    sim_matrix = cosine_similarity(embeddings)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=[f"Paper {i+1}" for i in range(len(papers))],
                yticklabels=[f"Paper {i+1}" for i in range(len(papers))])
    plt.title("Cosine Similarity Between Abstracts")
    plt.xlabel("Paper")
    plt.ylabel("Paper")
    plt.tight_layout()
    if filename != '':
        plt.savefig(filename, dpi=300)
    plt.show()
    return plt

def agentic_rag_answer(question: str, model="gpt-4o", max_tokens=700):
    """
    Uses GPT-4o to reason through a multi-step answer plan using subqueries and document retrieval.
    """
    # Step 1: Decompose question into subqueries
    system_prompt = (
        "You are an expert AI agent assistant. Your job is to break a complex research question "
        "into a series of manageable sub-questions or actions. For each, provide one sentence description."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Break this question into steps: {question}"}
    ]

    plan_response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300,
        temperature=0.3
    )
    substeps = plan_response.choices[0].message.content.strip().split("\n")

    results = []
    for step in substeps:
        subquery = step.strip("- ").strip()
        if not subquery:
            continue
        # Retrieve for each subquery
        top_papers = retrieve_and_rerank(subquery, initial_k=10, final_k=3)
        summary = ask_question_about_abstracts(top_papers, subquery, model=model, max_tokens=300)
        results.append(f"**Subtask:** {subquery}\n\n{summary}\n")

    # Final synthesis
    final_input = "\n\n".join(results)
    messages = [
        {"role": "system", "content": "You are a senior AI assistant synthesizing multiple research findings."},
        {"role": "user", "content": f"Based on the following subtask results, synthesize a complete answer to the question: '{question}'\n\n{final_input}"}
    ]
    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3
    )
    return final_response.choices[0].message.content.strip()

