# Retrieval-Augmented Generation on arXiv Abstracts

This notebook demonstrates a simple RAG (Retrieval-Augmented Generation) system using abstracts from the [arXiv preprint server](https://arxiv.org).

We:

- Fetch recent abstracts from arXiv
- Embed them using a transformer-based sentence encoder
- Store and retrieve embeddings using FAISS
- Use GPT to generate answers from retrieved context

## FAISS: Facebook AI Similarity Search

FAISS (Facebook AI Similarity Search) is a high-performance library for **efficient similarity search** over dense vector representations. It is especially well-suited for applications like retrieval-augmented generation (RAG), recommendation systems, and nearest-neighbor search in embedding spaces.

### How It Works:

- FAISS stores all of our abstract embeddings as vectors in a **vector index**.
- When a user enters a query, we embed it using the same model (`all-MiniLM-L6-v2`).
- FAISS compares the query vector to the stored vectors and returns the **top-k most similar** entries based on distance (usually L2 or cosine).

### Why We Use FAISS:

- **Speed**: Handles millions of vectors efficiently with GPU/CPU support.
- **Scalability**: Works well for large-scale document search.
- **Simplicity**: Easy to use for exact or approximate nearest neighbor search.

### In This Project:

- We use `IndexFlatL2` (a brute-force exact search index using Euclidean distance).
- Abstracts are embedded once and stored.
- At query time, FAISS retrieves the most semantically similar papers in milliseconds.

This allows us to build a fast and responsive retrieval system that scales with more data.  Please visit [their github](https://github.com/facebookresearch/faiss/wiki/) for more info.

## Model Overview: `all-MiniLM-L6-v2`

We use the `all-MiniLM-L6-v2` model from the [SentenceTransformers](https://www.sbert.net/) library to convert text into dense vector embeddings. These embeddings represent the **semantic meaning** of text and are used for similarity search in our RAG system.

### Key Features:

- **Architecture**: MiniLM (6 Transformer layers, distilled from BERT)
- **Embedding Dimension**: 384
- **Speed**: Extremely fast, making it suitable for real-time or large-scale applications
- **Use Case**: Optimized for general-purpose semantic similarity tasks (e.g., question answering, duplicate detection, clustering)

### Why We Use It:

- Lightweight and fast — ideal for prototyping and scalable applications
- High-quality embeddings despite small size
- Pretrained on a diverse set of tasks like Natural Language Inference (NLI) and Semantic Textual Similarity (STS)

### Output:

Each input text (e.g., an arXiv abstract or a user query) is mapped to a 384-dimensional vector that can be compared to other vectors using cosine or Euclidean distance.

This model is especially useful for identifying semantically similar scientific texts, even when exact keywords don’t match.

### Model Overview: `cross-encoder/ms-marco-MiniLM-L-6-v2`

To improve the accuracy of document retrieval, we incorporate a **CrossEncoder** model to rerank candidate abstracts retrieved by the bi-encoder model. This creates a two-stage retrieval pipeline: **dense retrieval followed by reranking**.

#### What is a CrossEncoder?

Unlike a BiEncoder (which encodes the query and document separately), a **CrossEncoder** feeds both the query and the document into the same Transformer at the same time. This allows the model to jointly attend to all tokens in the input, enabling much more precise relevance scoring.

#### Why Use `cross-encoder/ms-marco-MiniLM-L-6-v2`?

- **Architecture**: A lightweight transformer with \~22M parameters, based on MiniLM
- **Training Data**: Trained on the MS MARCO dataset (real-world web search queries and passages)
- **Input**: Pairs of text — in this case, `(query, abstract)`
- **Output**: A single relevance score for each pair

#### Role in the RAG Pipeline:

1. **Initial Retrieval**:\
   Use `all-MiniLM-L6-v2` (BiEncoder) + FAISS to find top-k potentially relevant abstracts.

2. **Reranking**:\
   Score each `(query, abstract)` pair using the CrossEncoder.

3. **Top-N Selection**:\
   Return the top N highest-scoring abstracts to feed into GPT-4o for question answering.

This hybrid strategy significantly boosts semantic accuracy while maintaining good performance, especially when abstracts contain similar keywords but differ in meaning.

## Model Overview: GPT-4o in RAG

Once relevant arXiv abstracts are retrieved using FAISS, we use **GPT-4o** — OpenAI's state-of-the-art language model — to answer user questions based on those abstracts.

### Role in the RAG Pipeline

GPT-4o acts as the **generation component** of our Retrieval-Augmented Generation (RAG) setup:

1. **Retrieval (via FAISS):**\
   A query is embedded and matched against stored abstract vectors to retrieve the most semantically relevant papers.

2. **Context Construction:**\
   The titles and abstracts of the top-k matching papers are concatenated into a single prompt.

3. **Augmented Generation (via GPT-4o):**\
   The user's question and the retrieved context are sent to GPT-4o, which produces a coherent, context-aware answer grounded in the source material.

This approach ensures responses are **not hallucinated**, but are instead anchored in actual academic abstracts.

### Why GPT-4o?

- **Higher factual accuracy** than prior models
- **Faster and cheaper** than GPT-4-turbo for API use
- **128k token context window** allows handling of multiple full abstracts in a single query
- **Domain-agnostic** reasoning suitable for diverse arXiv categories

By combining **dense retrieval** with **generative reasoning**, this RAG system becomes a powerful tool for querying scientific literature with natural language.

## Graph-Augmented RAG (Graph RAG)

In addition to dense retrieval and reranking, this project implements a **Graph-Augmented Retrieval-Augmented Generation (Graph RAG)** pipeline that builds a similarity graph from research papers retrieved via the arXiv API.

### Motivation

Traditional RAG systems retrieve top-k documents and generate answers directly from them. However, this approach may miss useful information that is semantically adjacent but not ranked in the top-k.

**Graph RAG** addresses this by:

- Building a **graph of papers** based on abstract similarity (using cosine similarity on embeddings)
- Exploring the local neighborhood of the most relevant papers to enrich context
- Enabling **multi-hop reasoning** across semantically related documents

---

### How It Works

1. **Retrieve & Rerank** We use dense vector retrieval (FAISS) followed by a cross-encoder reranker to identify the top-N relevant papers for a user query.

2. **Construct Similarity Graph** A graph is constructed where:

   - Nodes = research papers
   - Edges = semantic similarity between abstracts exceeding a configurable threshold

3. **Heatmap Visualization** We visualize the pairwise similarity matrix of the top papers using a heatmap to understand semantic clustering.

---

### Example Output

Here’s a heatmap of cosine similarity between 20 physics abstracts retrieved via the arXiv API:

>

High-scoring cells indicate closely related research works, which are linked together in the semantic graph.

---

### Configuration

You can tune the graph construction by adjusting:

- `similarity_threshold` (default: `0.7`)
- Embedding model (e.g., `all-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- Edge types (future: shared authors, temporal proximity)

## Hybrid Retrieval (Dense + BM25)

We introduce a **hybrid retrieval strategy** that combines semantic search with traditional lexical scoring using **BM25**.

### Why Hybrid?

While dense retrieval captures semantic similarity, it sometimes overlooks important keyword matches. BM25 excels at this but lacks deeper semantic understanding. Hybrid retrieval offers the best of both worlds.

### How It Works

1. **Dense Scores:**\
   Compute similarity between the query embedding and abstract embeddings using FAISS.

2. **BM25 Scores:**\
   Compute token-level similarity scores using the BM25Okapi model.

3. **Score Fusion:**\
   Combine both scores using a tunable weight:

   `final_score = alpha * bm25_score + (1 - alpha) * dense_score`

   where `alpha` defaults to 0.5.

### Benefits

- Improves recall and diversity
- Balances exact keyword match with semantic relevance
- Performs well when queries include domain-specific jargon

## Agentic RAG

We implement an **Agentic RAG** pipeline that breaks down complex questions into smaller sub-tasks, performs targeted retrieval for each, and synthesizes a comprehensive answer.

### Motivation

Traditional RAG answers a question in one step. Agentic RAG **decomposes the query** into smaller reasoning steps, retrieves relevant documents per step, and combines insights into a final answer — mimicking expert researcher behavior.

### Pipeline

1. **Decompose Question**\
   Use GPT-4o to split the input question into logical sub-questions.

2. **Retrieve & Summarize**\
   For each sub-question, retrieve relevant papers and summarize their content.

3. **Synthesize Final Answer**\
   Feed all intermediate summaries into GPT-4o to generate a coherent final response.

### Example

Given:

> *"What are recent advances in QCD factorization techniques for jet substructure?"*

The agent breaks this into steps like:

- What is QCD factorization?
- What are jet substructure techniques?
- How has factorization evolved recently?

Then retrieves, summarizes, and composes a final narrative.

## The Streamlit App

A Streamlit app is included that you can launch on your local machine. To launch the Streamlit-based arXiv RAG (Retrieval-Augmented Generation) app on your machine, follow these steps:

### 1. **Clone the Repository**

```bash
git clone https://github.com/your-username/arXiv_RAG.git
cd arXiv_RAG
```

### 2. **Set Up a Virtual Environment (Optional but Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. **Install Dependencies**

Make sure you have all required packages:

```bash
pip install -r requirements.txt
```

### 4. **Set Your OpenAI API Key**

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

Alternatively, you can set it as an environment variable directly.

### 5. **Run the App**

```bash
streamlit run app.py
```

This will open the app in your default browser at `http://localhost:8501`.

