# Retrieval-Augmented Generation on arXiv Abstracts

This notebook demonstrates a simple RAG (Retrieval-Augmented Generation) system using abstracts from the [arXiv preprint server](https://arxiv.org).

We:
- Fetch recent abstracts from arXiv
- Embed them using a transformer-based sentence encoder
- Store and retrieve embeddings using FAISS
- Use GPT to generate answers from retrieved context

##  FAISS: Facebook AI Similarity Search

FAISS (Facebook AI Similarity Search) is a high-performance library for **efficient similarity search** over dense vector representations. It is especially well-suited for applications like retrieval-augmented generation (RAG), recommendation systems, and nearest-neighbor search in embedding spaces.  

###  How It Works:
- FAISS stores all of our abstract embeddings as vectors in a **vector index**.
- When a user enters a query, we embed it using the same model (`all-MiniLM-L6-v2`).
- FAISS compares the query vector to the stored vectors and returns the **top-k most similar** entries based on distance (usually L2 or cosine).

###  Why We Use FAISS:
- **Speed**: Handles millions of vectors efficiently with GPU/CPU support.
- **Scalability**: Works well for large-scale document search.
- **Simplicity**: Easy to use for exact or approximate nearest neighbor search.

###  In This Project:
- We use `IndexFlatL2` (a brute-force exact search index using Euclidean distance).
- Abstracts are embedded once and stored.
- At query time, FAISS retrieves the most semantically similar papers in milliseconds.

This allows us to build a fast and responsive retrieval system that scales with more data.  Please visit [their gitub](https://github.com/facebookresearch/faiss/wiki/) for more info.


##  Model Overview: `all-MiniLM-L6-v2`

We use the `all-MiniLM-L6-v2` model from the [SentenceTransformers](https://www.sbert.net/) library to convert text into dense vector embeddings. These embeddings represent the **semantic meaning** of text and are used for similarity search in our RAG system.

###  Key Features:
- **Architecture**: MiniLM (6 Transformer layers, distilled from BERT)
- **Embedding Dimension**: 384
- **Speed**: Extremely fast, making it suitable for real-time or large-scale applications
- **Use Case**: Optimized for general-purpose semantic similarity tasks (e.g., question answering, duplicate detection, clustering)

###  Why We Use It:
- Lightweight and fast — ideal for prototyping and scalable applications
- High-quality embeddings despite small size
- Pretrained on a diverse set of tasks like Natural Language Inference (NLI) and Semantic Textual Similarity (STS)

###  Output:
Each input text (e.g., an arXiv abstract or a user query) is mapped to a 384-dimensional vector that can be compared to other vectors using cosine or Euclidean distance.

This model is especially useful for identifying semantically similar scientific texts, even when exact keywords don’t match.
