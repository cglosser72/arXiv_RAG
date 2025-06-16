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

## Model Overview: GPT-4o in RAG

Once relevant arXiv abstracts are retrieved using FAISS, we use **GPT-4o** — OpenAI's state-of-the-art language model — to answer user questions based on those abstracts.

### Role in the RAG Pipeline

GPT-4o acts as the **generation component** of our Retrieval-Augmented Generation (RAG) setup:

1. **Retrieval (via FAISS):**  
   A query is embedded and matched against stored abstract vectors to retrieve the most semantically relevant papers.

2. **Context Construction:**  
   The titles and abstracts of the top-k matching papers are concatenated into a single prompt.

3. **Augmented Generation (via GPT-4o):**  
   The user's question and the retrieved context are sent to GPT-4o, which produces a coherent, context-aware answer grounded in the source material.

This approach ensures responses are **not hallucinated**, but are instead anchored in actual academic abstracts.

### Why GPT-4o?

- **Higher factual accuracy** than prior models
- **Faster and cheaper** than GPT-4-turbo for API use
- **128k token context window** allows handling of multiple full abstracts in a single query
- **Domain-agnostic** reasoning suitable for diverse arXiv categories

By combining **dense retrieval** with **generative reasoning**, this RAG system becomes a powerful tool for querying scientific literature with natural language.

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

---

