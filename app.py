import streamlit as st
import re
from arXiv_rag import (
    fetch_arxiv_abstracts,
    embed_abstracts,
    store_faiss_index,
    retrieve_similar_abstracts,
    retrieve_and_rerank,
    ask_question_about_abstracts,
    format_for_markdown,
    extract_k_from_question,
    hybrid_retrieve,
    agentic_rag_answer
)


st.set_page_config(page_title="ArXiv RAG", layout="wide")
st.title("ArXiv Research Assistant")

# Archive options grouped by category
archive_options = {
    "Physics": ["astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "nucl-ex", "nucl-th", "physics", "quant-ph"],
    "Mathematics": ["math", "math-ph", "nlin"],
    "Computer Science": ["cs"],
    "Statistics": ["stat"],
    "Quantitative Biology": ["q-bio"],
    "Quantitative Finance": ["q-fin"],
    "Electrical Engineering & Systems Science": ["eess"],
    "Economics": ["econ"]
}

# Two-tier dropdowns
category = st.selectbox("Select a research category:", list(archive_options.keys()))
query = st.selectbox("Select an arXiv sub-archive:", archive_options[category])

num_abstracts = st.number_input(
    "Number of abstracts to fetch and embed:",
    min_value=5,
    max_value=500,
    value=25,
    step=5
)

# Fetch and embed new data
if st.button("Fetch Latest & Embed"):
    with st.spinner("Fetching and embedding latest abstracts..."):
        papers = fetch_arxiv_abstracts(query=query, max_results=num_abstracts)
        embeddings = embed_abstracts(papers, show_progress_bar=True)
        store_faiss_index(embeddings, papers)
    st.success("Index updated with latest abstracts!")

# Question input
question = st.text_area("What would you like to know?", "")

#use_rerank = st.checkbox("Use Cross-Encoder Reranking", value=True)
retrieval_mode = st.selectbox(
    "Select retrieval strategy:",
    ["Dense Only", "Dense + Rerank", "Hybrid (Dense + BM25)", "Agentic RAG"]
)



# Ask a question using RAG
if st.button("Ask Question"):
    with st.spinner("Retrieving relevant abstracts and generating answer..."):
        k = extract_k_from_question(question, default_k=4, max_k=25)
#        if use_rerank:
#            top_papers = retrieve_and_rerank(question, initial_k=max(2*k, 10), final_k=k)
#        else:
#            top_papers = retrieve_similar_abstracts(question, k=k, include_abstract=False)

#        if retrieval_mode == "Hybrid (Dense + BM25)":
#            top_papers = hybrid_retrieve(question, top_k=k)
#        elif retrieval_mode == "Dense + Rerank":
#            top_papers = retrieve_and_rerank(question, initial_k=max(2*k, 10), final_k=k)
#        else:
#            top_papers = retrieve_similar_abstracts(question, k=k, include_abstract=False)

        if retrieval_mode == "Hybrid (Dense + BM25)":
            top_papers = hybrid_retrieve(question, top_k=k)
            answer = ask_question_about_abstracts(top_papers, question)
        elif retrieval_mode == "Dense + Rerank":
            top_papers = retrieve_and_rerank(question, initial_k=max(2*k, 10), final_k=k)
            answer = ask_question_about_abstracts(top_papers, question)
        elif retrieval_mode == "Dense Only":
            top_papers = retrieve_similar_abstracts(question, k=k, include_abstract=False)
            answer = ask_question_about_abstracts(top_papers, question)
        elif retrieval_mode == "Agentic RAG":
            answer = agentic_rag_answer(question)
            top_papers = []  # Optional: could capture subquery retrievals
        else:
            top_papers = []
            answer = "Invalid retrieval mode."
        
        answer = ask_question_about_abstracts(top_papers, question)
        st.markdown("### GPT-4 Answer")
        st.markdown(answer)

        with st.expander("View Source Papers"):
            for p in top_papers:
                title_md = format_for_markdown(p['title'])
                abstract_md = format_for_markdown(p['abstract'])                
                st.markdown(f"### {title_md}")
                st.markdown(f"*Published:* {p.get('published', 'N/A')}")
                st.markdown(f"*Authors:* {', '.join(p.get('authors', []))}")
                if p.get("comment"):
                    st.markdown(f"*Comment:* {p['comment']}")
                st.markdown(f"**Abstract:**\n\n{abstract_md}")
                st.markdown(f"[PDF Link]({p['url']})\n---")
