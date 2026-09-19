"""
Implements:

* Hybrid search — an EnsembleRetriever that merges the dense (Chroma)
  retriever and a sparse BM25 retriever with reciprocal-rank fusion
  ("RAG-Fusion" idea, but applied across two retrieval *methods* instead 
  of multiple query rewrites).
* A free local cross-encoder re-ranker that re-scores the top-N hybrid
  candidates against the raw query for higher precision before the LLM
  ever sees them ("Re-ranking", uses a free local model instead of 
  the paid Cohere Rerank API).
* Optional per-paper metadata filtering (a lightweight
  "Query Construction") so a question that names a paper only
  retrieves from that paper.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

import config
from src import indexing

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def build_hybrid_retriever(k: int = config.TOP_K_CANDIDATES):
    """Combine Chroma (dense) + BM25 (sparse) retrievers into one hybrid
    retriever using LangChain's EnsembleRetriever (reciprocal rank fusion)."""
    vectorstore = indexing.load_vectorstore()
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    bm25_data = indexing.load_bm25_index()
    bm25_retriever = BM25Retriever.from_documents(bm25_data["docs"])
    bm25_retriever.k = k

    hybrid = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[config.HYBRID_VECTOR_WEIGHT, 1 - config.HYBRID_VECTOR_WEIGHT],
    )
    return hybrid


def filter_by_paper(docs: List[Document], paper_title_hint: Optional[str]) -> List[Document]:
    """If the user mentioned a specific paper title/filename, narrow the
    candidate pool down to that paper only (soft substring match)."""
    if not paper_title_hint:
        return docs
    hint = paper_title_hint.lower()
    filtered = [
        d
        for d in docs
        if hint in d.metadata.get("paper_title", "").lower()
        or hint in d.metadata.get("filename", "").lower()
    ]
    return filtered or docs  # fall back to unfiltered if nothing matched


def rerank(query: str, docs: List[Document], top_k: int = config.TOP_K_FINAL) -> List[Document]:
    """Cross-encoder re-ranking: scores each (query, chunk) pair jointly,
    which is far more precise than embedding cosine similarity alone."""
    if not docs:
        return []
    reranker = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:top_k]]


def retrieve(
    query: str,
    paper_hint: Optional[str] = None,
    candidate_k: int = config.TOP_K_CANDIDATES,
    final_k: int = config.TOP_K_FINAL,
) -> List[Document]:
    """Standard high-precision retrieval path: hybrid search -> top 20
    candidates -> optional paper filter -> cross-encoder re-rank -> top k."""
    hybrid = build_hybrid_retriever(k=candidate_k)
    candidates = hybrid.invoke(query)
    candidates = filter_by_paper(candidates, paper_hint)
    return rerank(query, candidates, top_k=final_k)


def retrieve_for_summary(paper_hint: str, query: str) -> List[Document]:
    """Broader retrieval used for whole-paper / concept summarization:
    pulls a wider pool from the target paper so the summary isn't based on
    a handful of possibly-unrepresentative chunks."""
    docstore = indexing.load_docstore()
    paper_docs = filter_by_paper(docstore, paper_hint)
    if len(paper_docs) <= config.SUMMARY_TOP_K:
        return paper_docs
    return rerank(query, paper_docs, top_k=config.SUMMARY_TOP_K)


def retrieve_multi(sub_queries: List[str], final_k: int = config.TOP_K_FINAL) -> List[Document]:
    """Used for cross-paper / comparison questions (Part 7, Decomposition):
    retrieve separately for each sub-question, then de-duplicate and
    re-rank the union against the original combined intent."""
    hybrid = build_hybrid_retriever(k=config.TOP_K_CANDIDATES)
    seen = {}
    for sq in sub_queries:
        for d in hybrid.invoke(sq):
            key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:80])
            seen[key] = d
    merged_query = " ".join(sub_queries)
    return rerank(merged_query, list(seen.values()), top_k=final_k)
