"""
Vector Index and RAG Retrieval for Flipkart Support Agent (Part 3 Task 2 & Task 10).
Uses sentence-transformers (all-MiniLM-L6-v2) and Faiss (free, local vector indexing).
"""

import os
import json
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from part3_knowledge_base import POLICY_DOCUMENTS, RETRIEVAL_EVAL_ANSWER_KEY, chunk_policy_documents

_embedding_model_cache = None

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _embedding_model_cache
    if _embedding_model_cache is None:
        _embedding_model_cache = SentenceTransformer(model_name)
    return _embedding_model_cache

class PolicyVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = get_embedding_model(model_name)
        self.chunks = chunk_policy_documents(POLICY_DOCUMENTS)
        self.texts = [c["text"] for c in self.chunks]
        
        # Compute normalized embeddings for cosine similarity (Inner Product on normalized vectors)
        embeddings = self.model.encode(self.texts, show_progress_bar=False, normalize_embeddings=True)
        self.dimension = embeddings.shape[1]
        
        # Build Faiss IndexFlatIP (cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))
        
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for top_k relevant chunks given a query.
        Returns list of chunks with similarity score.
        """
        query_vec = self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_vec, dtype=np.float32), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk_data = dict(self.chunks[idx])
            chunk_data["similarity_score"] = float(score)
            results.append(chunk_data)
        return results

    def search_parent_docs(self, query: str, top_k_chunks: int = 5, top_k_docs: int = 3) -> List[Dict]:
        """
        Search and aggregate back to parent document level.
        Preserves ranking based on highest chunk similarity.
        """
        chunk_results = self.search(query, top_k=top_k_chunks)
        seen_docs = set()
        doc_results = []
        for cr in chunk_results:
            doc_id = cr["doc_id"]
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                doc_results.append({
                    "doc_id": doc_id,
                    "doc_title": cr["doc_title"],
                    "category": cr["category"],
                    "best_score": cr["similarity_score"],
                    "full_doc_content": cr["full_doc_content"]
                })
            if len(doc_results) >= top_k_docs:
                break
        return doc_results

def evaluate_retrieval(top_k: int = 3) -> Dict:
    """
    Task 10: Evaluate retrieval using Precision@k and Recall@k on ground-truth query-document pairs.
    """
    vector_store = PolicyVectorStore()
    eval_records = []
    
    total_precision = 0.0
    total_recall = 0.0
    
    for item in RETRIEVAL_EVAL_ANSWER_KEY:
        query_id = item["query_id"]
        query = item["query"]
        ground_truth_docs = set(item["relevant_docs"])
        
        # Retrieve top_k parent docs
        retrieved_docs_meta = vector_store.search_parent_docs(query, top_k_chunks=10, top_k_docs=top_k)
        retrieved_doc_ids = [d["doc_id"] for d in retrieved_docs_meta]
        
        # Compute intersection
        relevant_and_retrieved = [doc_id for doc_id in retrieved_doc_ids if doc_id in ground_truth_docs]
        num_relevant_retrieved = len(relevant_and_retrieved)
        
        # Precision@k = |Retrieved ∩ Relevant| / k
        precision_k = num_relevant_retrieved / top_k
        
        # Recall@k = |Retrieved ∩ Relevant| / |Relevant|
        recall_k = num_relevant_retrieved / len(ground_truth_docs)
        
        total_precision += precision_k
        total_recall += recall_k
        
        eval_records.append({
            "query_id": query_id,
            "query": query,
            "ground_truth_docs": list(ground_truth_docs),
            "retrieved_docs": retrieved_doc_ids,
            "relevant_retrieved": relevant_and_retrieved,
            "precision_at_k": precision_k,
            "recall_at_k": recall_k,
            "arithmetic_precision": f"{num_relevant_retrieved}/{top_k} = {precision_k:.4f}",
            "arithmetic_recall": f"{num_relevant_retrieved}/{len(ground_truth_docs)} = {recall_k:.4f}"
        })
        
    num_queries = len(RETRIEVAL_EVAL_ANSWER_KEY)
    mean_precision = total_precision / num_queries
    mean_recall = total_recall / num_queries
    
    return {
        "top_k": top_k,
        "num_queries": num_queries,
        "eval_records": eval_records,
        "mean_precision_at_k": mean_precision,
        "mean_recall_at_k": mean_recall
    }

if __name__ == "__main__":
    results = evaluate_retrieval(top_k=3)
    print("\n" + "="*70)
    print("TASK 10 RETRIEVAL EVALUATION (Precision@3 and Recall@3)")
    print("="*70)
    for r in results["eval_records"]:
        print(f"\n[{r['query_id']}] Query: '{r['query']}'")
        print(f"  Ground Truth: {r['ground_truth_docs']}")
        print(f"  Retrieved:    {r['retrieved_docs']}")
        print(f"  P@3: {r['arithmetic_precision']}")
        print(f"  R@3: {r['arithmetic_recall']}")
        
    print("\n" + "-"*70)
    print(f"Average Precision@3: {results['mean_precision_at_k']:.4f}")
    print(f"Average Recall@3:    {results['mean_recall_at_k']:.4f}")
    print("="*70)
