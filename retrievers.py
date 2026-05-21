%%writefile retrievers.py
import bm25s
import faiss
import numpy as np

class BM25Retriever:
    def __init__(self, texts, truncate_len):
        truncated = [str(t).lower()[:truncate_len] for t in texts]
        tokens = bm25s.tokenize(truncated, stopwords='de')
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens)

    def search(self, query, top_k):
        query_tokens = bm25s.tokenize([query.lower()], stopwords='de')
        docs, scores = self.retriever.retrieve(query_tokens, k=top_k)
        return [(int(docs[0][i]), float(scores[0][i])) for i in range(len(docs[0]))]

class FaissRetriever:
    def __init__(self, embeddings):
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))

    def search(self, query_embedding, top_k):
        D, I = self.index.search(query_embedding.reshape(1, -1), top_k)
        return [(int(I[0][i]), float(D[0][i])) for i in range(len(I[0]))]

def reciprocal_rank_fusion(sparse_results, dense_results, k=60):
    scores = {}
    for rank, (idx, _) in enumerate(sparse_results):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
    for rank, (idx, _) in enumerate(dense_results):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
