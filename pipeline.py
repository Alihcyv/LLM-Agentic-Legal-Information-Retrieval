%%writefile pipeline.py

from retrievers import reciprocal_rank_fusion
import numpy as np


class LegalRetrievalPipeline:
    def __init__(self, config, emb_manager, reranker, bm25_laws, bm25_courts, faiss_laws, laws_df, court_df, law_map, court_map):
        self.config = config
        self.emb_manager = emb_manager
        self.reranker = reranker
        self.bm25_laws = bm25_laws
        self.bm25_courts = bm25_courts
        self.faiss_laws = faiss_laws
        self.laws_df = laws_df
        self.court_df = court_df
        self.law_map = law_map
        self.court_map = court_map

    def retrieve(self, query, query_embedding):
        sparse_laws = self.bm25_laws.search(query, self.config.TOP_K_RETRIEVAL)
        dense_laws = self.faiss_laws.search(query_embedding, self.config.TOP_K_RETRIEVAL)
        fused_laws = reciprocal_rank_fusion(sparse_laws, dense_laws, self.config.RRF_K)
        
        law_citations = [self.laws_df.iloc[idx]['citation'] for idx, _ in fused_laws[:self.config.TOP_K_RETRIEVAL]]

        expansion = ' '.join(law_citations[:self.config.LAWS_EXPANSION_TOP_K])
        expanded_query = f'{query} {expansion}' if expansion else query
        court_results = self.bm25_courts.search(expanded_query, self.config.TOP_K_RETRIEVAL)
        court_citations = [self.court_df.iloc[idx]['citation'] for idx, _ in court_results]

        all_cits = list(dict.fromkeys(law_citations + court_citations))[:self.config.TOP_K_RETRIEVAL]
        candidates = []
        for cit in all_cits:
            text = self.law_map.get(cit) or self.court_map.get(cit, '')
            if text: candidates.append((cit, text))

        if not candidates: return []
        
        pairs = [[query, text] for _, text in candidates]
        scores = self.reranker.predict(pairs, self.config.RERANKER_BATCH_SIZE, self.config.TASK_INSTRUCTION, self.config.TEXT_TRUNCATE)
        
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [cit for (cit, _), sc in reranked[:self.config.TOP_K_FINAL]]


def compute_macro_f1(all_predicted, all_gold):
    f1_scores = []
    for pred, gold in zip(all_predicted, all_gold):
        pred_set, gold_set = set(pred), set(gold)
        if not gold_set: continue
        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gold_set) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return np.mean(f1_scores) if f1_scores else 0.0
