%%writefile main.py
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import gc
import torch

from config import Config
from data_utils import load_datasets, prepare_texts
from models import EmbeddingManager, Qwen3Reranker
from retrievers import BM25Retriever, FaissRetriever
from pipeline import LegalRetrievalPipeline, compute_macro_f1

def main():
    cfg = Config()
    
    train_df, val_df, test_df, laws_df, court_df = load_datasets(cfg)
    laws_texts, court_texts, law_map, court_map = prepare_texts(laws_df, court_df)
    
    emb_manager = EmbeddingManager(cfg)
    reranker = Qwen3Reranker(cfg)
    
    bm25_laws = BM25Retriever(laws_texts, cfg.TEXT_TRUNCATE)
    bm25_courts = BM25Retriever(court_texts, cfg.TEXT_TRUNCATE)
    
    print("Encoding laws...")
    laws_emb = emb_manager.encode_corpus(laws_texts, cfg.EMBEDDING_BATCH_SIZE, cfg.TEXT_TRUNCATE)
    faiss_laws = FaissRetriever(laws_emb)
    
    del laws_emb
    gc.collect()
    torch.cuda.empty_cache()
    
    pipeline = LegalRetrievalPipeline(cfg, emb_manager, reranker, bm25_laws, bm25_courts, faiss_laws, laws_df, court_df, law_map, court_map)
    
    print("Running Validation...")
    val_queries = val_df['query'].tolist()
    val_embs = emb_manager.encode_queries(val_queries)
    
    val_preds, val_gold = [], []
    for i, row in val_df.iterrows():
        val_preds.append(pipeline.retrieve(row['query'], val_embs[i]))
        val_gold.append([c.strip() for c in str(row.get('gold_citations', '')).split(';') if c.strip()])
    
    print(f"Validation Macro F1: {compute_macro_f1(val_preds, val_gold):.5f}")
    
    print("Running Test...")
    test_queries = test_df['query'].tolist()
    test_embs = emb_manager.encode_queries(test_queries)
    
    results = []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        preds = pipeline.retrieve(row['query'], test_embs[i])
        results.append({'query_id': row['query_id'], 'predicted_citations': ';'.join(preds)})
        
    pd.DataFrame(results).to_csv(cfg.OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()
