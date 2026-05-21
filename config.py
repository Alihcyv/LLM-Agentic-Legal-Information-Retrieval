%%writefile config.py
import torch

class Config:
    EMBEDDING_MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
    RERANKER_MODEL_NAME = 'Qwen/Qwen3-Reranker-0.6B'
    TASK_INSTRUCTION = 'Given a legal question, retrieve relevant Swiss law articles and court decisions.'

    TOP_K_RETRIEVAL = 60
    TOP_K_FINAL = 10
    TEXT_TRUNCATE = 3500
    EMBEDDING_LEN = 512
    RRF_K = 60
    LAWS_EXPANSION_TOP_K = 3
    EMBEDDING_BATCH_SIZE = 16
    RERANKER_BATCH_SIZE = 8

    MAIN_PATH = '/kaggle/input/competitions/llm-agentic-legal-information-retrieval'
    TRAIN_PATH = f'{MAIN_PATH}/train.csv'
    VAL_PATH = f'{MAIN_PATH}/val.csv'
    TEST_PATH = f'{MAIN_PATH}/test.csv'
    LAWS_PATH = f'{MAIN_PATH}/laws_de.csv'
    COURT_PATH = f'{MAIN_PATH}/court_considerations.csv'
    OUTPUT_PATH = 'submission.csv'

    DEVICE_EMB = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    DEVICE_RERANK = 'cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0' if torch.cuda.is_available() else 'cpu'
