%%writefile models.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, config):
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME,
            device=config.DEVICE_EMB,
            model_kwargs={'torch_dtype': torch.float16},
        )
        self.model.max_seq_length = config.EMBEDDING_LEN
        self.instruction = config.TASK_INSTRUCTION

    def encode_queries(self, queries):
        prefixed = [f'Instruct: {self.instruction}\nQuery: {q}' for q in queries]
        return self.model.encode(prefixed, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)

    def encode_corpus(self, texts, batch_size, truncate_len):
        truncated = [str(t)[:truncate_len] for t in texts]
        return self.model.encode(truncated, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)

class Qwen3Reranker:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.RERANKER_MODEL_NAME, padding_side='left', trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.RERANKER_MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True
        ).to(config.DEVICE_RERANK).eval()
        
        self.device = config.DEVICE_RERANK
        self.yes_id = self.tokenizer.convert_tokens_to_ids('yes')
        self.no_id = self.tokenizer.convert_tokens_to_ids('no')
        self.system_prompt = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'

    def predict(self, pairs, batch_size, instruction, truncate_len):
        all_scores = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            prompts = [f'<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d[:truncate_len]}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' for q, d in batch]
            
            inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=4096, return_tensors='pt').to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]
                yes_no = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
                probs = F.softmax(yes_no, dim=1)
                all_scores.extend(probs[:, 1].cpu().numpy().tolist())
        return all_scores
