# embeddings.py
# Lightweight embedding engine using HuggingFace transformers

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


class EmbeddingEngine:
    """
    Embedding engine using HuggingFace MiniLM model without sentence_transformers.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Load tokenizer + model.
        """
        print(f"[EmbeddingEngine] Loading HF model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        print("[EmbeddingEngine] Model loaded successfully (HF).")

    def _mean_pool(self, model_output, attention_mask):
        """
        Mean Pooling - take attention mask into account.
        """
        token_embeddings = model_output[0]  # First element is the token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, sentences):
        """
        Encode a list of sentences into embeddings.
        """
        if not isinstance(sentences, list):
            raise ValueError("Input must be a list of sentences.")

        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = self._mean_pool(model_output, encoded_input['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def encode_single(self, sentence: str):
        """
        Encode a single sentence.
        """
        return self.encode([sentence])[0]
