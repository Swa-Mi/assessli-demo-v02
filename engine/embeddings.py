# embeddings.py
# Lightweight embedding engine without sentence-transformers or torch

from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch


class EmbeddingEngine:
    """
    Generates sentence embeddings using HuggingFace MiniLM model.
    Works on Streamlit Cloud (CPU-only).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[EmbeddingEngine] Loading transformer model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        print("[EmbeddingEngine] Model loaded successfully!")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def encode(self, sentences):
        if not isinstance(sentences, list):
            raise ValueError("Input must be a list of sentences.")

        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            model_output = self.model(**encoded)

        embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def encode_single(self, sentence: str):
        return self.encode([sentence])[0]
