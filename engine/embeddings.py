# embeddings.py
# Embedding engine using pure HuggingFace Transformers (no sentence-transformers)

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class EmbeddingEngine:
    """
    Generates embeddings using a lightweight HuggingFace model.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Loads tokenizer + model.
        """
        print(f"[EmbeddingEngine] Loading HF model: {model_name} ...")
        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("[EmbeddingEngine] Model loaded successfully.")

    def encode(self, sentences):
        """
        Generate embeddings using mean pooling.
        """
        if not isinstance(sentences, list):
            raise ValueError("Input must be a list of sentences.")

        inputs = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings

    def encode_single(self, sentence: str):
        """
        Encode one sentence.
        """
        return self.encode([sentence])[0]
