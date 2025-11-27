# embeddings.py
# Lightweight ONNX-based embedding engine for Streamlit Cloud

from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingEngine:
    """
    Fast, lightweight embedding engine using an ONNX-optimized MiniLM model.
    Works on Streamlit Cloud (torch not required).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[EmbeddingEngine] Loading ONNX model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            device="cpu",
            use_auth_token=None
        )
        print("[EmbeddingEngine] Model loaded successfully (ONNX).")

    def encode(self, sentences):
        if not isinstance(sentences, list):
            raise ValueError("Input must be a list of strings.")
        
        embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings

    def encode_single(self, sentence: str):
        return self.encode([sentence])[0]
