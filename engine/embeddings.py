# embeddings.py
# Lightweight embedding engine using TF-IDF (Streamlit Cloud compatible)

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class EmbeddingEngine:
    """
    Simple TF-IDF embedding engine fully compatible with Streamlit Cloud.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000  # adjust depending on dataset size
        )

    def fit(self, texts):
        """
        Fit vectorizer to dataset.
        """
        self.vectorizer.fit(texts)

    def encode(self, texts):
        """
        Transform list of sentences into TF-IDF vectors.
        """
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of sentences.")
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings

    def encode_single(self, text: str):
        return self.encode([text])[0]
