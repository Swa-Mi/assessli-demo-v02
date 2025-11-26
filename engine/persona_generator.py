
# Converts clustered behavior data into human-readable personality personas.

from collections import Counter
import numpy as np


class PersonaGenerator:
    """
    Generates simple personas from clustered text statements.
    No LLMs used (rule-based summarization).
    """

    def __init__(self):
        print("[PersonaGenerator] Persona generator initialized.")

    def _extract_keywords(self, statements):
        """
        Extracts frequently appearing keywords from cluster statements.
        Very primitive but works well for rule-based summarization.
        """
        words = " ".join(statements).lower().split()

        blacklist = {
            "i", "and", "the", "to", "a", "in", "my", "on",
            "for", "with", "of", "am", "that", "this", "you",
            "your", "but", "was", "are", "have", "has"
        }

        cleaned = [
            w.strip(",.!?") for w in words
            if w not in blacklist and len(w) > 3
        ]

        most_common = [w for w, _ in Counter(cleaned).most_common(5)]
        return most_common

    def _generate_title(self, keywords):
        """
        Creates a persona title using top keywords.
        """
        if not keywords:
            return "General Thinker"

        return f"{keywords[0].capitalize()} Oriented Personality"

    def generate_personas(self, cluster_dict):
        """
        Takes cluster_id -> [statements] and generates persona summaries.
        Designed to work with KMeans (no outlier labels).
        """

        personas = {}

        for cluster_id, statements in cluster_dict.items():

            # Handle extremely small clusters (1–2 statements)
            if len(statements) <= 2:
                personas[cluster_id] = {
                    "title": "Micro-Trait Cluster",
                    "keywords": self._extract_keywords(statements),
                    "description": (
                        "This tiny cluster represents very specific or unique traits. "
                        "Because it contains only 1–2 statements, it captures sharp individual preferences "
                        "rather than broad behavioral patterns."
                    )
                }
                continue

            keywords = self._extract_keywords(statements)
            title = self._generate_title(keywords)

            description = (
                f"This persona reflects a **{title}**. "
                f"Key recurring ideas include: {', '.join(keywords)}. "
                f"The statements show a consistent theme unique to Cluster {cluster_id}."
            )

            personas[cluster_id] = {
                "title": title,
                "keywords": keywords,
                "description": description
            }

        return personas
