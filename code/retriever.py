"""
TF-IDF Retriever: Finds most relevant corpus documents for a support ticket.
"""

from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:
    """
    Builds a TF-IDF index over the corpus and retrieves top-k docs
    for a given query, optionally filtered by company.
    """

    def __init__(self, corpus: List[Dict]):
        self.corpus = corpus
        self._build_index()

    def _build_index(self):
        texts = [doc["full_text"] for doc in self.corpus]
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Retrieve top_k most relevant documents for the query.
        If company is provided, give 2x weight to docs matching that company.
        Index/navigation-only files are penalized and kept only as fallback.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Boost scores for matching company
        if company and company != "None":
            for i, doc in enumerate(self.corpus):
                if doc["company"].lower() == company.lower():
                    scores[i] *= 2.0

        # Penalize index/navigation files - they are tables-of-contents, not answers
        for i, doc in enumerate(self.corpus):
            fname = doc["id"].split("/")[-1].lower()
            if fname in ("index.md", "support.md") or doc["title"].lower() in ("index", ""):
                scores[i] *= 0.1

        top_indices = scores.argsort()[::-1][:top_k * 5]  # get more, then filter

        results = []
        for idx in top_indices:
            if scores[idx] < 0.01:
                break
            doc = dict(self.corpus[idx])
            doc["score"] = float(scores[idx])
            results.append(doc)
            if len(results) >= top_k:
                break

        return results
