import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import process, fuzz


@dataclass
class QAMeta:
    strategy: str
    score: float
    best_question: str
    distance: float


class QASystem:
    def __init__(self, knowledge_base_path: str) -> None:
        self.knowledge_base_path = knowledge_base_path
        self._lock = threading.Lock()

        # Lazy-loaded resources
        self._df: pd.DataFrame | None = None
        self._model: SentenceTransformer | None = None
        self._nn: NearestNeighbors | None = None
        self._question_embeddings: np.ndarray | None = None

    # ---------- Public API ----------
    def get_response(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        user_query = user_query.strip()
        if self._is_gibberish(user_query):
            return (
                "I'm sorry, I couldn't understand that. Could you please rephrase your question?",
                {"strategy": "filter:gibberish"},
            )

        # Greetings shortcut
        greet = self._match_greeting(user_query)
        if greet is not None:
            return self._greeting_response(greet)

        df, model, nn = self._ensure_resources()

        # 1) Fuzzy match against questions
        questions = df["questions"].astype(str).tolist()
        best_match = process.extractOne(
            user_query, questions, scorer=fuzz.token_sort_ratio
        )
        if best_match:
            best_question, score, index = best_match[0], best_match[1], best_match[2]
            if score > 70:
                answer = df.iloc[index]["answers"]
                return (
                    str(answer),
                    {
                        "strategy": "fuzzy",
                        "score": float(score),
                        "best_question": str(best_question),
                        "distance": 0.0,
                    },
                )

        # 2) Semantic nearest neighbor search
        query_embed = model.encode([user_query])
        distances, indices = nn.kneighbors(query_embed)
        best_idx = int(indices[0][0])
        best_dist = float(distances[0][0])

        if best_dist > 0.45:
            return (
                "I'm sorry, I couldn't understand that. Could you please rephrase your question?",
                {
                    "strategy": "semantic:low-confidence",
                    "score": 0.0,
                    "best_question": str(df.iloc[best_idx]["questions"]),
                    "distance": best_dist,
                },
            )

        return (
            str(df.iloc[best_idx]["answers"]),
            {
                "strategy": "semantic",
                "score": 1.0 - best_dist,  # Higher is better
                "best_question": str(df.iloc[best_idx]["questions"]),
                "distance": best_dist,
            },
        )

    # ---------- Internal ----------
    def _ensure_resources(self) -> Tuple[pd.DataFrame, SentenceTransformer, NearestNeighbors]:
        # Double-checked locking to avoid duplicate heavy loads
        if self._df is not None and self._model is not None and self._nn is not None:
            return self._df, self._model, self._nn

        with self._lock:
            if self._df is not None and self._model is not None and self._nn is not None:
                return self._df, self._model, self._nn

            if not os.path.exists(self.knowledge_base_path):
                raise FileNotFoundError(
                    f"Knowledge base not found at '{self.knowledge_base_path}'. Place 'dataset.csv' next to the backend app or set KNOWLEDGE_BASE_PATH."
                )

            # Load CSV (expected columns: questions, answers, categories, tags)
            df = pd.read_csv(self.knowledge_base_path)
            required_columns = {"questions", "answers", "categories", "tags"}
            missing = required_columns.difference(df.columns)
            if missing:
                raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

            model = SentenceTransformer("all-MiniLM-L6-v2")

            question_texts = df["questions"].astype(str).tolist()
            embeddings = model.encode(question_texts)
            embeddings = np.asarray(embeddings)

            nn = NearestNeighbors(n_neighbors=1, metric="cosine")
            nn.fit(embeddings)

            self._df = df
            self._model = model
            self._nn = nn
            self._question_embeddings = embeddings

            return self._df, self._model, self._nn

    @staticmethod
    def _is_gibberish(text: str) -> bool:
        text = text.strip()
        if len(text) < 2 or re.fullmatch(r"[^\w\s]+", text) or len(set(text)) < 3:
            return True
        words = text.split()
        if len(words) > 0 and sum(1 for w in words if not w.isalpha()) / len(words) > 0.5:
            return True
        return False

    @staticmethod
    def _match_greeting(text: str) -> str | None:
        greetings = [
            "hello",
            "hi",
            "hey",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "what's up",
            "sup",
            "thank you",
            "thanks",
            "bye",
            "goodbye",
        ]
        lowered = text.lower()
        best = process.extractOne(lowered, greetings, scorer=fuzz.WRatio)
        if best and best[1] >= 85:
            return str(best[0])
        return None

    @staticmethod
    def _greeting_response(greet: str) -> Tuple[str, Dict[str, Any]]:
        responses = {
            "hello": "Hello! 👋 How can I help you today?",
            "hi": "Hi there! How can I assist you?",
            "hey": "Hey! How can I help you?",
            "greetings": "Greetings! How can I help you?",
            "good morning": "Good morning! ☀️ How can I help?",
            "good afternoon": "Good afternoon! How can I help?",
            "good evening": "Good evening! How can I help?",
            "how are you": "I'm just a bot, but I'm here to help you! 😊",
            "what's up": "I'm here to help with your IT queries!",
            "sup": "All good! How can I assist you?",
            "thank you": "You're welcome! Let me know if you have more questions.",
            "thanks": "You're welcome!",
            "bye": "Thank you for chatting, Mata Ne! (see you later) 👋",
            "goodbye": "Thank you for chatting, Mata Ne! (see you later) 👋",
        }
        return responses.get(greet, "Hello! How can I help you?"), {"strategy": "greeting", "match": greet}


