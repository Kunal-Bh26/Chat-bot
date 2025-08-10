import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import difflib


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
        self._embedder: Any | None = None
        self._nn: NearestNeighbors | None = None
        self._question_embeddings: np.ndarray | None = None

    # ---------- Public API ----------
    def warmup(self) -> None:
        """Eagerly load dataset, model and embeddings so the first request is fast.

        This method is safe to call multiple times; subsequent calls are no-ops
        once resources are initialized.
        """
        df, embedder, _ = self._ensure_resources()
        # Force a tiny embed so the ONNX session is fully initialized
        _ = self._embed_texts(embedder, ["warmup"])  # noqa: F841

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

        df, embedder, nn = self._ensure_resources()

        # 1) Fuzzy match against questions
        questions = df["questions"].astype(str).tolist()
        best_question, score, index = self._best_fuzzy_match(user_query, questions)
        if best_question is not None and index is not None:
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
        query_embedding = self._embed_texts(embedder, [user_query])
        distances, indices = nn.kneighbors(query_embedding)
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
    def _ensure_resources(self) -> Tuple[pd.DataFrame, Any, NearestNeighbors]:
        # Double-checked locking to avoid duplicate heavy loads
        if self._df is not None and self._embedder is not None and self._nn is not None:
            return self._df, self._embedder, self._nn

        with self._lock:
            if self._df is not None and self._embedder is not None and self._nn is not None:
                return self._df, self._embedder, self._nn

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

            try:
                # Runtime import to avoid linter "unresolved import" noise locally
                from fastembed import TextEmbedding as _TextEmbedding  # type: ignore
            except Exception as _exc:
                raise RuntimeError(
                    "FastEmbed is required. Ensure 'fastembed' is installed to run on Railway free tier."
                ) from _exc

            # Small, high-quality English embedding model with low memory footprint
            # Ref: https://github.com/qdrant/fastembed (models: BAAI/bge-small-en-v1.5, thenlper/gte-small, etc.)
            embedder = _TextEmbedding("BAAI/bge-small-en-v1.5")

            question_texts = df["questions"].astype(str).tolist()

            # Persist embeddings next to the dataset to avoid recomputation on cold starts
            embeddings_cache_path = f"{self.knowledge_base_path}.embeddings.npy"
            if os.path.exists(embeddings_cache_path):
                embeddings = np.load(embeddings_cache_path)
            else:
                embeddings = self._embed_texts(embedder, question_texts)
                # Safe write via temp file then atomic replace
                tmp_path = f"{embeddings_cache_path}.tmp"
                np.save(tmp_path, embeddings)
                os.replace(tmp_path, embeddings_cache_path)

            nn = NearestNeighbors(n_neighbors=1, metric="cosine")
            nn.fit(embeddings)

            self._df = df
            self._embedder = embedder
            self._nn = nn
            self._question_embeddings = embeddings

            return self._df, self._embedder, self._nn

    @staticmethod
    def _embed_texts(embedder: Any, texts: list[str]) -> np.ndarray:
        # fastembed returns a generator of embeddings; convert to numpy array
        vectors = list(embedder.embed(texts))
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def _best_fuzzy_match(query: str, choices: list[str]) -> Tuple[str | None, float, int | None]:
        # Use difflib to avoid external dependency; compute best ratio and index
        if not choices:
            return None, 0.0, None
        best_score = -1.0
        best_index = None
        best_choice = None
        for idx, candidate in enumerate(choices):
            score = difflib.SequenceMatcher(None, query.lower(), candidate.lower()).ratio() * 100.0
            if score > best_score:
                best_score = score
                best_index = idx
                best_choice = candidate
        return best_choice, float(best_score), int(best_index) if best_index is not None else None

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
        # Use difflib for lightweight approximate matching
        best_choice = None
        best_score = 0.0
        for candidate in greetings:
            score = difflib.SequenceMatcher(None, lowered, candidate).ratio() * 100.0
            if score > best_score:
                best_score = score
                best_choice = candidate
        if best_choice is not None and best_score >= 85.0:
            return best_choice
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


