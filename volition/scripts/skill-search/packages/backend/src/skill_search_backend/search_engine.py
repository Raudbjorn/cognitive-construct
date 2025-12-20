"""Vector search engine for finding relevant skills."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .result import Err, Ok, Result
from .types import SearchError, SearchResult, Skill

logger = logging.getLogger(__name__)


class SkillSearchEngine:
    """Search engine for finding relevant skills using vector similarity.

    This is a mutable class because it maintains state (skills, embeddings, model).

    Attributes
    ----------
    model : SentenceTransformer | None
        Embedding model for generating vectors (lazy-loaded).
    model_name : str
        Name of the sentence-transformers model to use.
    skills : list[Skill]
        List of indexed skills.
    embeddings : np.ndarray | None
        Embeddings matrix for all skill descriptions.
    _lock : threading.Lock
        Lock for thread-safe access to skills and embeddings.
    """

    def __init__(self, model_name: str):
        """Initialize the search engine.

        Parameters
        ----------
        model_name : str
            Name of the sentence-transformers model to use.
        """
        logger.info(
            f"Search engine initialized (model: {model_name}, lazy-loading enabled)"
        )
        self.model: SentenceTransformer | None = None
        self.model_name = model_name
        self.skills: list[Skill] = []
        self.embeddings: np.ndarray | None = None
        self._lock = threading.Lock()

    def _ensure_model_loaded(self) -> Result[SentenceTransformer, SearchError]:
        """Ensure the embedding model is loaded (lazy initialization).

        Returns
        -------
        Result[SentenceTransformer, SearchError]
            Ok with the loaded model or Err if loading failed.
        """
        if self.model is not None:
            return Ok(self.model)

        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded: {self.model_name}")
            return Ok(self.model)
        except Exception as e:
            error_msg = f"Failed to load embedding model: {e}"
            logger.error(error_msg)
            return Err(SearchError(message=error_msg))

    def index_skills(self, skills: list[Skill]) -> Result[int, SearchError]:
        """Index a list of skills by generating their embeddings.

        Parameters
        ----------
        skills : list[Skill]
            Skills to index.

        Returns
        -------
        Result[int, SearchError]
            Ok with number of skills indexed or Err if indexing failed.
        """
        with self._lock:
            if not skills:
                logger.warning("No skills to index")
                self.skills = []
                self.embeddings = None
                return Ok(0)

            logger.info(f"Indexing {len(skills)} skills...")
            self.skills = skills

            # Generate embeddings from skill descriptions
            descriptions = [skill.description for skill in skills]

            model_result = self._ensure_model_loaded()
            if model_result.is_err():
                return Err(model_result.error)

            model = model_result.value

            try:
                self.embeddings = model.encode(descriptions, convert_to_numpy=True)
                logger.info(f"Successfully indexed {len(skills)} skills")
                return Ok(len(skills))
            except Exception as e:
                error_msg = f"Failed to generate embeddings: {e}"
                logger.error(error_msg)
                return Err(SearchError(message=error_msg))

    def add_skills(self, skills: list[Skill]) -> Result[int, SearchError]:
        """Add skills incrementally and update embeddings.

        Parameters
        ----------
        skills : list[Skill]
            Skills to add to the index.

        Returns
        -------
        Result[int, SearchError]
            Ok with total number of skills or Err if adding failed.
        """
        if not skills:
            return Ok(len(self.skills))

        with self._lock:
            logger.info(f"Adding {len(skills)} skills to index...")

            descriptions = [skill.description for skill in skills]

            model_result = self._ensure_model_loaded()
            if model_result.is_err():
                return Err(model_result.error)

            model = model_result.value

            try:
                new_embeddings = model.encode(descriptions, convert_to_numpy=True)

                self.skills.extend(skills)

                if self.embeddings is None:
                    self.embeddings = new_embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])

                logger.info(
                    f"Successfully added {len(skills)} skills. Total: {len(self.skills)} skills"
                )
                return Ok(len(self.skills))

            except Exception as e:
                error_msg = f"Failed to add skills: {e}"
                logger.error(error_msg)
                return Err(SearchError(message=error_msg))

    def search(
        self, query: str, top_k: int = 3
    ) -> Result[list[SearchResult], SearchError]:
        """Search for the most relevant skills based on a query.

        Parameters
        ----------
        query : str
            The task description or query to search for.
        top_k : int, optional
            Number of top results to return, by default 3.

        Returns
        -------
        Result[list[SearchResult], SearchError]
            Ok with list of SearchResult or Err if search failed.
        """
        with self._lock:
            if not self.skills or self.embeddings is None:
                logger.warning("No skills indexed, returning empty results")
                return Ok([])

            top_k = min(top_k, len(self.skills))

            logger.info(f"Searching for: '{query}' (top_k={top_k})")

            model_result = self._ensure_model_loaded()
            if model_result.is_err():
                return Err(model_result.error)

            model = model_result.value

            try:
                query_embedding = model.encode([query], convert_to_numpy=True)[0]

                similarities = self._cosine_similarity(query_embedding, self.embeddings)

                top_indices = np.argsort(similarities)[::-1][:top_k]

                results: list[SearchResult] = []
                for idx in top_indices:
                    skill = self.skills[idx]
                    score = float(similarities[idx])

                    result = SearchResult(
                        name=skill.name,
                        description=skill.description,
                        content=skill.content,
                        source=skill.source,
                        relevance_score=score,
                        documents=skill.documents,
                    )
                    results.append(result)

                    logger.debug(f"Found skill: {skill.name} (score: {score:.4f})")

                logger.info(f"Returning {len(results)} results")
                return Ok(results)

            except Exception as e:
                error_msg = f"Search failed: {e}"
                logger.error(error_msg)
                return Err(SearchError(message=error_msg, query=query))

    def search_dict(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search for skills and return results as dictionaries.

        This method maintains backward compatibility with code expecting dict results.

        Parameters
        ----------
        query : str
            The task description or query to search for.
        top_k : int, optional
            Number of top results to return, by default 3.

        Returns
        -------
        list[dict[str, Any]]
            List of skill dictionaries with relevance scores.
        """
        result = self.search(query, top_k)

        if result.is_err():
            logger.error(f"Search error: {result.error.message}")
            return []

        return [
            {
                "name": r.name,
                "description": r.description,
                "content": r.content,
                "source": r.source,
                "relevance_score": r.relevance_score,
                "documents": r.documents,
            }
            for r in result.value
        ]

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and a matrix of vectors.

        Parameters
        ----------
        vec : np.ndarray
            Query vector.
        matrix : np.ndarray
            Matrix of vectors to compare against.

        Returns
        -------
        np.ndarray
            Similarity scores.
        """
        vec_norm = vec / np.linalg.norm(vec)
        matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

        similarities = np.dot(matrix_norm, vec_norm)

        return similarities
