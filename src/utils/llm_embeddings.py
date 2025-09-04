"""
Module for working with OpenAI Embeddings API.

Provides functions for obtaining vector representations of text
and calculating cosine similarity between vectors.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import tiktoken
from openai import OpenAI

# Logging setup
logger = logging.getLogger(__name__)


class EmbeddingsClient:
    """
    Client for working with OpenAI Embeddings API.

    Features:
    - Batch processing (up to 2048 texts per request)
    - TPM limit control (350,000 tokens/min)
    - Long text handling (up to 8192 tokens)
    - Retry logic with exponential backoff
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Client initialization.

        Args:
            config: Configuration dictionary with keys:
                - embedding_api_key: API key (optional, fallback to api_key)
                - api_key: Main API key
                - embedding_model: Model for embeddings
                - embedding_tpm_limit: TPM limit (default 350000)
                - max_retries: Number of retry attempts (default 3)
        """
        # Routing: external OpenAI vs internal HTTP service
        # Flags (by analogy with LLM internal routing)
        self.use_internal_auth: bool = bool(config.get("embedding_use_internal_auth", False))
        self.base_url: Optional[str] = config.get("embedding_base_url")

        # API key / OAuth token resolution
        # - Internal: use dedicated INTERNAL_EMBEDDING_API_KEY env or embedding_api_key from config
        # - External: OpenAI API key (embedding_api_key preferred, fallback to api_key)
        if self.use_internal_auth and self.base_url:
            resolved_key = config.get("embedding_api_key") or os.getenv("INTERNAL_EMBEDDING_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "Internal embeddings token not found (dedup/refiner.embedding_api_key or env INTERNAL_EMBEDDING_API_KEY)"
                )
            # Internal endpoint over HTTP; store token and skip OpenAI client
            self.client = None
            self.internal_oauth_token = resolved_key
        else:
            resolved_key = config.get("embedding_api_key") or config.get("api_key")
            if not resolved_key:
                raise ValueError(
                    "API key not found for external embeddings (embedding_api_key or api_key)"
                )
            # Default to external OpenAI embeddings API
            self.client = OpenAI(api_key=resolved_key)

        self.model = config.get("embedding_model", "text-embedding-3-small")

        # TPM control
        self.tpm_limit = config.get("embedding_tpm_limit", 1000000)
        self.remaining_tokens = self.tpm_limit
        self.reset_time = time.time()

        # Retry parameters
        self.max_retries = config.get("max_retries", 3)

        # Batch processing parameters from config
        self.max_batch_tokens = config.get("max_batch_tokens", 100000)
        self.max_texts_per_batch = config.get("max_texts_per_batch", 2048)
        self.truncate_tokens = config.get("truncate_tokens", 8000)

        # Tokenizer for counting tokens (cl100k_base for embedding models)
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # API limits
        self.max_texts_per_request = self.max_texts_per_batch
        self.max_tokens_per_text = 8192

        # Embedding dimensionality
        # - External OpenAI returns 1536 dims
        # - Internal service dimension is determined from first response
        self.embedding_dim: Optional[int] = 1536 if not self.use_internal_auth else None

        logger.info(
            f"EmbeddingsClient initialized with model={self.model}, tpm_limit={self.tpm_limit}, "
            f"max_batch_tokens={self.max_batch_tokens}, max_texts_per_batch={self.max_texts_per_batch}"
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncate text to specified number of tokens."""
        if max_tokens is None:
            max_tokens = self.truncate_tokens

        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        # Truncate and decode back
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

    def _update_tpm_state(self, tokens_used: int, headers: Optional[Dict[str, str]] = None):
        """
        Update TPM bucket state.

        Args:
            tokens_used: Number of tokens used
            headers: Response headers from API (if available)
        """
        current_time = time.time()

        # If we have headers from API - use them
        if headers:
            remaining = headers.get("x-ratelimit-remaining-tokens")
            reset_str = headers.get("x-ratelimit-reset-tokens")

            if remaining is not None:
                self.remaining_tokens = int(remaining)

            if reset_str:
                # Format: "XXXms", "XXXs", or just number - convert to seconds
                if reset_str.endswith("ms"):
                    reset_ms = int(reset_str.rstrip("ms"))
                    reset_seconds = reset_ms / 1000
                elif reset_str.endswith("s"):
                    # Format like "1.625s" or "0s"
                    reset_seconds = float(reset_str.rstrip("s"))
                else:
                    # Just a number (assumed to be seconds)
                    reset_seconds = float(reset_str)
                
                # If reset_seconds is 0 or very small, set a minimum of 1 second in future
                # to maintain the invariant that reset_time is in the future
                if reset_seconds <= 0.1:
                    reset_seconds = 1.0
                    
                self.reset_time = current_time + reset_seconds
                logger.debug(
                    f"TPM state from headers: remaining={self.remaining_tokens}, "
                    f"reset_in={reset_seconds:.1f}s"
                )
                return  # Exit early when headers are available

        # Fallback: Simple token subtraction when headers not available
        logger.debug("Headers not available, using fallback TPM tracking")
        if current_time >= self.reset_time:
            # Minute has passed - restore limit
            self.remaining_tokens = self.tpm_limit
            self.reset_time = current_time + 60

        self.remaining_tokens -= tokens_used
        logger.debug(
            f"TPM state updated (fallback): remaining={self.remaining_tokens}, used={tokens_used}"
        )

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """L2-normalize a vector; return original if zero norm."""
        import math

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [float(v / norm) for v in vector]

    def _get_internal_embedding(self, text: str) -> List[float]:
        """
        Call internal embeddings endpoint and return normalized vector.

        Expects JSON response with {"Embedding": [...]}.
        """
        if not self.base_url:
            raise ValueError("embedding_base_url must be provided for internal embeddings")

        # Import requests lazily to avoid hard dependency unless needed
        try:
            import requests  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "The 'requests' package is required for internal embeddings mode"
            ) from e

        headers = {
            "authorization": f"OAuth {self.internal_oauth_token}",
            "content-type": "application/json",
        }
        payload = {
            "TextSegments": {"prompt": text}
        }

        resp = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Internal embeddings API error: HTTP {resp.status_code} - {resp.text[:200]}")

        data = resp.json()
        if not isinstance(data, dict) or "Embedding" not in data:
            raise ValueError("Invalid response from internal embeddings API: missing 'Embedding'")

        vector = data["Embedding"]
        if not isinstance(vector, list) or not vector:
            raise ValueError("Invalid 'Embedding' format from internal embeddings API")

        # Record embedding dimension on first successful call
        if self.embedding_dim is None:
            self.embedding_dim = int(len(vector))

        return self._normalize_vector([float(v) for v in vector])

    def _wait_for_tokens(self, required_tokens: int, safety_margin: float = 0.15):
        """
        Wait for token availability.

        Args:
            required_tokens: Required number of tokens
            safety_margin: Safety margin
        """
        required_with_margin = int(required_tokens * (1 + safety_margin))

        if self.remaining_tokens < required_with_margin:
            wait_time = max(0, self.reset_time - time.time())
            if wait_time > 0:
                logger.info(
                    f"Waiting {wait_time:.1f}s for TPM limit reset (need {required_with_margin} tokens)"
                )
                utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(
                    f"[{utc3_time}] EMBEDDINGS | ⏳ Waiting {wait_time:.1f}s for TPM limit reset..."
                )
                time.sleep(wait_time + 0.1)  # +0.1s for reliability
                self.remaining_tokens = self.tpm_limit
                self.reset_time = time.time() + 60
                utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(f"[{utc3_time}] EMBEDDINGS | ✅ TPM limit reset, continuing...")

    def _batch_texts(self, texts: List[str]) -> List[List[str]]:
        """
        Split texts into batches considering API limits.

        Limits:
        - Maximum 2048 texts per batch
        - Recommended no more than ~100K tokens per batch

        Returns:
            List of batches
        """
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            # Check text length
            text_tokens = self._count_tokens(text)

            # If text is too long - truncate it
            if text_tokens > self.max_tokens_per_text:
                logger.warning(
                    f"Text with {text_tokens} tokens exceeds limit, "
                    f"truncating to {self.truncate_tokens}"
                )
                utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(
                    f"[{utc3_time}] EMBEDDINGS | ⚠️ Text truncated: "
                    f"{text_tokens} → {self.truncate_tokens} tokens"
                )
                text = self._truncate_text(text, self.truncate_tokens)
                text_tokens = self.truncate_tokens

            # Check if it fits in current batch
            if len(current_batch) >= self.max_texts_per_request or (
                current_tokens + text_tokens > self.max_batch_tokens and current_batch
            ):
                # Save current batch and start new one
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        # Add last batch
        if current_batch:
            batches.append(current_batch)

        logger.info(f"Split {len(texts)} texts into {len(batches)} batches")
        return batches

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of texts to process

        Returns:
            numpy array shape (n_texts, 1536) with normalized vectors

        Raises:
            Exception: On API errors after all retry attempts
        """
        if not texts:
            return np.array([])

        # Save indices of non-empty texts
        non_empty_indices = []
        non_empty_texts = []

        for i, text in enumerate(texts):
            if text.strip():  # Non-empty text
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # If all texts are empty - return zero vectors
        if not non_empty_texts:
            logger.warning(f"All {len(texts)} texts are empty, returning zero vectors")
            return np.zeros((len(texts), 1536), dtype=np.float32)

        # Split only non-empty texts into batches
        batches = self._batch_texts(non_empty_texts)

        if len(batches) > 1:
            utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(
                f"[{utc3_time}] EMBEDDINGS | Processing {len(non_empty_texts)} texts in {len(batches)} batches..."
            )

        all_embeddings = []

        for batch_idx, batch in enumerate(batches):
            batch_tokens = sum(self._count_tokens(text) for text in batch)

            # Wait for token availability
            self._wait_for_tokens(batch_tokens)

            # Retry logic for the whole batch (external) or per-text (internal)
            if self.use_internal_auth and self.base_url:
                # Internal: process texts sequentially with per-text retries
                for text in batch:
                    text_tokens = self._count_tokens(text)
                    last_error = None
                    for attempt in range(self.max_retries):
                        try:
                            vec = self._get_internal_embedding(text)
                            all_embeddings.append(vec)
                            # Fallback TPM tracking when headers unavailable
                            self._update_tpm_state(text_tokens, headers=None)
                            break
                        except Exception as e:
                            last_error = e
                            error_str = str(e)
                            if "429" in error_str or "rate" in error_str.lower():
                                wait_time = (2**attempt) * 10
                                logger.warning(
                                    f"Internal rate limit, retry {attempt + 1}/{self.max_retries} in {wait_time}s"
                                )
                                time.sleep(wait_time)
                                # Reset TPM state windows
                                self.remaining_tokens = 0
                                self.reset_time = time.time() + wait_time
                            else:
                                wait_time = (2**attempt) * 5
                                logger.warning(
                                    f"Internal API error: {type(e).__name__}, retry {attempt + 1}/{self.max_retries} in {wait_time}s"
                                )
                                time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Failed to get internal embedding after {self.max_retries} attempts"
                        )
                        raise last_error

                if len(batches) > 1:
                    utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"[{utc3_time}] EMBEDDINGS | ✅ Batch {batch_idx + 1}/{len(batches)} completed"
                    )
            else:
                # External OpenAI: one call per batch with raw headers
                last_error = None
                for attempt in range(self.max_retries):
                    try:
                        logger.debug(
                            f"Processing batch {batch_idx + 1}/{len(batches)}, {len(batch)} texts"
                        )

                        raw_response = self.client.embeddings.with_raw_response.create(
                            input=batch, model=self.model, encoding_format="float"
                        )

                        headers = {
                            "x-ratelimit-remaining-tokens": raw_response.headers.get(
                                "x-ratelimit-remaining-tokens"
                            ),
                            "x-ratelimit-reset-tokens": raw_response.headers.get(
                                "x-ratelimit-reset-tokens"
                            ),
                        }

                        # Update TPM state with real headers
                        self._update_tpm_state(batch_tokens, headers)

                        # Get parsed response
                        response = raw_response.parse()

                        # Extract vectors
                        embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(embeddings)

                        logger.debug(f"Batch {batch_idx + 1} processed successfully")

                        if len(batches) > 1:
                            utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                            print(
                                f"[{utc3_time}] EMBEDDINGS | ✅ Batch {batch_idx + 1}/{len(batches)} completed"
                            )

                        break

                    except Exception as e:
                        last_error = e
                        error_type = type(e).__name__

                        if "rate_limit" in str(e).lower() or "429" in str(e):
                            wait_time = (2**attempt) * 10
                            logger.warning(
                                f"Rate limit hit, attempt {attempt + 1}/{self.max_retries}, waiting {wait_time}s"
                            )
                            utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                            print(
                                f"[{utc3_time}] EMBEDDINGS | ⏳ Rate limit hit, retry {attempt + 1}/{self.max_retries} in {wait_time}s..."
                            )
                            time.sleep(wait_time)
                            self.remaining_tokens = 0
                            self.reset_time = time.time() + wait_time
                        else:
                            wait_time = (2**attempt) * 5
                            logger.warning(
                                f"API error: {error_type}, attempt {attempt + 1}/{self.max_retries}, waiting {wait_time}s"
                            )
                            time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to get embeddings after {self.max_retries} attempts"
                    )
                    raise last_error

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Restore original order with zero vectors for empty texts
        if len(non_empty_indices) < len(texts):
            # Use detected dimension for internal, otherwise default 1536
            dim = self.embedding_dim or 1536
            result = np.zeros((len(texts), dim), dtype=np.float32)
            for i, idx in enumerate(non_empty_indices):
                result[idx] = embeddings_array[i]
            logger.info(
                f"Processed {len(non_empty_texts)} non-empty texts out of {len(texts)} total"
            )
            if len(batches) > 1:
                utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(
                    f"[{utc3_time}] EMBEDDINGS | ✅ Completed: {len(non_empty_texts)} texts processed"
                )
            return result

        if len(batches) > 1:
            utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{utc3_time}] EMBEDDINGS | ✅ Completed: {len(texts)} texts processed")

        return embeddings_array


def get_embeddings(texts: List[str], config: Dict[str, Any]) -> np.ndarray:
    """
    Simple wrapper for getting embeddings.

    Args:
        texts: List of texts
        config: Configuration (must contain embedding_api_key or api_key)

    Returns:
        numpy array with embeddings
    """
    client = EmbeddingsClient(config)
    return client.get_embeddings(texts)


def cosine_similarity_batch(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between two sets of vectors.

    Args:
        embeddings1: Vector array shape (n1, dim)
        embeddings2: Vector array shape (n2, dim)

    Returns:
        Cosine similarity array shape (n1, n2)

    Note:
        Vectors must be normalized (OpenAI API returns normalized vectors)
    """
    # For normalized vectors cosine similarity = dot product
    return np.dot(embeddings1, embeddings2.T)
