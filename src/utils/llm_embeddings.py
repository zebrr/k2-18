"""
Module for working with OpenAI Embeddings API.

Provides functions for obtaining vector representations of text
and calculating cosine similarity between vectors.
"""

import logging
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
        # API key - try embedding_api_key first, then api_key
        api_key = config.get("embedding_api_key") or config.get("api_key")
        if not api_key:
            raise ValueError("API key not found in config (embedding_api_key or api_key)")

        self.client = OpenAI(api_key=api_key)
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

            # Retry logic
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    logger.debug(
                        f"Processing batch {batch_idx + 1}/{len(batches)}, {len(batch)} texts"
                    )

                    # API call with raw response to get headers
                    raw_response = self.client.embeddings.with_raw_response.create(
                        input=batch, model=self.model, encoding_format="float"
                    )

                    # Extract headers for TPM tracking
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
                        # Rate limit - use exponential backoff
                        wait_time = (2**attempt) * 10
                        logger.warning(
                            f"Rate limit hit, attempt {attempt + 1}/{self.max_retries}, waiting {wait_time}s"
                        )
                        utc3_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
                        print(
                            f"[{utc3_time}] EMBEDDINGS | ⏳ Rate limit hit, retry {attempt + 1}/{self.max_retries} in {wait_time}s..."
                        )
                        time.sleep(wait_time)

                        # Reset TPM state
                        self.remaining_tokens = 0
                        self.reset_time = time.time() + wait_time
                    else:
                        # Other errors - also retry with backoff
                        wait_time = (2**attempt) * 5
                        logger.warning(
                            f"API error: {error_type}, attempt {attempt + 1}/{self.max_retries}, waiting {wait_time}s"
                        )
                        time.sleep(wait_time)
            else:
                # All attempts exhausted
                logger.error(f"Failed to get embeddings after {self.max_retries} attempts")
                raise last_error

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Restore original order with zero vectors for empty texts
        if len(non_empty_indices) < len(texts):
            result = np.zeros((len(texts), 1536), dtype=np.float32)
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
