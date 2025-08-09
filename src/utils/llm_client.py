"""
LLM Client for working with OpenAI Responses API.

Module provides classes for working with OpenAI API with support for:
- Token limit control via response headers
- Response chains via previous_response_id
- Reasoning models (o*)
- Exponential backoff retry logic
- Error handling and model refusal processing

Usage examples:
    >>> config = {
    ...     'api_key': 'sk-...',
    ...     'model': 'gpt-4o',
    ...     'tpm_limit': 120000,
    ...     'tpm_safety_margin': 0.15,
    ...     'max_completion': 4096,
    ...     'max_retries': 6
    ... }
    >>> client = OpenAIClient(config)
    >>> response, response_id, usage = client.create_response(
    ...     "You are a helpful assistant",
    ...     "What is the capital of France?"
    ... )
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import openai
import tiktoken
from openai import OpenAI


@dataclass
class ResponseUsage:
    """
    Structure for tracking used tokens.

    Attributes:
        input_tokens (int): Number of tokens in input data
        output_tokens (int): Number of tokens in model response
        total_tokens (int): Total number of tokens (input + output)
        reasoning_tokens (int): Number of reasoning tokens (for o* models)

    Example:
        >>> usage = ResponseUsage(
        ...     input_tokens=150,
        ...     output_tokens=50,
        ...     total_tokens=200,
        ...     reasoning_tokens=30
        ... )
        >>> print(f"Total cost: {usage.total_tokens} tokens")
        >>> if usage.reasoning_tokens > 0:
        ...     print(f"Including {usage.reasoning_tokens} reasoning tokens")
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0


class TPMBucket:
    """
    Tokens per minute (TPM) limit control via response headers.

    Uses headers from OpenAI API for precise tracking of limit state.
    Headers come in format:
    - x-ratelimit-remaining-tokens: number of remaining tokens
    - x-ratelimit-reset-tokens: reset time in milliseconds (e.g., "820ms")

    Attributes:
        initial_limit (int): Initial limit from config
        remaining_tokens (int): Remaining tokens from headers
        reset_time (int): Unix timestamp of limit reset

    Example:
        >>> bucket = TPMBucket(120000)
        >>> # After request, update from headers
        >>> bucket.update_from_headers(response_headers)
        >>> # Before next request, check
        >>> bucket.wait_if_needed(5000, safety_margin=0.15)
    """

    def __init__(self, initial_limit: int):
        """
        Initialize TPM bucket.

        Args:
            initial_limit: Initial tokens per minute limit from config
        """
        self.initial_limit = initial_limit
        self.remaining_tokens = initial_limit
        self.reset_time = None
        self.logger = logging.getLogger(__name__)

    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """
        Update state from response headers.

        Processes headers in OpenAI format:
        - remaining tokens as number
        - reset time in milliseconds with "ms" suffix

        Args:
            headers: Headers from OpenAI API response or prepared dict

        Note:
            OpenAI API returns reset time in "XXXms" format (milliseconds).
            For example: "820ms" means reset in 820 milliseconds.

            This function performs the "reimburse" role in traditional bucket logic -
            state is updated based on actual data from the server.
        """
        # Update remaining tokens (consume effect)
        if "x-ratelimit-remaining-tokens" in headers:
            remaining = headers.get("x-ratelimit-remaining-tokens")
            if remaining:
                try:
                    old_remaining = self.remaining_tokens
                    self.remaining_tokens = int(remaining)

                    # Detailed logging of changes
                    tokens_consumed = old_remaining - self.remaining_tokens
                    if tokens_consumed > 0:
                        self.logger.debug(
                            f"TPM consumed: {tokens_consumed} tokens "
                            f"({old_remaining} → {self.remaining_tokens})"
                        )
                    elif tokens_consumed < 0:
                        self.logger.debug(
                            f"TPM reimbursed: {-tokens_consumed} tokens "
                            f"({old_remaining} → {self.remaining_tokens})"
                        )
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to parse remaining tokens '{remaining}': {e}")

        # Update reset time
        if "x-ratelimit-reset-tokens" in headers:
            reset_value = headers.get("x-ratelimit-reset-tokens")
            if reset_value and reset_value != "0ms":
                try:
                    # Handle different formats
                    if isinstance(reset_value, str):
                        if reset_value.endswith("ms"):
                            # Format "XXXms" - milliseconds until reset
                            reset_ms = int(reset_value.rstrip("ms"))
                            self.reset_time = int(time.time() + (reset_ms / 1000.0))

                            # Log in readable format
                            reset_datetime = datetime.fromtimestamp(self.reset_time)
                            reset_in_seconds = reset_ms / 1000.0
                            self.logger.debug(
                                f"TPM reset in {reset_in_seconds:.1f}s "
                                f"(at {reset_datetime.strftime('%H:%M:%S')})"
                            )
                        elif reset_value.endswith("s"):
                            # Format "X.XXXs" - seconds until reset
                            reset_seconds = float(reset_value.rstrip("s"))
                            self.reset_time = int(time.time() + reset_seconds)

                            # Log in readable format
                            reset_datetime = datetime.fromtimestamp(self.reset_time)
                            self.logger.debug(
                                f"TPM reset in {reset_seconds:.1f}s "
                                f"(at {reset_datetime.strftime('%H:%M:%S')})"
                            )
                        else:
                            # Assume unix timestamp
                            self.reset_time = int(float(reset_value))
                            reset_datetime = datetime.fromtimestamp(self.reset_time)
                            self.logger.debug(
                                f"TPM reset time: {reset_datetime.strftime('%H:%M:%S')}"
                            )
                    else:
                        # If already a number
                        self.reset_time = int(reset_value)
                        reset_datetime = datetime.fromtimestamp(self.reset_time)
                        self.logger.debug(f"TPM reset time: {reset_datetime.strftime('%H:%M:%S')}")

                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to parse reset time '{reset_value}': {e}")
                    self.reset_time = None

    def wait_if_needed(self, required_tokens: int, safety_margin: float = 0.15) -> None:
        """
        Check token sufficiency and wait if necessary.

        This function performs the "consume" check role in traditional bucket logic -
        checks if there are enough tokens for the request, and waits for recovery if
        necessary.

        Args:
            required_tokens: Number of tokens needed for the request
            safety_margin: Safety margin (default 15%)

        Note:
            After waiting for limit reset, remaining_tokens is set to
            initial_limit, assuming full bucket recovery.
        """
        required_with_margin = int(required_tokens * (1 + safety_margin))

        self.logger.debug(
            f"TPM check: need {required_with_margin} tokens "
            f"(base: {required_tokens} + margin: {int(required_tokens * safety_margin)}), "
            f"have {self.remaining_tokens}"
        )

        if self.remaining_tokens >= required_with_margin:
            self.logger.debug("TPM check passed: sufficient tokens available")
            return

        # Insufficient tokens
        if self.reset_time:
            current_time = time.time()
            wait_time = self.reset_time - current_time

            if wait_time > 0:
                # Format wait time for readability
                if wait_time < 60:
                    wait_str = f"{wait_time:.1f}s"
                else:
                    minutes = int(wait_time // 60)
                    seconds = int(wait_time % 60)
                    wait_str = f"{minutes}m {seconds}s"

                self.logger.info(
                    f"TPM limit reached: {self.remaining_tokens}/{required_with_margin} tokens. "
                    f"Waiting {wait_str} until reset..."
                )

                # Wait with small buffer for reliability
                time.sleep(wait_time + 0.1)

                # After waiting, restore initial limit
                old_remaining = self.remaining_tokens
                self.remaining_tokens = self.initial_limit
                self.logger.info(
                    f"TPM limit reset: tokens restored "
                    f"({old_remaining} → {self.remaining_tokens})"
                )
                # Add console output
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time}] INFO     | ✅ TPM limit reset, continuing...")
            else:
                # Reset time already passed, can assume limit is restored
                self.logger.debug("TPM reset time already passed, assuming limit restored")
                self.remaining_tokens = self.initial_limit
        else:
            # No reset time information available
            self.logger.warning(
                f"TPM limit low ({self.remaining_tokens}/{required_with_margin}), "
                f"but no reset time available. Proceeding anyway..."
            )


class IncompleteResponseError(Exception):
    """Raised when response generation is incomplete (hit token limit)."""

    pass


class OpenAIClient:
    """
    Client for working with OpenAI Responses API.

    Provides:
    - Automatic previous_response_id management for chains
    - Support for reasoning models with special parameters
    - Exponential backoff retry on errors
    - Limit control via response headers (x-ratelimit-*)

    Attributes:
        config (dict): Client configuration
        client (OpenAI): OpenAI client instance
        tpm_bucket (TPMBucket): Token limit controller
        last_response_id (str): ID of last successful response
        is_reasoning_model (bool): True for o* models
        encoder: Tokenizer for precise counting

    Example:
        >>> config = {
        ...     'api_key': 'sk-...',
        ...     'model': 'gpt-4o',
        ...     'tpm_limit': 120000,
        ...     'tpm_safety_margin': 0.15,
        ...     'max_completion': 4096,
        ...     'timeout': 45,
        ...     'max_retries': 6,
        ...     'temperature': 1.0
        ... }
        >>> client = OpenAIClient(config)
        >>>
        >>> # First request
        >>> text1, id1, usage1 = client.create_response(
        ...     "You are a helpful assistant",
        ...     "Hello!"
        ... )
        >>>
        >>> # Second request with context from first
        >>> text2, id2, usage2 = client.create_response(
        ...     "Continue being helpful",
        ...     "What's the weather?"
        ... )
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI client.

        Args:
            config: Configuration dictionary with following keys:
                - api_key (str): OpenAI API key
                - model (str): Model name (gpt-4o, o1-preview, etc.)
                - tpm_limit (int): Tokens per minute limit
                - tpm_safety_margin (float): Safety margin (default 0.15)
                - max_completion (int): Maximum tokens for generation
                - timeout (int): Request timeout in seconds
                - max_retries (int): Number of retry attempts
                - temperature (float, optional): Temperature for regular models
                - reasoning_effort (str, optional): Effort level for reasoning models
                - reasoning_summary (str, optional): Summary type for reasoning
                - response_chain_depth (int, optional): Response chain depth management
                - truncation (str, optional): Truncation strategy ("auto", "disabled")
        """
        self.config = config
        self.client = OpenAI(api_key=config["api_key"], timeout=config.get("timeout", 60.0))
        self.tpm_bucket = TPMBucket(config["tpm_limit"])
        self.logger = logging.getLogger(__name__)

        # Last response_id for chain of responses
        self.last_response_id: Optional[str] = None

        # Last usage info for context accumulation
        self.last_usage: Optional[ResponseUsage] = None

        # ИЗМЕНЕНИЕ: Используем явный параметр is_reasoning
        if "is_reasoning" not in config:
            raise ValueError("Parameter 'is_reasoning' is required in config")

        self.is_reasoning_model = config["is_reasoning"]

        # Response chain management
        self.response_chain_depth = config.get("response_chain_depth")
        self.truncation = config.get("truncation")

        # Initialize response chain only if depth > 0
        if self.response_chain_depth is not None and self.response_chain_depth > 0:
            self.response_chain = deque()
            chain_mode = f"sliding window (depth={self.response_chain_depth})"
        elif self.response_chain_depth == 0:
            self.response_chain = None
            chain_mode = "independent requests"
        else:
            self.response_chain = None
            chain_mode = "unlimited chain"

        self.logger.info(
            f"Initialized OpenAI client: model={config['model']}, "
            f"reasoning={self.is_reasoning_model}, chain_mode={chain_mode}"
        )

        # Initialize tokenizer for precise counting
        self.encoder = tiktoken.get_encoding("o200k_base")

    def _delete_response(self, response_id: str) -> None:
        """
        Delete response via OpenAI API.

        Args:
            response_id: ID of response to delete

        Raises:
            ValueError: If deletion fails
        """
        try:
            result = self.client.responses.delete(response_id)
            if hasattr(result, "deleted") and result.deleted:
                self.logger.debug(f"Successfully deleted response {response_id[:12]}")
            else:
                raise ValueError(f"Delete returned unexpected result: {result}")
        except Exception as e:
            self.logger.warning(f"Failed to delete response {response_id[:12]}: {e}")
            raise ValueError(f"Failed to delete response: {e}") from e

    def _prepare_request_params(
        self,
        instructions: str,
        input_data: str,
        previous_response_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare request parameters for Responses API.

        Args:
            instructions: System prompt (instructions for the model)
            input_data: User input
            previous_response_id: ID of previous response for context

        Returns:
            dict: Parameters for responses.create()
        """
        params = {
            "model": self.config["model"],
            "instructions": instructions,
            "input": input_data,
            "max_output_tokens": self.config["max_completion"],
            "store": True,  # required for Responses API
        }

        # Add previous_response_id if available
        if previous_response_id:
            params["previous_response_id"] = previous_response_id

        # ИЗМЕНЕНИЕ: Добавляем только не-null параметры

        # Temperature (для любых моделей, если не null)
        if self.config.get("temperature") is not None:
            params["temperature"] = self.config["temperature"]

        # Reasoning параметры (если не null)
        reasoning_effort = self.config.get("reasoning_effort")
        reasoning_summary = self.config.get("reasoning_summary")
        if reasoning_effort is not None or reasoning_summary is not None:
            params["reasoning"] = {}
            if reasoning_effort is not None:
                params["reasoning"]["effort"] = reasoning_effort
            if reasoning_summary is not None:
                params["reasoning"]["summary"] = reasoning_summary

        # verbosity - параметр верхнего уровня (как temperature, model и т.д.)
        if self.config.get("verbosity") is not None:
            params["verbosity"] = self.config["verbosity"]

        # truncation - управление обрезкой контекста
        if self.truncation is not None:
            params["truncation"] = self.truncation

        return params

    def _clean_json_response(self, response_text: str) -> str:
        """
        Clean response from markdown wrappers and other artifacts.

        Automatically removes:
        - ```json...``` wrappers
        - ```...``` wrappers without language
        - Extra spaces at beginning/end

        Args:
            response_text: Raw response text from model

        Returns:
            str: Cleaned text
        """
        text = response_text.strip()

        # Remove ```json...``` wrappers
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()  # Remove ```json at start and ``` at end
            self.logger.debug("Removed ```json wrapper from response")

        # Remove regular ``` wrappers
        elif text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()  # Remove ``` at start and end
            self.logger.debug("Removed ``` wrapper from response")

        return text

    def _extract_response_content(self, response) -> str:
        """
        Extract response text from Responses API.

        For reasoning models expected structure:
        - output[0]: reasoning
        - output[1]: message

        For regular models:
        - output[0]: message
        """
        try:
            # Check response status
            if hasattr(response, "status"):
                # Allow both completed and incomplete statuses
                if response.status not in ["completed", "incomplete"]:
                    raise ValueError(f"Response has unexpected status: {response.status}")

                # Log warning for incomplete
                if response.status == "incomplete":
                    reason = None
                    if hasattr(response, "incomplete_details") and response.incomplete_details:
                        reason = getattr(response.incomplete_details, "reason", "unknown")
                    self.logger.warning(
                        f"Response is incomplete (reason: {reason}). "
                        f"This may happen with low max_output_tokens limits."
                    )

            # Check for output presence
            if not response.output or len(response.output) == 0:
                raise ValueError("Response has empty output")

            # Determine message index based on model type
            if self.is_reasoning_model:
                # Reasoning models should have at least 2 elements
                # output[0] - reasoning, output[1] - message
                if len(response.output) < 2:
                    raise ValueError(
                        f"Reasoning model returned insufficient output items: {len(response.output)}"
                    )
                message_output = response.output[1]
            else:
                # Regular models have message in first element
                message_output = response.output[0]

            # Check output element type
            if not hasattr(message_output, "type") or message_output.type != "message":
                raise ValueError(
                    f"Expected message output, got type: {getattr(message_output, 'type', 'unknown')}"
                )

            # Extract text from content
            if not hasattr(message_output, "content") or not message_output.content:
                raise ValueError("Message has no content")

            # content is an array, take first element with type output_text
            text_content = None
            refusal_content = None

            for content_item in message_output.content:
                if hasattr(content_item, "type"):
                    if content_item.type == "output_text":
                        if hasattr(content_item, "text"):
                            text_content = content_item.text
                            break
                    elif content_item.type == "refusal":
                        if hasattr(content_item, "refusal"):
                            refusal_content = content_item.refusal

            # Check refusal first
            if refusal_content is not None:
                raise ValueError(f"Model refused to respond: {refusal_content}")

            if text_content is None:
                raise ValueError("No text content found in message")

            # Clean from possible markdown wrappers
            cleaned_text = self._clean_json_response(text_content)

            return cleaned_text

        except Exception as e:
            self.logger.error(f"Failed to extract response content: {e}")
            self.logger.debug(f"Response structure: {response}")
            raise

    def _extract_usage_info(self, response) -> ResponseUsage:
        """
        Extract information about used tokens.

        Args:
            response: Response object from OpenAI API

        Returns:
            ResponseUsage: Structure with token information
        """
        try:
            usage = response.usage
            reasoning_tokens = 0

            # Extract reasoning tokens if available
            if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
                reasoning_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", 0)

            return ResponseUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                reasoning_tokens=reasoning_tokens,
            )
        except AttributeError as e:
            self.logger.warning(f"Could not extract usage info: {e}")
            return ResponseUsage(0, 0, 0, 0)

    def _update_tpm_via_probe(self) -> None:
        """
        Get current rate limit data via probe request.

        Executes minimal synchronous request to cheapest model
        to get response headers with limit information.

        Note:
            In background mode (async) OpenAI doesn't return rate limit headers,
            so we use periodic probe requests for updates.
        """
        try:
            self.logger.debug("Executing TPM probe request")

            # Use with_raw_response to access headers
            raw = self.client.responses.with_raw_response.create(
                model="gpt-4.1-nano-2025-04-14",
                input="2+2=?",  # Simple math question
                max_output_tokens=20,  # Minimum 16, set 20 with buffer
                temperature=0.1,  # Minimal temperature for determinism
                background=False,  # IMPORTANT: synchronous mode!
            )

            # Extract headers
            headers = dict(raw.headers)

            # Update TPM bucket
            self.tpm_bucket.update_from_headers(headers)

            self.logger.debug(
                f"TPM probe successful: remaining={self.tpm_bucket.remaining_tokens}, "
                f"reset_time={self.tpm_bucket.reset_time}"
            )

        except Exception as e:
            self.logger.warning(f"TPM probe failed: {e}")
            # Not critical - continue with current data

    def _create_response_async(
        self,
        instructions: str,
        input_data: str,
        previous_response_id: Optional[str] = None,
        is_repair: bool = False,
    ) -> Tuple[str, str, ResponseUsage]:
        """
        Create response in asynchronous mode with polling.

        Uses background=True for immediate response_id retrieval,
        then performs polling to track generation status.

        Args:
            instructions: System prompt (instructions for model)
            input_data: User data (text or JSON)
            previous_response_id: ID of previous response for context
            is_repair: If True, response_id won't be added to chain

        Returns:
            tuple: (response_text, response_id, usage_info)

        Raises:
            TimeoutError: When overall timeout is exceeded
            IncompleteResponseError: When status is incomplete
            ValueError: When status is failed or model refuses
        """
        # Handle response_chain_depth == 0 (independent requests)
        if self.response_chain_depth == 0:
            previous_response_id = None
            self.logger.debug("Using independent request mode (response_chain_depth=0)")
        # Prepare parameters as usual
        params = self._prepare_request_params(instructions, input_data, previous_response_id)

        # IMPORTANT: enable background mode
        params["background"] = True

        # Get context limit from config (with default fallback)
        max_context_tokens = self.config.get("max_context_tokens", 128000)

        # Correct token estimation with accumulated context
        if self.last_usage and previous_response_id:
            # Use input_tokens from last request as base (includes all context)
            base_context_tokens = self.last_usage.input_tokens

            # Add only new content
            new_content_tokens = len(self.encoder.encode(input_data))

            # Estimate new context size
            estimated_input_tokens = base_context_tokens + new_content_tokens

            # Check context limit - OpenAI will truncate to this limit
            if estimated_input_tokens > max_context_tokens:
                self.logger.warning(
                    f"Context approaching limit: {estimated_input_tokens} tokens "
                    f"(max: {max_context_tokens}). Old content may be truncated by OpenAI."
                )
                # Cap at max for accurate TPM calculation
                estimated_input_tokens = max_context_tokens
        else:
            # First request or no previous_response_id
            full_prompt = instructions + "\n\n" + input_data
            estimated_input_tokens = len(self.encoder.encode(full_prompt))

        # Now correctly calculate required tokens
        required_tokens = estimated_input_tokens + self.config["max_completion"]

        # Add debug logging
        self.logger.debug(
            f"Token estimation: estimated_input={estimated_input_tokens}, "
            f"with_output={required_tokens}, previous_id={previous_response_id is not None}, "
            f"max_context={max_context_tokens}"
        )

        # Update TPM data via probe before main request
        # In async mode OpenAI doesn't return rate limit headers,
        # so we use probe to get current information
        self._update_tpm_via_probe()

        # Check token sufficiency with safety margin
        safety_margin = self.config.get("tpm_safety_margin", 0.15)
        self.tpm_bucket.wait_if_needed(required_tokens, safety_margin)

        retry_count = 0
        last_exception = None

        while retry_count <= self.config["max_retries"]:
            try:
                self.logger.debug(
                    f"Async API request attempt {retry_count + 1}: model={params['model']}, "
                    f"estimated_tokens={required_tokens}, previous_response_id={params.get('previous_response_id')}"
                )

                # Step 1: Create background request
                raw_initial = self.client.responses.with_raw_response.create(**params)
                # initial_headers = raw_initial.headers  # Currently unused
                initial_response = raw_initial.parse()
                response_id = initial_response.id

                self.logger.info(f"Background response created: {response_id[:8]}...")

                # Update TPM from initial headers
                # COMMENTED: in async mode headers don't contain rate limit info
                # if initial_headers:
                #    self.tpm_bucket.update_from_headers(initial_headers)

                # Step 2: Polling loop
                start_time = time.time()
                poll_count = 0
                poll_interval = self.config.get("poll_interval", 5)

                while True:
                    elapsed = time.time() - start_time

                    # Check hard timeout
                    if elapsed > self.config["timeout"]:
                        self.logger.warning(
                            f"Response generation exceeded timeout ({self.config['timeout']}s), cancelling..."
                        )
                        try:
                            self.client.responses.cancel(response_id)
                            self.logger.info(f"Successfully cancelled response {response_id[:12]}")
                        except Exception as e:
                            # Race condition: response may already be completed/failed
                            self.logger.debug(f"Could not cancel {response_id[:12]}: {e}")

                        raise TimeoutError(
                            f"Response generation exceeded {self.config['timeout']}s timeout"
                        )

                    # Get current status
                    try:
                        raw_status = self.client.responses.with_raw_response.retrieve(response_id)
                        # status_headers = raw_status.headers  # Currently unused
                        response = raw_status.parse()
                    except Exception as e:
                        self.logger.error(f"Failed to retrieve response status: {e}")
                        raise

                    # Update TPM from status request headers
                    # COMMENTED: in async mode headers don't contain rate limit info
                    # if status_headers:
                    #     self.tpm_bucket.update_from_headers(status_headers)

                    # Handle statuses
                    status = response.status

                    if status == "completed":

                        # Successful completion
                        self.logger.info(f"Response {response_id[:12]} completed successfully")

                        # Extract result
                        response_text = self._extract_response_content(response)
                        usage_info = self._extract_usage_info(response)

                        # Save response_id for next call (unless it's repair)
                        if not is_repair:
                            self.last_response_id = response_id

                            # Manage response chain if configured
                            if self.response_chain is not None and self.response_chain_depth > 0:
                                # Add new response to chain
                                self.response_chain.append(response_id)

                                # Remove old responses if chain exceeds depth
                                while len(self.response_chain) > self.response_chain_depth:
                                    old_response_id = self.response_chain.popleft()
                                    try:
                                        self._delete_response(old_response_id)
                                        self.logger.info(
                                            f"Deleted old response {old_response_id[:12]} from chain "
                                            f"(chain size: {len(self.response_chain)})"
                                        )
                                    except ValueError as e:
                                        # Deletion failed - log warning and continue
                                        self.logger.warning(
                                            f"Failed to delete old response {old_response_id[:12]}: {e}"
                                        )
                                        # Continue anyway - chain is already shortened locally
                        else:
                            self.logger.debug("Repair response - not adding to chain")

                        self.logger.debug(
                            f"Async response success: id={response_id}, "
                            f"tokens={usage_info.total_tokens} "
                            f"(reasoning: {usage_info.reasoning_tokens})"
                        )

                        # Save usage for next request
                        self.last_usage = usage_info

                        return response_text, response_id, usage_info

                    elif status == "incomplete":
                        # Critical error - response truncated
                        reason = "unknown"
                        if hasattr(response, "incomplete_details") and response.incomplete_details:
                            reason = getattr(response.incomplete_details, "reason", "unknown")

                        # Try to get partial result for logging
                        partial_tokens = 0
                        if hasattr(response, "usage") and response.usage:
                            partial_tokens = response.usage.output_tokens

                        self.logger.error(
                            f"Response {response_id[:12]} incomplete: {reason}. "
                            f"Generated {partial_tokens} tokens before hitting limit."
                        )

                        # Output to terminal
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(f"[{current_time}] ERROR    | ❌ Response incomplete: {reason}")
                        print(
                            f"[{current_time}] ERROR    |    Generated only {partial_tokens} tokens"
                        )

                        # For reasoning models this is especially critical
                        if self.is_reasoning_model and reason == "max_output_tokens":
                            current_time = datetime.now().strftime("%H:%M:%S")
                            print(
                                f"[{current_time}] HINT     | 💡 Reasoning model needs more tokens. "
                                f"Current limit: {self.config['max_completion']}"
                            )
                            print(
                                f"[{current_time}] HINT     | 💡 Consider increasing max_completion or enabling truncation='auto'"
                            )

                        # NO RETRY for incomplete responses - raise immediately
                        raise IncompleteResponseError(
                            f"Response generation incomplete ({reason}). "
                            f"Generated only {partial_tokens} tokens. "
                            f"Context size: {estimated_input_tokens} tokens."
                        )

                    elif status == "failed":
                        # Generation error
                        error_msg = "Unknown error"
                        if hasattr(response, "error") and response.error:
                            error_msg = getattr(response.error, "message", str(response.error))

                        self.logger.error(f"Response {response_id[:12]} failed: {error_msg}")

                        raise ValueError(f"Response generation failed: {error_msg}")

                    elif status == "cancelled":
                        # Should not happen unless we cancelled ourselves
                        self.logger.error(f"Response {response_id[:12]} was cancelled unexpectedly")
                        raise ValueError("Response was cancelled")

                    elif status == "queued":
                        # Show initial status only once
                        if poll_count == 0:
                            current_time = datetime.now().strftime("%H:%M:%S")
                            print(
                                f"[{current_time}] QUEUE    | ⏳ Response {response_id[:12]}... in progress"
                            )

                        # Show progress every 3 checks (~21 sec with poll_interval=7)
                        elif poll_count > 0 and poll_count % 3 == 0:
                            elapsed_time = int(time.time() - start_time)
                            current_time = datetime.now().strftime("%H:%M:%S")

                            # Format time
                            if elapsed_time < 60:
                                time_str = f"{elapsed_time}s"
                            else:
                                minutes = elapsed_time // 60
                                seconds = elapsed_time % 60
                                time_str = f"{minutes}m {seconds}s"

                            print(f"[{current_time}] PROGRESS | ⏳ Elapsed: {time_str}")

                        # Adaptive polling interval
                        if poll_count < 3:
                            time.sleep(2)  # First 3 times check quickly
                        else:
                            time.sleep(poll_interval)  # Then use configured interval

                        poll_count += 1
                        continue

                    else:
                        # Unknown status
                        self.logger.error(f"Unknown response status: {status}")
                        raise ValueError(f"Unknown response status: {status}")

            except IncompleteResponseError as e:
                # NO RETRY for incomplete responses - raise immediately
                self.logger.error(f"Response incomplete, not retrying: {e}")
                raise e

            except openai.RateLimitError as e:
                # Rate limit errors can be retried
                retry_count += 1
                if retry_count > self.config["max_retries"]:
                    self.logger.error(f"Error after all retries: {type(e).__name__}: {e}")
                    raise e

                wait_time = 20 * (2 ** (retry_count - 1))
                self.logger.warning(
                    f"{type(e).__name__}, retry {retry_count}/{self.config['max_retries']} in {wait_time}s"
                )
                current_time = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{current_time}] RETRY    | ⏳ Waiting {wait_time}s before retry {retry_count}/{self.config['max_retries']}..."
                )
                time.sleep(wait_time)
                last_exception = e

            except TimeoutError as e:
                # Timeout not retryable - throw immediately
                raise e

            except Exception as e:
                # Other errors can also be retried
                retry_count += 1
                if retry_count > self.config["max_retries"]:
                    self.logger.error(f"Error after all retries: {type(e).__name__}: {e}")
                    raise e

                wait_time = 20 * (2 ** (retry_count - 1))
                self.logger.warning(
                    f"{type(e).__name__}, retry {retry_count}/{self.config['max_retries']} in {wait_time}s: {e}"
                )
                time.sleep(wait_time)
                last_exception = e

        # If we got here - all retries exhausted
        raise last_exception or Exception("Max retries exceeded")

    def create_response(
        self,
        instructions: str,
        input_data: str,
        previous_response_id: Optional[str] = None,
        is_repair: bool = False,
    ) -> Tuple[str, str, ResponseUsage]:
        """
        Create response via OpenAI Responses API.

        Automatically:
        - Manages previous_response_id for context preservation
        - Checks limits via headers with safety margin
        - Performs retry with exponential backoff on errors
        - Handles reasoning model specifics
        - Updates TPM bucket from response headers
        - Manages response chain with configurable depth

        Args:
            instructions: System prompt (instructions for model)
            input_data: User data (text or JSON)
            previous_response_id: ID of previous response (if not specified,
                uses saved last_response_id)
            is_repair: If True, response won't be added to chain (for repair requests)

        Returns:
            tuple: (response_text, response_id, usage_info)
                - response_text (str): Model response text
                - response_id (str): Response ID for use in chain
                - usage_info (ResponseUsage): Information about used tokens

        Raises:
            openai.RateLimitError: When rate limit exceeded after all retries
            openai.APIError: On API errors after all retries
            ValueError: On incorrect response or model refusal
            IncompleteResponseError: When response is incomplete (no retry)

        Example:
            >>> # Simple request
            >>> text, resp_id, usage = client.create_response(
            ...     "You are a helpful assistant",
            ...     "What is 2+2?"
            ... )
            >>> print(f"Response: {text}")
            >>> print(f"Used {usage.total_tokens} tokens")
            >>>
            >>> # Chain of requests with context
            >>> text2, resp_id2, usage2 = client.create_response(
            ...     "Continue the conversation",
            ...     "And what is 3+3?",
            ...     previous_response_id=resp_id
            ... )
        """
        # Use passed previous_response_id or saved one
        if previous_response_id is None:
            previous_response_id = self.last_response_id

        # Call new asynchronous method with is_repair flag
        return self._create_response_async(
            instructions, input_data, previous_response_id, is_repair
        )

    def repair_response(
        self, instructions: str, input_data: str, previous_response_id: Optional[str] = None
    ) -> Tuple[str, str, ResponseUsage]:
        """
        Repair request with specified previous_response_id.

        Pure transport layer method that delegates to create_response with is_repair=True.
        The caller is responsible for any special instructions needed
        for repair (e.g., JSON formatting requirements).

        IMPORTANT: Repair responses are NOT added to the response chain.

        Args:
            instructions: System prompt (caller should include repair instructions)
            input_data: User data
            previous_response_id: ID of previous response to use as context.
                If None, uses self.last_response_id for backward compatibility.

        Returns:
            tuple: (response_text, response_id, usage_info)
                See create_response() for details

        Example:
            >>> # Rollback to last successful response
            >>> repair_text, repair_id, usage = client.repair_response(
            ...     instructions="Return valid JSON\n" + original_prompt,
            ...     input_data=data,
            ...     previous_response_id=last_successful_id
            ... )
        """
        # Use provided previous_response_id or fall back to last_response_id
        if previous_response_id is not None:
            response_id_to_use = previous_response_id
        else:
            response_id_to_use = self.last_response_id

        # Delegate to create_response with is_repair=True
        return self.create_response(instructions, input_data, response_id_to_use, is_repair=True)
