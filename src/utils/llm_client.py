"""
LLM Client для работы с OpenAI Responses API.

Модуль предоставляет классы для работы с OpenAI API с поддержкой:
- Контроля лимитов токенов через response headers
- Цепочек ответов через previous_response_id
- Reasoning моделей (o*)
- Exponential backoff retry логики
- Обработки ошибок и отказов модели

Примеры использования:
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

import time
import logging
import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import openai
from openai import OpenAI
import tiktoken


@dataclass
class ResponseUsage:
    """
    Структура для учета использованных токенов.
    
    Attributes:
        input_tokens (int): Количество токенов во входных данных
        output_tokens (int): Количество токенов в ответе модели
        total_tokens (int): Общее количество токенов (input + output)
        reasoning_tokens (int): Количество токенов на reasoning (для o* моделей)
        
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
    Контроль лимитов токенов в минуту (TPM) через response headers.
    
    Использует headers от OpenAI API для точного отслеживания состояния
    лимитов. Headers приходят в формате:
    - x-ratelimit-remaining-tokens: число оставшихся токенов
    - x-ratelimit-reset-tokens: время сброса в миллисекундах (например, "820ms")
    
    Attributes:
        initial_limit (int): Начальный лимит из конфига
        remaining_tokens (int): Оставшиеся токены из headers
        reset_time (int): Unix timestamp сброса лимита
        
    Example:
        >>> bucket = TPMBucket(120000)
        >>> # После запроса обновляем из headers
        >>> bucket.update_from_headers(response_headers)
        >>> # Перед следующим запросом проверяем
        >>> bucket.wait_if_needed(5000, safety_margin=0.15)
    """
    
    def __init__(self, initial_limit: int):
        """
        Инициализация TPM bucket.
        
        Args:
            initial_limit: Начальный лимит токенов в минуту из конфига
        """
        self.initial_limit = initial_limit
        self.remaining_tokens = initial_limit
        self.reset_time = None
        self.logger = logging.getLogger(__name__)
        
    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """
        Обновление состояния из response headers.
        
        Обрабатывает headers в формате OpenAI:
        - remaining токены как число
        - reset время в миллисекундах с суффиксом "ms"
        
        Args:
            headers: Headers от OpenAI API response или подготовленный dict
            
        Note:
            OpenAI API возвращает время сброса в формате "XXXms" (миллисекунды).
            Например: "820ms" означает сброс через 820 миллисекунд.
            
            Эта функция выполняет роль "reimburse" в традиционной bucket логике -
            состояние обновляется на основе актуальных данных от сервера.
        """
        # Обновляем remaining tokens (consume эффект)
        if 'x-ratelimit-remaining-tokens' in headers:
            remaining = headers.get('x-ratelimit-remaining-tokens')
            if remaining:
                try:
                    old_remaining = self.remaining_tokens
                    self.remaining_tokens = int(remaining)
                    
                    # Детальное логирование изменений
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
            
        # Обновляем reset time
        if 'x-ratelimit-reset-tokens' in headers:
            reset_value = headers.get('x-ratelimit-reset-tokens')
            if reset_value and reset_value != '0ms':
                try:
                    # Обработка различных форматов
                    if isinstance(reset_value, str):
                        if reset_value.endswith('ms'):
                            # Формат "XXXms" - миллисекунды до сброса
                            reset_ms = int(reset_value.rstrip('ms'))
                            self.reset_time = int(time.time() + (reset_ms / 1000.0))
                            
                            # Логируем в читаемом формате
                            reset_datetime = datetime.fromtimestamp(self.reset_time)
                            reset_in_seconds = reset_ms / 1000.0
                            self.logger.debug(
                                f"TPM reset in {reset_in_seconds:.1f}s "
                                f"(at {reset_datetime.strftime('%H:%M:%S')})"
                            )
                        elif reset_value.endswith('s'):
                            # Формат "X.XXXs" - секунды до сброса
                            reset_seconds = float(reset_value.rstrip('s'))
                            self.reset_time = int(time.time() + reset_seconds)
                            
                            # Логируем в читаемом формате
                            reset_datetime = datetime.fromtimestamp(self.reset_time)
                            self.logger.debug(
                                f"TPM reset in {reset_seconds:.1f}s "
                                f"(at {reset_datetime.strftime('%H:%M:%S')})"
                            )
                        else:
                            # Предполагаем unix timestamp
                            self.reset_time = int(float(reset_value))
                            reset_datetime = datetime.fromtimestamp(self.reset_time)
                            self.logger.debug(f"TPM reset time: {reset_datetime.strftime('%H:%M:%S')}")
                    else:
                        # Если уже число
                        self.reset_time = int(reset_value)
                        reset_datetime = datetime.fromtimestamp(self.reset_time)
                        self.logger.debug(f"TPM reset time: {reset_datetime.strftime('%H:%M:%S')}")
                        
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to parse reset time '{reset_value}': {e}")
                    self.reset_time = None
    
    def wait_if_needed(self, required_tokens: int, safety_margin: float = 0.15) -> None:
        """
        Проверка достаточности токенов и ожидание при необходимости.
        
        Эта функция выполняет роль "consume" проверки в традиционной bucket логике -
        проверяет, достаточно ли токенов для запроса, и ждёт восстановления при
        необходимости.
        
        Args:
            required_tokens: Количество токенов, необходимое для запроса
            safety_margin: Запас безопасности (по умолчанию 15%)
            
        Note:
            После ожидания сброса лимита, remaining_tokens устанавливается в
            initial_limit, предполагая полное восстановление bucket.
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
            
        # Недостаточно токенов
        if self.reset_time:
            current_time = time.time()
            wait_time = self.reset_time - current_time
            
            if wait_time > 0:
                # Форматируем время ожидания для читаемости
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
                
                # Ждём с небольшим запасом для надёжности
                time.sleep(wait_time + 0.1)
                
                # После ожидания восстанавливаем начальный лимит
                old_remaining = self.remaining_tokens
                self.remaining_tokens = self.initial_limit
                self.logger.info(
                    f"TPM limit reset: tokens restored "
                    f"({old_remaining} → {self.remaining_tokens})"
                )
                # Добавляем вывод в консоль
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time}] INFO     | ✅ TPM limit reset, continuing...")
            else:
                # Reset time уже прошло, можно считать что лимит восстановлен
                self.logger.debug("TPM reset time already passed, assuming limit restored")
                self.remaining_tokens = self.initial_limit
        else:
            # Нет информации о reset time
            self.logger.warning(
                f"TPM limit low ({self.remaining_tokens}/{required_with_margin}), "
                f"but no reset time available. Proceeding anyway..."
            )


class IncompleteResponseError(Exception):
    """Raised when response generation is incomplete (hit token limit)."""
    pass


class OpenAIClient:
    """
    Клиент для работы с OpenAI Responses API.
    
    Обеспечивает:
    - Автоматическое управление previous_response_id для цепочек
    - Поддержку reasoning моделей с специальными параметрами
    - Exponential backoff retry при ошибках
    - Контроль лимитов через response headers (x-ratelimit-*)
    
    Attributes:
        config (dict): Конфигурация клиента
        client (OpenAI): Экземпляр OpenAI клиента
        tpm_bucket (TPMBucket): Контроллер лимитов токенов
        last_response_id (str): ID последнего успешного ответа
        is_reasoning_model (bool): True для o* моделей
        encoder: Токенайзер для точного подсчёта
        
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
        >>> # Первый запрос
        >>> text1, id1, usage1 = client.create_response(
        ...     "You are a helpful assistant",
        ...     "Hello!"
        ... )
        >>> 
        >>> # Второй запрос с контекстом первого
        >>> text2, id2, usage2 = client.create_response(
        ...     "Continue being helpful",
        ...     "What's the weather?"
        ... )
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация OpenAI клиента.
        
        Args:
            config: Словарь конфигурации со следующими ключами:
                - api_key (str): Ключ OpenAI API
                - model (str): Название модели (gpt-4o, o1-preview и т.д.)
                - tpm_limit (int): Лимит токенов в минуту
                - tpm_safety_margin (float): Запас безопасности (default 0.15)
                - max_completion (int): Максимум токенов для генерации
                - timeout (int): Таймаут запроса в секундах
                - max_retries (int): Количество повторных попыток
                - temperature (float, optional): Температура для обычных моделей
                - reasoning_effort (str, optional): Уровень для reasoning моделей
                - reasoning_summary (str, optional): Тип summary для reasoning
        """
        self.config = config
        self.client = OpenAI(
            api_key=config['api_key'],
            timeout=config.get('timeout', 60.0)
        )
        self.tpm_bucket = TPMBucket(config['tpm_limit'])
        self.logger = logging.getLogger(__name__)
        
        # Последний response_id для chain of responses
        self.last_response_id: Optional[str] = None
        
        # Проверяем является ли модель reasoning (o*)
        self.is_reasoning_model = config['model'].startswith('o')
        self.logger.info(f"Initialized OpenAI client: model={config['model']}, reasoning={self.is_reasoning_model}")
        
        # Инициализация токенайзера для точного подсчёта
        self.encoder = tiktoken.get_encoding("o200k_base")
    
    def _prepare_request_params(self, instructions: str, input_data: str, 
                              previous_response_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Подготовка параметров запроса для Responses API.
        
        Args:
            instructions: Системный промпт (инструкции для модели)
            input_data: Пользовательский ввод
            previous_response_id: ID предыдущего ответа для контекста
            
        Returns:
            dict: Параметры для responses.create()
        """
        params = {
            'model': self.config['model'],
            'instructions': instructions,
            'input': input_data,
            'max_output_tokens': self.config['max_completion'],
            'store': True,  # обязательно для Responses API
        }
        
        # Добавляем previous_response_id если есть
        if previous_response_id:
            params['previous_response_id'] = previous_response_id
        
        # Параметры для reasoning моделей (o*)
        if self.is_reasoning_model:
            params['reasoning'] = {
                'effort': self.config.get('reasoning_effort', 'medium'),
                'summary': self.config.get('reasoning_summary', 'auto')
            }
            # Для reasoning моделей НЕ указываем temperature
        else:
            # Для обычных моделей указываем temperature
            params['temperature'] = self.config.get('temperature', 1.0)
        
        return params
    
    def _clean_json_response(self, response_text: str) -> str:
        """
        Очистка ответа от markdown оберток и других артефактов.
        
        Автоматически удаляет:
        - ```json...``` обертки
        - ```...``` обертки без языка
        - Лишние пробелы в начале/конце
        
        Args:
            response_text: Сырой текст ответа от модели
            
        Returns:
            str: Очищенный текст
        """
        text = response_text.strip()
        
        # Удаляем ```json...``` обертки
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()  # Убираем ```json в начале и ``` в конце
            self.logger.debug("Removed ```json wrapper from response")
        
        # Удаляем обычные ``` обертки
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()  # Убираем ``` в начале и в конце
            self.logger.debug("Removed ``` wrapper from response")
        
        return text

    def _extract_response_content(self, response) -> str:
        """
        Извлечение текста ответа из Responses API.
        
        Для reasoning моделей ожидается структура:
        - output[0]: reasoning
        - output[1]: message
        
        Для обычных моделей:
        - output[0]: message
        """
        try:
            # Проверяем статус ответа
            if hasattr(response, 'status'):
                # Разрешаем как completed, так и incomplete статусы
                if response.status not in ['completed', 'incomplete']:
                    raise ValueError(f"Response has unexpected status: {response.status}")
                
                # Логируем предупреждение для incomplete
                if response.status == 'incomplete':
                    reason = None
                    if hasattr(response, 'incomplete_details') and response.incomplete_details:
                        reason = getattr(response.incomplete_details, 'reason', 'unknown')
                    self.logger.warning(
                        f"Response is incomplete (reason: {reason}). "
                        f"This may happen with low max_output_tokens limits."
                    )
            
            # Проверяем наличие output
            if not response.output or len(response.output) == 0:
                raise ValueError("Response has empty output")
            
            # Определяем индекс message в зависимости от модели
            if self.is_reasoning_model:
                # У reasoning моделей должно быть минимум 2 элемента
                # output[0] - reasoning, output[1] - message
                if len(response.output) < 2:
                    raise ValueError(f"Reasoning model returned insufficient output items: {len(response.output)}")
                message_output = response.output[1]
            else:
                # У обычных моделей message в первом элементе
                message_output = response.output[0]
            
            # Проверяем тип output элемента
            if not hasattr(message_output, 'type') or message_output.type != 'message':
                raise ValueError(f"Expected message output, got type: {getattr(message_output, 'type', 'unknown')}")
            
            # Извлекаем текст из content
            if not hasattr(message_output, 'content') or not message_output.content:
                raise ValueError("Message has no content")
            
            # content - это массив, берем первый элемент с типом output_text
            text_content = None
            refusal_content = None

            for content_item in message_output.content:
                if hasattr(content_item, 'type'):
                    if content_item.type == 'output_text':
                        if hasattr(content_item, 'text'):
                            text_content = content_item.text
                            break
                    elif content_item.type == 'refusal':
                        if hasattr(content_item, 'refusal'):
                            refusal_content = content_item.refusal

            # Проверяем refusal первым
            if refusal_content is not None:
                raise ValueError(f"Model refused to respond: {refusal_content}")
                
            if text_content is None:
                raise ValueError("No text content found in message")
            
            # Очищаем от возможных markdown оберток
            cleaned_text = self._clean_json_response(text_content)
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Failed to extract response content: {e}")
            self.logger.debug(f"Response structure: {response}")
            raise

    def _extract_usage_info(self, response) -> ResponseUsage:
        """
        Извлечение информации об использованных токенах.
        
        Args:
            response: Объект ответа от OpenAI API
            
        Returns:
            ResponseUsage: Структура с информацией о токенах
        """
        try:
            usage = response.usage
            reasoning_tokens = 0
            
            # Извлекаем reasoning токены если есть
            if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
                reasoning_tokens = getattr(usage.output_tokens_details, 'reasoning_tokens', 0)
            
            return ResponseUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                reasoning_tokens=reasoning_tokens
            )
        except AttributeError as e:
            self.logger.warning(f"Could not extract usage info: {e}")
            return ResponseUsage(0, 0, 0, 0)

    def _update_tpm_via_probe(self) -> None:
            """
            Получение актуальных rate limit данных через probe запрос.
            
            Выполняет минимальный синхронный запрос к самой дешевой модели
            для получения response headers с информацией о лимитах.
            
            Note:
                В background режиме (async) OpenAI не возвращает rate limit headers,
                поэтому используем периодические probe запросы для актуализации.
            """
            try:
                self.logger.debug("Executing TPM probe request")
                
                # Используем with_raw_response для доступа к headers
                raw = self.client.responses.with_raw_response.create(
                    model="gpt-4.1-nano-2025-04-14",
                    input="2+2=?",  # Простой математический вопрос
                    max_output_tokens=20,  # Минимум 16, ставим 20 с запасом
                    temperature=0.1,  # Минимальная температура для детерминированности
                    background=False  # ВАЖНО: синхронный режим!
                )
                
                # Извлекаем headers
                headers = dict(raw.headers)
                
                # Обновляем TPM bucket
                self.tpm_bucket.update_from_headers(headers)
                
                self.logger.debug(
                    f"TPM probe successful: remaining={self.tpm_bucket.remaining_tokens}, "
                    f"reset_time={self.tpm_bucket.reset_time}"
                )
                
            except Exception as e:
                self.logger.warning(f"TPM probe failed: {e}")
                # Не критично - продолжаем с текущими данными

    def _create_response_async(self, instructions: str, input_data: str, 
                             previous_response_id: Optional[str] = None) -> Tuple[str, str, ResponseUsage]:
        """
        Создание response в асинхронном режиме с polling.
        
        Использует background=True для немедленного получения response_id,
        затем выполняет polling для отслеживания статуса генерации.
        
        Args:
            instructions: Системный промпт (инструкции для модели)
            input_data: Пользовательские данные (текст или JSON)
            previous_response_id: ID предыдущего ответа для контекста
            
        Returns:
            tuple: (response_text, response_id, usage_info)
            
        Raises:
            TimeoutError: При превышении общего timeout
            IncompleteResponseError: При incomplete статусе
            ValueError: При failed статусе или отказе модели
        """
        # Подготовка параметров как обычно
        params = self._prepare_request_params(instructions, input_data, previous_response_id)
        
        # ВАЖНО: включаем background режим
        params['background'] = True
        
        # Точный подсчёт токенов для проверки лимитов
        full_prompt = instructions + "\n\n" + input_data
        estimated_input_tokens = len(self.encoder.encode(full_prompt))
        required_tokens = estimated_input_tokens + self.config['max_completion']

        # Обновляем TPM данные через probe перед основным запросом
        # В async режиме OpenAI не возвращает rate limit headers,
        # поэтому используем probe для получения актуальной информации
        self._update_tpm_via_probe()

        # Проверяем достаточность токенов с safety margin
        safety_margin = self.config.get('tpm_safety_margin', 0.15)
        self.tpm_bucket.wait_if_needed(required_tokens, safety_margin)
        
        retry_count = 0
        last_exception = None
        
        while retry_count <= self.config['max_retries']:
            try:
                self.logger.debug(f"Async API request attempt {retry_count + 1}: model={params['model']}, "
                                f"estimated_tokens={required_tokens}, previous_response_id={params.get('previous_response_id')}")
                
                # Шаг 1: Создаем background запрос
                raw_initial = self.client.responses.with_raw_response.create(**params)
                initial_headers = raw_initial.headers
                initial_response = raw_initial.parse()
                response_id = initial_response.id
                
                self.logger.info(f"Background response created: {response_id[:8]}...")
                
                # Обновляем TPM из начальных headers
                # ЗАКОММЕНТИРОВАНО: в async режиме headers не содержат rate limit info
                # if initial_headers:
                #    self.tpm_bucket.update_from_headers(initial_headers)
                
                # Шаг 2: Polling loop
                start_time = time.time()
                poll_count = 0
                poll_interval = self.config.get('poll_interval', 5)
                
                while True:
                    elapsed = time.time() - start_time
                    
                    # Проверка hard timeout
                    if elapsed > self.config['timeout']:
                        self.logger.warning(f"Response generation exceeded timeout ({self.config['timeout']}s), cancelling...")
                        try:
                            self.client.responses.cancel(response_id)
                            self.logger.info(f"Successfully cancelled response {response_id[:12]}")
                        except Exception as e:
                            # Race condition: response может уже быть completed/failed
                            self.logger.debug(f"Could not cancel {response_id[:12]}: {e}")
                        
                        raise TimeoutError(f"Response generation exceeded {self.config['timeout']}s timeout")
                    
                    # Получаем текущий статус
                    try:
                        raw_status = self.client.responses.with_raw_response.retrieve(response_id)
                        status_headers = raw_status.headers
                        response = raw_status.parse()
                    except Exception as e:
                        self.logger.error(f"Failed to retrieve response status: {e}")
                        raise
                    
                    # Обновляем TPM из headers статусного запроса
                    # ЗАКОММЕНТИРОВАНО: в async режиме headers не содержат rate limit info
                    # if status_headers:
                    #     self.tpm_bucket.update_from_headers(status_headers)
                    
                    # Обработка статусов
                    status = response.status
                    
                    if status == 'completed':

                        # Успешное завершение
                        self.logger.info(f"Response {response_id[:12]} completed successfully")
                        
                        # Извлекаем результат
                        response_text = self._extract_response_content(response)
                        usage_info = self._extract_usage_info(response)
                        
                        # Сохраняем response_id для следующего вызова
                        self.last_response_id = response_id
                        
                        self.logger.debug(f"Async response success: id={response_id}, "
                                        f"tokens={usage_info.total_tokens} "
                                        f"(reasoning: {usage_info.reasoning_tokens})")
                        
                        return response_text, response_id, usage_info
                    
                    elif status == 'incomplete':
                        # Критическая ошибка - ответ обрезан
                        reason = "unknown"
                        if hasattr(response, 'incomplete_details') and response.incomplete_details:
                            reason = getattr(response.incomplete_details, 'reason', 'unknown')
                        
                        # Пытаемся получить частичный результат для логирования
                        partial_tokens = 0
                        if hasattr(response, 'usage') and response.usage:
                            partial_tokens = response.usage.output_tokens
                        
                        self.logger.error(
                            f"Response {response_id[:12]} incomplete: {reason}. "
                            f"Generated {partial_tokens} tokens before hitting limit."
                        )
                        
                        # Выводим в терминал
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(f"[{current_time}] ERROR    | ❌ Response incomplete: {reason}")
                        print(f"[{current_time}] ERROR    |    Generated only {partial_tokens} tokens")
                        
                        # Для reasoning моделей это особенно критично
                        if self.is_reasoning_model and reason == "max_output_tokens":
                            current_time = datetime.now().strftime("%H:%M:%S")
                            print(f"[{current_time}] HINT     | 💡 Reasoning model needs more tokens. "
                                f"Current limit: {self.config['max_completion']}")
                        
                        # Бросаем специальное исключение для обработки в retry логике
                        raise IncompleteResponseError(
                            f"Response generation incomplete ({reason}). "
                            f"Generated only {partial_tokens} tokens."
                        )
                    
                    elif status == 'failed':
                        # Ошибка генерации
                        error_msg = "Unknown error"
                        if hasattr(response, 'error') and response.error:
                            error_msg = getattr(response.error, 'message', str(response.error))
                        
                        self.logger.error(f"Response {response_id[:12]} failed: {error_msg}")
                        
                        raise ValueError(f"Response generation failed: {error_msg}")
                    
                    elif status == 'cancelled':
                        # Не должно происходить, если мы сами не отменили
                        self.logger.error(f"Response {response_id[:12]} was cancelled unexpectedly")
                        raise ValueError("Response was cancelled")

                    elif status == 'queued':
                        # Показываем начальный статус только один раз
                        if poll_count == 0:
                            current_time = datetime.now().strftime("%H:%M:%S")
                            print(f"[{current_time}] QUEUE    | ⏳ Response {response_id[:12]}... in progress")
                        
                        # Показываем прогресс каждые 3 проверки (~21 сек при poll_interval=7)
                        elif poll_count > 0 and poll_count % 3 == 0:
                            elapsed_time = int(time.time() - start_time)
                            current_time = datetime.now().strftime("%H:%M:%S")
                            
                            # Форматируем время
                            if elapsed_time < 60:
                                time_str = f"{elapsed_time}s"
                            else:
                                minutes = elapsed_time // 60
                                seconds = elapsed_time % 60
                                time_str = f"{minutes}m {seconds}s"
                            
                            print(f"[{current_time}] PROGRESS | ⏳ Elapsed: {time_str}")
                        
                        # Адаптивный интервал polling
                        if poll_count < 3:
                            time.sleep(2)  # Первые 3 раза проверяем быстро
                        else:
                            time.sleep(poll_interval)  # Потом используем настроенный интервал
                        
                        poll_count += 1
                        continue
                    
                    else:
                        # Неизвестный статус
                        self.logger.error(f"Unknown response status: {status}")
                        raise ValueError(f"Unknown response status: {status}")

            except (openai.RateLimitError, IncompleteResponseError) as e:
                # Эти ошибки можем retry
                retry_count += 1
                if retry_count > self.config['max_retries']:
                    self.logger.error(f"Error after all retries: {type(e).__name__}: {e}")
                    raise e
                
                # Для IncompleteResponseError увеличиваем лимит токенов
                if isinstance(e, IncompleteResponseError):
                    # Сохраняем текущий лимит перед изменением
                    old_limit = params.get('max_output_tokens', self.config['max_completion'])
                    if retry_count == 1:
                        params['max_output_tokens'] = int(old_limit * 1.5)
                    elif retry_count == 2:
                        params['max_output_tokens'] = int(old_limit * 2.0)
                    else:
                        # Больше не увеличиваем
                        raise ValueError(
                            f"Response still incomplete after {retry_count} retries. "
                            f"Max tokens tried: {params['max_output_tokens']}"
                        )
                    
                    self.logger.info(
                        f"Retrying with increased token limit: {old_limit} → {params['max_output_tokens']}"
                    )
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"[{current_time}] RETRY    | 🔄 Increasing token limit: {old_limit} → {params['max_output_tokens']}")
                
                wait_time = 20 * (2 ** (retry_count - 1))
                self.logger.warning(f"{type(e).__name__}, retry {retry_count}/{self.config['max_retries']} in {wait_time}s")
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time}] RETRY    | ⏳ Waiting {wait_time}s before retry {retry_count}/{self.config['max_retries']}...")
                time.sleep(wait_time)
                last_exception = e
                
            except TimeoutError as e:
                # Timeout не retry - сразу выбрасываем
                raise e
                
            except Exception as e:
                # Другие ошибки тоже можем попробовать retry
                retry_count += 1
                if retry_count > self.config['max_retries']:
                    self.logger.error(f"Error after all retries: {type(e).__name__}: {e}")
                    raise e
                
                wait_time = 20 * (2 ** (retry_count - 1))
                self.logger.warning(f"{type(e).__name__}, retry {retry_count}/{self.config['max_retries']} in {wait_time}s: {e}")
                time.sleep(wait_time)
                last_exception = e
        
        # Если дошли сюда - исчерпаны все retry
        raise last_exception or Exception("Max retries exceeded")

    def create_response(self, instructions: str, input_data: str, 
                       previous_response_id: Optional[str] = None) -> Tuple[str, str, ResponseUsage]:
        """
        Создание response через OpenAI Responses API.
        
        Автоматически:
        - Управляет previous_response_id для сохранения контекста
        - Проверяет лимиты через headers с safety margin
        - Выполняет retry с exponential backoff при ошибках
        - Обрабатывает специфику reasoning моделей
        - Обновляет TPM bucket из response headers
        
        Args:
            instructions: Системный промпт (инструкции для модели)
            input_data: Пользовательские данные (текст или JSON)
            previous_response_id: ID предыдущего ответа (если не указан,
                использует сохраненный last_response_id)
            
        Returns:
            tuple: (response_text, response_id, usage_info)
                - response_text (str): Текст ответа модели
                - response_id (str): ID ответа для использования в цепочке
                - usage_info (ResponseUsage): Информация об использованных токенах
                
        Raises:
            openai.RateLimitError: При превышении rate limit после всех retry
            openai.APIError: При ошибках API после всех retry
            ValueError: При некорректном ответе или отказе модели
            
        Example:
            >>> # Простой запрос
            >>> text, resp_id, usage = client.create_response(
            ...     "You are a helpful assistant",
            ...     "What is 2+2?"
            ... )
            >>> print(f"Response: {text}")
            >>> print(f"Used {usage.total_tokens} tokens")
            >>> 
            >>> # Цепочка запросов с контекстом
            >>> text2, resp_id2, usage2 = client.create_response(
            ...     "Continue the conversation",
            ...     "And what is 3+3?",
            ...     previous_response_id=resp_id
            ... )
        """
        # Используем переданный previous_response_id или сохраненный
        if previous_response_id is None:
            previous_response_id = self.last_response_id
            
        # Вызываем новый асинхронный метод
        return self._create_response_async(instructions, input_data, previous_response_id)

    def repair_response(self, instructions: str, input_data: str) -> Tuple[str, str, ResponseUsage]:
        """
        Repair запрос с тем же previous_response_id.
        
        Используется когда LLM вернул некорректный ответ (например,
        невалидный JSON). Добавляет к инструкциям требование
        вернуть только валидный JSON и повторяет запрос с тем же
        контекстом.
        
        Args:
            instructions: Оригинальный системный промпт
            input_data: Оригинальные пользовательские данные
            
        Returns:
            tuple: (response_text, response_id, usage_info)
                См. create_response() для деталей
                
        Example:
            >>> try:
            ...     text, resp_id, usage = client.create_response(prompt, data)
            ...     result = json.loads(text)  # Может упасть
            ... except json.JSONDecodeError:
            ...     # Пробуем repair
            ...     text, resp_id, usage = client.repair_response(prompt, data)
            ...     result = json.loads(text)  # Должен быть валидный JSON
        """
        repair_instructions = (
            instructions + 
            "\n\nIMPORTANT: Return **valid JSON** only. "
            "Do NOT use markdown formatting like ```json```. "
            "Do NOT include any explanatory text. "
            "Return ONLY the raw JSON object."
        )
        
        # Используем тот же previous_response_id что и в предыдущем запросе
        return self.create_response(repair_instructions, input_data, self.last_response_id)