#!/usr/bin/env python3
"""
Диагностический скрипт для отладки OpenAI Responses API
Проверяет реальные статусы и timing работы с o4-mini и обычными моделями

Использование:
- Измените test_reasoning = True для тестирования reasoning модели
- Измените test_reasoning = False для тестирования обычной модели
"""

import os
import time
import json
from datetime import datetime
from openai import OpenAI

# Инициализация клиента
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Создаём запрос
try:
    # Выбираем модель для теста
    test_reasoning = False  # Переключатель для тестирования
    
    if test_reasoning:
        # Reasoning модель
        model = "o4-mini-2025-04-16"
        params = {
            "model": model,
            "instructions": "Ты литературный критик.",
            "input": "Напиши краткое содержание Гамлета? Сцены, описания события, персонажи. Пиши литературно текстом, без таблиц и списков, минимум 30 абзацев.",
            "max_output_tokens": 25000,
            "background": True,  # ВАЖНО: асинхронный режим!
            "reasoning": {
                "effort": "medium",
                "summary": "auto"
            },
            "store": True
        }
    else:
        # Обычная модель
        model = "gpt-4.1-mini-2025-04-14"
        params = {
            "model": model,
            "instructions": "Ты литературный критик.",
            "input": "Напиши краткое содержание Гамлета? Сцены, описания события, персонажи. Пиши литературно текстом, без таблиц и списков, минимум 30 абзацев.",
            "max_output_tokens": 10000,  # Меньше для обычной модели
            "background": True,  # ВАЖНО: асинхронный режим!
            "temperature": 0.7,  # Для обычных моделей
            "store": True
        }
    
    response = client.responses.create(**params)
    
    response_id = response.id
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Response ID: {response_id}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initial status: {response.status}")
    
    # Показываем все доступные поля для отладки
    available_fields = [attr for attr in dir(response) if not attr.startswith('_')]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Available fields: {', '.join(available_fields[:10])}...")
    print("-" * 80)
    
except Exception as e:
    print(f"ERROR creating response: {e}")
    exit(1)

# Polling loop
poll_count = 0
start_time = time.time()
statuses_seen = []

while True:
    poll_count += 1
    elapsed = time.time() - start_time
    
    try:
        # Получаем статус
        response = client.responses.retrieve(response_id)
        
        # Сохраняем историю статусов
        if not statuses_seen or statuses_seen[-1] != response.status:
            statuses_seen.append(response.status)
        
        # Формируем вывод
        timestamp = datetime.now().strftime('%H:%M:%S')
        if elapsed < 60:
            elapsed_str = f"{int(elapsed)}s"
        else:
            elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
        
        # Минимальный JSON с важными полями
        status_info = {
            "poll": poll_count,
            "elapsed": elapsed_str,
            "status": response.status,
            "created_at": response.created_at,
            "model": response.model
        }
        
        # Для отладки - показываем service_tier если есть
        if hasattr(response, 'service_tier'):
            status_info["service_tier"] = response.service_tier
        
        # Добавляем usage если есть
        if hasattr(response, 'usage') and response.usage:
            usage_data = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.total_tokens
            }
            # Reasoning токены только для reasoning моделей
            if response.usage.output_tokens_details and hasattr(response.usage.output_tokens_details, 'reasoning_tokens'):
                usage_data["reasoning"] = response.usage.output_tokens_details.reasoning_tokens
            status_info["usage"] = usage_data
        
        # Показываем error если есть
        if hasattr(response, 'error') and response.error:
            status_info["error"] = str(response.error)
        
        # Если есть output, показываем длину текста
        if hasattr(response, 'output') and response.output:
            output_info = []
            for item in response.output:
                if item.type == "reasoning":
                    output_info.append("reasoning")
                elif item.type == "message" and hasattr(item, 'content'):
                    for content in item.content:
                        if hasattr(content, 'text'):
                            status_info["output_length"] = len(content.text)
                            output_info.append("message")
                            break
            if output_info:
                status_info["output_types"] = output_info
        
        print(f"[{timestamp}] {json.dumps(status_info, ensure_ascii=False)}")
        
        # Специальная отладка для queued статуса
        if response.status == "queued" and poll_count == 1:
            print(f"[{timestamp}] DEBUG: Response is queued, this should be expected with background=True")
        
        # Проверяем завершение
        if response.status in ["completed", "failed", "cancelled", "incomplete"]:
            print("-" * 80)
            print(f"[{timestamp}] FINAL STATUS: {response.status}")
            
            if response.status == "completed":
                # Показываем начало текста
                # У reasoning моделей: output[0]=reasoning, output[1]=message
                # У обычных моделей: output[0]=message
                for idx, item in enumerate(response.output):
                    if item.type == "message":
                        print(f"Message found at output[{idx}]")
                        for content in item.content:
                            if hasattr(content, 'text'):
                                print(f"Output preview (first 200 chars): {content.text[:200]}...")
                                break
                        break
            elif response.status == "failed" and hasattr(response, 'error'):
                print(f"Error: {response.error}")
            elif response.status == "incomplete" and hasattr(response, 'incomplete_details'):
                print(f"Incomplete reason: {response.incomplete_details}")
                
            break
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR polling: {e}")
        break
    
    # Ждём 3 секунды
    time.sleep(3)

print(f"\nModel: {model} ({'REASONING' if test_reasoning else 'REGULAR'})")
print(f"Total time: {time.time() - start_time:.1f}s")
print(f"Total polls: {poll_count}")
print(f"Status sequence: {' -> '.join(statuses_seen)}")