# K2-18: Руководство по доменной адаптации промптов

## 1. Введение и проблематика

### 1.1. Зачем нужна доменная адаптация

Проект K2-18 разрабатывался и тестировался на материалах по Computer Science. В результате все промпты системы содержат CS-специфичные примеры, критерии и терминологию, что создает систематические проблемы при обработке других предметных областей:

**Смещение извлечения (Extraction Bias)**
- LLM ищет "алгоритмические" паттерны в неалгоритмических текстах
- Игнорирует доменно-специфичные типы концептов
- Неверно интерпретирует важность концептов

**Некорректная типизация связей**
- Экономика: "Спрос влияет на цену" → PREREQUISITE (неверно) вместо ELABORATES
- История: "Причина → Следствие" → EXAMPLE_OF (неверно)
- Веса связей откалиброваны под технические зависимости

**Неадекватная оценка сложности**
- В CS difficulty=5 = "research-level алгоритмы"
- В экономике difficulty=5 должна означать эконометрические модели
- В истории — анализ источников и историография

**Потеря доменной специфики**
- Экономические теории и модели плохо укладываются в паттерн "алгоритм"
- Исторические события требуют временной контекстуализации
- Причинно-следственные связи в гуманитарных науках отличаются от технических

### 1.2. Эволюция поддержки доменов

- **v1.0**: Computer Science (`_cs`) — базовая реализация
- **v2.0**: Economics (`_econ`) — первая адаптация, выявление паттернов
- **v2.5**: Management (`_mgmt`) + Communications (`_comm`) — расширение на управленческие и коммуникационные дисциплины

На основе опыта адаптации 4 доменов сформирован универсальный подход для любых будущих предметных областей.

### 1.3. Выбранное решение

**Создание доменно-специфичных наборов промптов** с сохранением структуры данных и типов связей.

**Ключевой принцип**: меняем только промпты, код остается неизменным.

---

## 2. Архитектура доменной адаптации

### 2.1. Организация файловой структуры

```
/src/prompts/
├── itext2kg_concepts_extraction.md         # Активные промпты (текущий домен)
├── itext2kg_graph_extraction.md
├── refiner_longrange_fw.md
├── refiner_longrange_bw.md
│
├── itext2kg_concepts_extraction_cs.md      # Computer Science
├── itext2kg_graph_extraction_cs.md
├── refiner_longrange_fw_cs.md
├── refiner_longrange_bw_cs.md
│
├── itext2kg_concepts_extraction_econ.md    # Economics
├── ...
│
├── itext2kg_concepts_extraction_mgmt.md    # Management
├── ...
│
├── itext2kg_concepts_extraction_comm.md    # Communications
└── ...
```

### 2.2. Процесс переключения доменов

**Текущая реализация** - ручное переключение:
1. Копировать нужные `*_[domain].md` → `*.md` (без суффикса)
2. Запустить pipeline

### 2.3. Принципы разделения ответственности

**Что контролирует код**:
- Структуры данных (JSON схемы)
- Алгоритмы обработки
- Валидация
- Workflow pipeline

**Что контролируют промпты**:
- Извлечение концептов
- Определение типов связей
- Оценка сложности
- Интерпретация доменной специфики

---

## 3. Инварианты системы (что НЕ меняется)

При создании нового домена следующие элементы **остаются строго фиксированными**:

### 3.1. Структуры данных
- **JSON схемы**: `ConceptDictionary.schema.json`, `LearningChunkGraph.schema.json`  
- **Валидация данных**: унифицирована во всех утилитах, выполняется автоматически  
- **Формат метаданных**: `_meta` блоки в выходных файлах

### 3.2. Типы узлов и связей
- **3 типа узлов**: Chunk, Concept, Assessment  
- **9 типов связей**: PREREQUISITE, ELABORATES, EXAMPLE_OF, PARALLEL, TESTS, REVISION_OF, HINT_FORWARD, REFER_BACK, MENTIONS

### 3.3. Алгоритмы
- **Алгоритм приоритизации связей**: строго фиксированный порядок проверки типов  
- **Система весов**: [0.3, 1.0] с дискретизацией по 0.05  
  - ≥ 0.80 — сильная связь (PREREQUISITE, TESTS, REVISION_OF)
  - 0.50–0.75 — ясная связь (ELABORATES, EXAMPLE_OF, PARALLEL)
  - 0.30–0.45 — слабая связь (HINT_FORWARD, REFER_BACK, MENTIONS)
  - < 0.30 — не создавать

### 3.4. Структура промптов
- **Разделы промптов**: одинаковая структура для всех доменов  
- **Формат инструкций**: единообразный стиль команд для LLM  
- **JSON выходы**: идентичные форматы ответов

---

## 4. Адаптируемые элементы (что ВСЕГДА меняется)

При создании нового домена **обязательно адаптируются**:

### 4.1. Примеры входных данных (Input: Context Provided)

Место адаптации: промпты извлечения концептов и построения графа, секция **### Sub-categories and Nuanced Constraints → Input: Context Provided**.

**Для `itext2kg_concepts_extraction_[domain].md`:**
- Пример ConceptDictionary с 1-2 доменными концептами (с concept_id, term, definition)
- Пример Slice с доменным текстом в поле `text`

**Для `itext2kg_graph_extraction_[domain].md`:**
- Пример ConceptDictionary с доменными концептами
- Пример Slice с доменным текстом

**Требования:**
- Тексты должны быть **реалистичными** для домена
- ConceptDictionary примеры должны соответствовать формату схемы
- Slug в Slice должен отражать домен (cs101, econ101, mgmt101, comm101)

### 4.2. Критерии извлечения концептов

Каждый домен имеет специфические типы сущностей, которые должны распознаваться как концепты.

#### Идентификация кандидатов в концепты

Место адаптации: `itext2kg_concepts_extraction_[domain].md`, секция **Reasoning Steps, пункт 1**.

**Для Computer Science**:
```
Identify key candidate concepts - distinct algorithms, data structures, complexity measures, 
programming paradigms, or computational concepts that appear in the slice.text.
```

**Для Economics**:
```
Identify key candidate concepts - distinct economic terms, market mechanisms, theoretical models, 
economic policies, or financial instruments that appear in the slice.text.
```

**Для Management**:
```
Identify key candidate concepts - distinct management terms, methodologies, frameworks, 
organizational phenomena, or management tools that appear in the slice.text.
```

**Для Communications**:
```
Identify key candidate concepts - distinct communication channels, audience types, content formats, 
media strategies, or messaging approaches that appear in the slice.text.
```

**Общее правило**: Prioritize terms that are highlighted (e.g., in bold or italics) and immediately followed by their definition.

#### Детальные критерии извлечения

Место адаптации: `itext2kg_concepts_extraction_[domain].md`, секция **Reasoning Steps, пункт 4**.

**Для Computer Science**:
- Структуры данных и алгоритмы
- Вычислительная сложность (Big O notation)
- Парадигмы программирования
- Формальные методы и доказательства

**Для Economics**:
- Экономические агенты и институты (домохозяйства, фирмы, государство)
- Рыночные механизмы (спрос, предложение, равновесие)
- Макро- и микроэкономические модели (IS-LM, DSGE)
- Экономические политики (фискальная, монетарная)

**Для Management**:
- Управленческие роли и функции
- Методологии и фреймворки (PDCA, DMAIC, Scrum)
- Организационные феномены (культура, климат, конфликты)
- Инструменты управления (от диаграммы Ганта до BSC)
- Теории и модели (от Тейлора до Голдратта)

**Для Communications**:
- Каналы и инструменты коммуникации
- Типы аудиторий и стейкхолдеров
- Форматы контента и сообщений
- Метрики и показатели эффективности
- Коммуникационные стратегии и тактики

### 4.3. Примеры извлечения концептов

Место адаптации: `itext2kg_concepts_extraction_[domain].md` секция **Examples** содержит:
- 4 обязательных базовых примера
- 1-3 доменно-специфичных примера

#### Структура примеров (строго в этом порядке):

**Базовые примеры (4 штуки, одинаковые для всех доменов по логике):**
1. **New concept extraction** - извлечение нового концепта с определением
2. **Existing concept - no addition** - концепт уже есть в словаре, ничего не добавляем
3. **Adding aliases to existing concept** - добавление новых алиасов к существующему концепту
4. **Discarding a minor term** - отбрасывание неважного термина, который только упомянут

**Доменно-специфичные примеры (1-3 штуки):**
- Примеры, демонстрирующие специфику извлечения концептов в данном домене
- Например для Management: "Organizational theory extraction", "Management methodology with framework", "Leadership concept"

#### Формат каждого примера:

```
### Example: [Scenario name]
**Slice.text**: "[Fragment of educational text in domain language]"
**ConceptDictionary**: [current state - empty or contains some concepts]
**Output**:
```json
{
  "concepts_added": {
    "concepts": [...]
  }
}
```

### 4.4. Шкала difficulty узлов Chunk и Assessment (для graph extraction)

Место адаптации: `itext2kg_graph_extraction_[domain].md`, секция **Reasoning Steps → Phase 1: NODES EXTRACTION**.

**Структура сохраняется** (5 уровней от базового до исследовательского), но **содержание переопределяется** под специфику домена.

| Уровень | CS | Economics | Management | Communications |
|---------|----|-----------|-----------|--------------------|
| 1 | Простое определение | Базовые термины | Основные понятия | Базовые определения |
| 2 | Простой алгоритм | Простая модель | Инструменты и методы | Простые инструменты |
| 3 | Алгоритм O(n²) | Модель с 3-5 переменными | Системные подходы | Интегрированные кампании |
| 4 | Доказательство, сложный код | Сложные модели (DSGE) | Стратегический уровень | Стратегическое планирование |
| 5 | Research алгоритмы | Нобелевские теории | Теория сложности | Нейромаркетинг, big data |

### 4.5. Определение типов связей (для graph extraction)

Место адаптации: `itext2kg_graph_extraction_[domain].md`, секция **Reasoning Steps → Phase 2: EDGES GENERATION**, Inline-примеры пунктов 1-3 в алгоритме приоритизации связей **обязательно адаптируются** под домен.

Для связи типа **PREREQUISITE**:
- CS: `(e.g., {source} defines "Graph," and {target} describes "BFS algorithm")`
- Econ: `(e.g., {source} defines "спрос и предложение," and {target} describes "рыночное равновесие")`
- Mgmt: `(e.g., {source} defines "организационная структура," and {target} describes "матричная структура управления")`
- Comm: `(e.g., {source} defines "целевая аудитория," and {target} describes "сегментация аудитории")`

Для связи типа **ELABORATES**:
- CS: `(e.g., {target} mentions "sorting," and {source} provides detailed Quicksort implementation)`
- Econ: `(e.g., {target} introduces equilibrium concept, and {source} provides mathematical IS-LM model)`
- Mgmt: `(e.g., {target} mentions Agile briefly, and {source} provides detailed Scrum implementation)`
- Comm: `(e.g., {target} introduces media planning, and {source} provides step-by-step campaign execution)`

Для связи типа **PARALLEL**: Если есть характерные альтернативные подходы в домене, добавить inline-пример

Типы связей **EXAMPLE_OF**, **TESTS**, **REVISION_OF**, **HINT_FORWARD**, **REFER_BACK** не требуют адаптации inline-примеров - они универсальны.

### 4.6. Standalone-примеры в Edge Types Heuristics Guide (для graph extraction)

Место адаптации: `itext2kg_graph_extraction_[domain].md`, секция **Example: Edge Types Heuristics Guide**.

#### Формат Standalone-примеров связей:

```
- **EDGE_TYPE**: Source description (`source`) → Target description (`target`) - explanation
```

#### Примеры по доменам:

**PREREQUISITE** (необходимо для понимания)
- CS: "Graph" → "Graph Search Algorithm"
- Econ: "Supply & Demand" → "Market Equilibrium"
- Mgmt: "Organizational Structure" → "Delegation"
- Comm: "Shannon-Weaver Model" → "Communication Barriers"

**ELABORATES** (детализирует)
- CS: "Formal proof of correctness" → "Algorithm description"
- Econ: "IS-LM mathematical model" → "Macroeconomic equilibrium concept"
- Mgmt: "Detailed description of matrix structure" → "Types of organizational structures"
- Comm: "Step-by-step media planning" → "Media planning basics"

**EXAMPLE_OF** (является примером)
- CS: "Bubble Sort" → "Sorting Algorithms"
- Econ: "Great Depression" → "Economic Crises"
- Mgmt: "Toyota Production System" → "Lean Methodology"
- Comm: "Dove Real Beauty Campaign" → "Socially Responsible Marketing"

**PARALLEL** (альтернативные подходы)
- CS: "DFS" ↔ "BFS"
- Econ: "Keynesian Theory" ↔ "Monetarism"
- Mgmt: "Theory X" ↔ "Theory Y"
- Comm: "PR Strategy" ↔ "Marketing Strategy"

**TESTS** (проверяет знания)
- CS: "Implement selection sort" → "Sorting Algorithms"
- Econ: "Analyze market failure case" → "Market Failures"
- Mgmt: "Develop crisis management plan" → "Crisis Management"
- Comm: "Create anti-crisis communication plan" → "Crisis Communications"

**REVISION_OF** (обновление/пересмотр)
- CS: "Quicksort optimized" → "Classic Quicksort"
- Econ: "New Keynesian Economics" → "Classical Keynesian"
- Mgmt: "Agile 2.0" → "Classic Agile"
- Comm: "Digital-first approach" → "Traditional media strategies"

**HINT_FORWARD** (отсылка вперед)
- CS: "We'll discuss sorting later" → "Sorting Algorithms"
- Econ: "We'll return to fiscal policy" → "Fiscal Policy Tools"
- Mgmt: "We'll explore motivation later" → "Motivation Theories"
- Comm: "We'll return to metrics later" → "Digital KPIs"

**REFER_BACK** (отсылка назад)
- CS: "As we discussed in data structures" → "Data Structures"
- Econ: "As discussed in market structures" → "Perfect Competition"
- Mgmt: "As mentioned in leadership topic" → "Leadership Styles"
- Comm: "Recall segmentation principles" → "Audience Segmentation"

**MENTIONS** (упоминает в контексте)
- CS: "Used in the context of O(n)" → "Algorithm Complexity"
- Econ: "Considering inflation trends" → "Inflation"
- Mgmt: "In KPI context" → "Balanced Scorecard"
- Comm: "Considering influencer trends" → "Influencer Marketing"

### 4.7. Комплексный пример графа (для graph extraction)

Место адаптации: `itext2kg_graph_extraction_[domain].md`, секция **Example: Output** в самом конце промпта.

**Что адаптируется:**
- Все тексты узлов (chunk_1, chunk_2, concept nodes, assessment_1)
- Все definitions узлов
- Все concept_id (должны соответствовать домену)
- Все примеры связей между узлами
- Поля difficulty для Chunk/Assessment узлов

**Требования:**
- Пример должен быть **реалистичным** для домена
- Демонстрировать **разнообразие типов связей** (минимум 5-6 разных типов)
- Показывать **корректную работу** алгоритма приоритизации

### 4.8. Адаптация рефайнеров (longrange forward/backward)

Место адаптации: промпты `refiner_longrange_fw_[domain].md` и `refiner_longrange_bw_[domain].md` адаптируются **аналогично основным промптам** в трех местах:

**Адаптируемые секции:**
1. **Reasoning Steps** - доменная терминология в описании логики поиска связей
2. **Examples: Edge Types Heuristics Guide** - примеры для каждого типа связи (по 1-2 на тип)
3. **Example: Input/Output** - комплексный пример с source_node и candidates

**Особенности:**
- **FW** (source раньше target): фокус на PREREQUISITE, EXAMPLE_OF, HINT_FORWARD, PARALLEL, TESTS, MENTIONS
- **BW** (source позже target): фокус на ELABORATES, REVISION_OF, TESTS, EXAMPLE_OF, PARALLEL, MENTIONS, REFER_BACK

**Важно**: При адаптации примеров связей необходимо сохранять **осмысленность направления** (source→target должно соответствовать временной последовательности в тексте).

---

## 5. Методология создания нового домена

### 5.1. Анализ предметной области

**Лучший источник для калибровки домена**: Рабочая программа дисциплины (РПД) из ВУЗа или вводная лекция по предмету. Эти материалы позволяют быстро понять:
- Структуру предметной области
- Ключевые концепты и их иерархию
- Типичные формулировки и терминологию
- Связи между темами

**Вопросы для анализа**:
- Какие дисциплины охватывает домен?
- Какой тип мышления преобладает? (алгоритмическое, аналитическое, системное, креативное)
- Какие типы сущностей являются ключевыми? (объекты, процессы, агенты, концепции)
- Есть ли формализация? (математика, модели, фреймворки)
- Как выражаются зависимости? (логические, причинно-следственные, иерархические)

### 5.2. Чеклист адаптации промптов

#### `itext2kg_concepts_extraction_[domain].md`
- [ ] **Input: Context Provided** - примеры ConceptDictionary и Slice с доменными данными
- [ ] **Reasoning Steps, пункт 1** - идентификация кандидатов в концепты (доменная формулировка)
- [ ] **Reasoning Steps, пункт 4** - критерии извлечения концептов (список типов сущностей)
- [ ] **Examples** - 5 базовых + 1-3 доменно-специфичных примера

#### `itext2kg_graph_extraction_[domain].md`
- [ ] **Input: Context Provided** - примеры ConceptDictionary и Slice с доменными данными
- [ ] **Reasoning Steps → Phase 1** - шкала difficulty (описание 5 уровней)
- [ ] **Reasoning Steps → Phase 2, пункты 1-3** - inline-примеры для PREREQUISITE, ELABORATES, PARALLEL
- [ ] **Edge Types Heuristics Guide** - примеры для всех 9 типов связей
- [ ] **Example: Output** - комплексный JSON пример графа

#### `refiner_longrange_fw_[domain].md`
- [ ] **Reasoning Steps** - доменная терминология
- [ ] **Examples: Edge Types Heuristics Guide** - примеры связей для FW типов
- [ ] **Example: Input/Output** - комплексный пример

#### `refiner_longrange_bw_[domain].md`
- [ ] **Reasoning Steps** - доменная терминология
- [ ] **Examples: Edge Types Heuristics Guide** - примеры связей для BW типов
- [ ] **Example: Input/Output** - комплексный пример

---

## 6. Справочник поддерживаемых доменов

### 6.1. Computer Science (`_cs`)

**Охват**: алгоритмы, структуры данных, программирование, теория вычислений, базы данных, сети

**Характеристики**:
- Тип мышления: алгоритмическое
- Фокус: код, структуры данных, вычислительная сложность
- Формализация: высокая (математика, псевдокод)
- Характер связей: логические, процессные

**Шкала difficulty**:
1. Простое определение (что такое переменная)
2. Простой алгоритм или структура данных (линейный поиск)
3. Алгоритм O(n²) или структура данных средней сложности (сортировка, граф)
4. Формальное доказательство, тяжёлый код (алгоритмы на графах, DP)
5. Research-level алгоритмы (ML, криптография, квантовые вычисления)

**Типичные концепты**: Stack, Queue, Graph, Algorithm, O(n), Recursion, Hash Table

**Паттерны связей**:
- PREREQUISITE: "Data Structure" → "Algorithm using it"
- PARALLEL: "DFS" ↔ "BFS"
- EXAMPLE_OF: "Bubble Sort" → "Sorting Algorithms"

### 6.2. Economics (`_econ`)

**Охват**: микро- и макроэкономика, эконометрика, финансы, экономическая теория, политэкономия

**Характеристики**:
- Тип мышления: аналитическое
- Фокус: модели, рынки, оптимизация, равновесие
- Формализация: высокая (математические модели)
- Характер связей: причинно-следственные, системные

**Шкала difficulty**:
1. Базовые термины (что такое ВВП, инфляция)
2. Простая модель или закон (кривая спроса, закон убывающей отдачи)
3. Модель с 3-5 переменными (теория потребительского выбора)
4. Сложные модели (DSGE, эконометрические регрессии)
5. Нобелевские теории и продвинутая эконометрика

**Типичные концепты**: GDP, Inflation, Elasticity, Supply & Demand, Market Equilibrium, IS-LM Model

**Паттерны связей**:
- PREREQUISITE: "Supply & Demand" → "Market Equilibrium"
- PARALLEL: "Keynesian Theory" ↔ "Monetarism"
- ELABORATES: "Mathematical IS-LM model" → "Macroeconomic equilibrium"

### 6.3. Management (`_mgmt`)

**Охват**: менеджмент, управление проектами, лидерство, организационное развитие, стратегическое управление

**Характеристики**:
- Тип мышления: системное
- Фокус: организации, процессы, люди, решения
- Формализация: средняя (фреймворки, модели)
- Характер связей: процессные, иерархические, практические

**Шкала difficulty**:
1. Основные понятия (организация, планирование, контроль)
2. Инструменты и методы (SWOT, базовые KPI, простые модели мотивации)
3. Системные подходы (процессное управление, Agile, Lean, работа с командами)
4. Стратегический уровень (организационные изменения, M&A, трансформация)
5. Исследовательский уровень (теория сложности, эмерджентность, организационная кибернетика)

**Типичные концепты**: KPI, Organizational Structure, Agile, Lean, Leadership, Delegation, SWOT, BSC

**Паттерны связей**:
- PREREQUISITE: "Organizational Structure" → "Delegation"
- PARALLEL: "Theory X" ↔ "Theory Y"
- EXAMPLE_OF: "Toyota Production System" → "Lean Methodology"

### 6.4. Communications (`_comm`)

**Охват**: PR, маркетинговые коммуникации, медиа, реклама, цифровые коммуникации, журналистика

**Характеристики**:
- Тип мышления: коммуникативное, креативное
- Фокус: сообщения, аудитории, каналы, влияние
- Формализация: средняя/низкая (модели, метрики, но много качественного анализа)
- Характер связей: процессные, стратегические, креативные

**Шкала difficulty**:
1. Базовые определения (коммуникация, целевая аудитория, канал, сообщение)
2. Простые инструменты (пресс-релиз, пост в соцсетях, базовая инфографика)
3. Интегрированные кампании (координация каналов, таргетирование, A/B тестирование)
4. Стратегическое планирование (репутационный менеджмент, антикризисные коммуникации, ребрендинг)
5. Исследовательский уровень (нейромаркетинг, семиотический анализ, big data в коммуникациях)

**Типичные концепты**: Target Audience, Content Strategy, Engagement, Brand Reputation, Media Planning

**Паттерны связей**:
- PREREQUISITE: "Shannon-Weaver Model" → "Communication Barriers"
- PARALLEL: "PR Strategy" ↔ "Marketing Strategy"
- EXAMPLE_OF: "Dove Real Beauty" → "Socially Responsible Marketing"

### Сравнительная матрица доменов

| Аспект | CS | Econ | Mgmt | Comm |
|--------|-----|------|------|------|
| **Тип мышления** | Алгоритмическое | Аналитическое | Системное | Коммуникативное |
| **Фокус** | Код, алгоритмы, структуры данных | Модели, рынки, оптимизация | Организации, процессы, люди | Сообщения, аудитории, влияние |
| **Формализация** | Высокая | Высокая | Средняя | Средняя/Низкая |
| **Difficulty 1** | Простое определение | Базовые термины | Основные понятия | Базовые определения |
| **Difficulty 3** | Алгоритм O(n²) | Модель с 3-5 переменными | Системные подходы (Agile) | Мультиканальная кампания |
| **Difficulty 5** | Research алгоритмы | Нобелевские теории | Теория сложности | Нейромаркетинг, big data |
| **PREREQUISITE пример** | Data Structure → Algorithm | Supply → Equilibrium | Structure → Delegation | Model → Barriers |
| **PARALLEL пример** | QuickSort ↔ MergeSort | Keynes ↔ Monetarism | Theory X ↔ Theory Y | PR ↔ Marketing |
| **Ключевые концепты** | Stack, Queue, Graph, O(n) | GDP, Inflation, Elasticity | KPI, Agile, Leadership | Audience, Content, Engagement |