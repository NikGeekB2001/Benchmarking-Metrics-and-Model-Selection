# **Сравнительный анализ моделей для вопросно-ответного поиска на русском языке**

## **Установка библиотек**

```python
# Раскомментируйте при первом запуске
# !pip install -q transformers torch datasets evaluate nltk matplotlib seaborn
```

## **Импорт библиотек и настройка**

```python
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoModelForQuestionAnswering, AutoTokenizer
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
import os
from getpass import getpass

# Загрузка необходимых данных NLTK
nltk.download('punkt', quiet=True)

# Безопасный ввод токена
HF_API_TOKEN = getpass("Введите ваш Hugging Face токен: ")
os.environ["HF_TOKEN"] = HF_API_TOKEN

# Кэш для моделей и токенизаторов
model_cache = {}
```

## **Загрузка и подготовка данных**

```python
try:
    # Загружаем датасет SberQuAD
    dataset = load_dataset("kuznetsoffandrey/sberquad", token=HF_API_TOKEN)
    print(f"✅ Датасет SberQuAD загружен! Размер: {len(dataset['validation'])} примеров")

    # Выбираем 50 случайных примеров для тестирования
    num_examples = 50
    test_data = dataset["validation"].shuffle(seed=42).select(range(num_examples))
    print(f"✅ Подготовлено {len(test_data)} случайных примеров для тестирования.")
except Exception as e:
    print(f"❌ Ошибка при загрузке датасета: {e}")
    test_data = None
```

## **Функции для оценки метрик**

```python
def simple_tokenize(text):
    """Простая токенизация по пробелам."""
    return text.split()

def compute_exact_match(prediction, reference):
    """Точное совпадение (Exact Match)."""
    return int(prediction.strip() == reference.strip())

def compute_f1(prediction, reference):
    """Расчёт F1 Score на уровне слов."""
    pred_tokens = set(simple_tokenize(prediction.lower()))
    ref_tokens = set(simple_tokenize(reference.lower()))

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0

    common_tokens = pred_tokens.intersection(ref_tokens)
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def compute_bleu(prediction, reference):
    """Расчёт BLEU Score с использованием NLTK."""
    smoothing = SmoothingFunction().method1
    pred_tokens = simple_tokenize(prediction.lower())
    ref_tokens = [simple_tokenize(reference.lower())]
    return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
```

## **Функции для работы с моделями**

```python
def get_model_and_tokenizer(model_name, model_type):
    """Загружает модель и токенизатор по имени."""
    if model_name in model_cache:
        return model_cache[model_name]

    try:
        if model_type == "T5":
            tokenizer = T5Tokenizer.from_pretrained(model_name, token=HF_API_TOKEN, legacy=False)
            model = T5ForConditionalGeneration.from_pretrained(model_name, token=HF_API_TOKEN)
        elif model_type == "GPT2":
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, token=HF_API_TOKEN)
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained(model_name, token=HF_API_TOKEN)
        elif model_type == "QA":
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_TOKEN)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name, token=HF_API_TOKEN)
        else:
            print(f"❌ Неизвестный тип модели: {model_type}")
            return None, None

        model_cache[model_name] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели {model_name}: {e}")
        return None, None
```

## **Функции для тестирования моделей**

```python
def test_t5_model_with_params(model_name, test_data, prompt_format="question: {question} context: {context}",
                             temperature=1.0, max_length=50):
    """Тестирует T5-модель с заданными параметрами."""
    try:
        model, tokenizer = get_model_and_tokenizer(model_name, "T5")
        if model is None or tokenizer is None:
            return []

        results = []
        for sample in test_data:
            question = sample['question']
            context = sample['context']
            reference = sample['answers']['text'][0]
            start_time = time.time()

            input_text = prompt_format.format(question=question, context=context) if not callable(prompt_format) else prompt_format(sample)
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True if temperature != 1.0 else False
                )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            elapsed_time = time.time() - start_time

            results.append({
                'question': question,
                'context': context,
                'reference': reference,
                'answer': answer,
                'elapsed_time': elapsed_time
            })

        return results
    except Exception as e:
        print(f"❌ Ошибка в test_t5_model_with_params: {e}")
        return []

def test_gpt2_model_with_params(model_name, test_data, prompt_format="Контекст: {context}\nВопрос: {question}\nОтвет:",
                               temperature=1.0, max_new_tokens=14):
    """Тестирует GPT2-модель."""
    try:
        model, tokenizer = get_model_and_tokenizer(model_name, "GPT2")
        if model is None or tokenizer is None:
            return []

        results = []
        for sample in test_data:
            question = sample['question']
            context = sample['context']
            reference = sample['answers']['text'][0]
            start_time = time.time()

            input_text = prompt_format.format(question=question, context=context) if not callable(prompt_format) else prompt_format(sample)
            inputs = tokenizer(input_text, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    do_sample=True if temperature != 1.0 else False,
                )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
            elapsed_time = time.time() - start_time

            results.append({
                'question': question,
                'context': context,
                'reference': reference,
                'answer': answer,
                'elapsed_time': elapsed_time
            })

        return results
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return []

def test_qa_model(model_name, test_data):
    """Тестирует модель для вопросно-ответных систем (QA)."""
    try:
        model, tokenizer = get_model_and_tokenizer(model_name, "QA")
        if model is None or tokenizer is None:
            return []

        results = []
        for sample in test_data:
            question = sample['question']
            context = sample['context']
            reference = sample['answers']['text'][0]
            start_time = time.time()

            inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = model(**inputs)

            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer_ids = inputs["input_ids"][0][answer_start:answer_end]
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
            elapsed_time = time.time() - start_time

            results.append({
                'question': question,
                'context': context,
                'reference': reference,
                'answer': answer,
                'elapsed_time': elapsed_time
            })

        return results
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return []
```

## **Функции для оценки и визуализации**

```python
def evaluate_metrics(results):
    """Оценивает все метрики по списку результатов."""
    metrics = []
    for result in results:
        prediction = result['answer']
        reference = result['reference']
        elapsed_time = result['elapsed_time']

        metric = {
            "exact_match": compute_exact_match(prediction, reference),
            "f1": compute_f1(prediction, reference),
            "bleu": compute_bleu(prediction, reference),
            "generation_time": elapsed_time,
            "length": len(prediction.split())
        }

        metrics.append(metric)

    return metrics

def print_average_metrics(metrics, title=""):
    """Выводит средние значения метрик."""
    if not metrics:
        print(f"{title} ❌ Нет данных.")
        return None

    avg = {
        "exact_match": np.mean([m['exact_match'] for m in metrics]),
        "f1": np.mean([m['f1'] for m in metrics]),
        "bleu": np.mean([m['bleu'] for m in metrics]),
        "generation_time": np.mean([m['generation_time'] for m in metrics]),
        "length": np.mean([m['length'] for m in metrics])
    }

    print(f"{title} Средние метрики:")
    print(f"  EM: {avg['exact_match']:.4f}")
    print(f"  F1: {avg['f1']:.4f}")
    print(f"  BLEU: {avg['bleu']:.4f}")
    print(f"  Время: {avg['generation_time']:.2f} с")
    print(f"  Длина: {avg['length']:.2f} слов\n")

    return avg

def visualize_all_temperature_results(all_results):
    """График зависимости метрик от температуры."""
    metrics = ['exact_match', 'f1', 'bleu', 'generation_time', 'length']

    for metric in metrics:
        plt.figure(figsize=(10, 5))

        for model, results in all_results.items():
            if model == "QA":
                value = results.get("default", {}).get(metric, 0)
                plt.plot([0.1, 2.0], [value]*2, 'o--', label=f"{model}")
                continue

            temp_points = []
            for key, res in results.items():
                if key.startswith("T="):
                    try:
                        temp = float(key.split("=", 1)[1])
                        value = res.get(metric, 0)
                        temp_points.append((temp, value))
                    except (ValueError, IndexError):
                        continue

            if temp_points:
                temp_points.sort(key=lambda x: x[0])
                temps, values = zip(*temp_points)
                plt.plot(temps, values, 'o-', label=model)

        plt.title(f'{metric.upper()} vs Температура')
        plt.xlabel('Температура')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
```

## **Эксперименты**

```python
def experiment_temperature(model_name, model_type, test_data):
    """Эксперимент: влияние температуры на качество генерации."""
    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
    results = {}

    for temp in temperatures:
        print(f"🌡️ Тестирование температуры {temp}...")
        if model_type == "T5":
            res = test_t5_model_with_params(model_name, test_data, temperature=temp)
        elif model_type == "GPT2":
            res = test_gpt2_model_with_params(model_name, test_data, temperature=temp)
        else:
            res = []

        metrics = evaluate_metrics(res)
        results[f"T={temp}"] = print_average_metrics(metrics, f"T={temp}: ")

    return results

def analyze_errors(results):
    """Собирает примеры ошибок."""
    return [
        {
            "question": r['question'],
            "reference": r['reference'],
            "prediction": r['answer']
        }
        for r in results if r['answer'].strip() != r['reference'].strip()
    ]
```

## **Основной блок тестирования**

```python
if test_data:
    all_results = {}

    # Тестирование T5
    print("🚀 Тестирование T5")
    t5_results = experiment_temperature("cointegrated/rut5-base-multitask", "T5", test_data)
    all_results["T5"] = t5_results

    # Тестирование GPT2
    print("🚀 Тестирование GPT2")
    gpt2_results = experiment_temperature("sberbank-ai/rugpt3small_based_on_gpt2", "GPT2", test_data)
    all_results["GPT2"] = gpt2_results

    # Тестирование QA
    print("🚀 Тестирование QA")
    qa_results = test_qa_model("Den4ikAI/rubert_large_squad_2", test_data)
    qa_metrics = evaluate_metrics(qa_results)
    qa_avg = print_average_metrics(qa_metrics, "QA: ")
    all_results["QA"] = {"default": qa_avg}

    # Визуализация
    visualize_all_temperature_results(all_results)

    # Анализ ошибок
    print("\nАнализ типичных ошибок для всех моделей\n")

    models_to_test = [
        ("cointegrated/rut5-base-multitask", "T5", "T5"),
        ("sberbank-ai/rugpt3small_based_on_gpt2", "GPT2", "GPT2"),
        ("Den4ikAI/rubert_large_squad_2", "QA", "QA")
    ]

    for model_name, model_type, display_name in models_to_test:
        try:
            print(f"🟢 Модель: {display_name} ({model_name})")

            if model_type == "T5":
                examples = test_t5_model_with_params(model_name, test_data, temperature=0.5)
            elif model_type == "GPT2":
                examples = test_gpt2_model_with_params(model_name, test_data, temperature=0.5)
            elif model_type == "QA":
                examples = test_qa_model(model_name, test_data)

            errors = analyze_errors(examples)

            if errors:
                for err in errors[:3]:
                    print(f"  Вопрос: {err['question']}")
                    print(f"    Ответ: {err['prediction']}")
                    print(f"    Эталон: {err['reference']}\n")
            else:
                print("  Нет ошибок (все ответы совпали с эталоном)\n")
        except Exception as e:
            print(f" Ошибка при тестировании модели: {e}\n")
else:
    print("❌ Нет данных для тестирования. Проверьте загрузку датасета.")
```