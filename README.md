# **Сравнительный анализ моделей для вопросно-ответного поиска**

## **Описание проекта**

Проект посвящен сравнительному анализу трех моделей для задачи вопросно-ответного поиска на русском языке:
- T5 (cointegrated/rut5-base-multitask)
- GPT2 (sberbank-ai/rugpt3small_based_on_gpt2)
- QA (Den4ikAI/rubert_large_squad_2)

## **Установка**

Для воспроизведения экспериментов необходимо установить следующие библиотеки:

```bash
pip install transformers torch datasets evaluate nltk matplotlib seaborn jupyter
```

## **Запуск**

1. Откройте Jupyter Notebook:

```bash
jupyter notebook
```

2. Запустите ноутбук `experiments.ipynb`.

3. При первом запуске потребуется ввести Hugging Face токен для доступа к датасету и моделям.

## **Структура проекта**

- `experiments.ipynb`: Jupyter Notebook с кодом и экспериментами
- `report.pdf`: PDF-отчёт с анализом результатов
- `README.md`: Инструкции по воспроизведению

## **Данные**

В проекте используется датасет [SberQuAD](https://huggingface.co/datasets/kuznetsoffandrey/sberquad) - русский аналог датасета SQuAD для вопросно-ответных систем. Данные загружаются автоматически при запуске ноутбука.

## **Требования**

1. Python 3.7+
2. Доступ в интернет для загрузки моделей и датасета
3. Hugging Face токен (можно получить на [huggingface.co](https://huggingface.co/))

## **Авторы**

[Ваше имя]
[Ваша контактная информация]