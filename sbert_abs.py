import time
import json
import torch
import numpy as np
import pymorphy3
import re
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from bert_score import score as bert_score
from evaluate import load
import nltk
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

# Загружаем необходимые данные
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Загружаем nltk punkt...")
    nltk.download('punkt', quiet=True)

morph = pymorphy3.MorphAnalyzer()

# Загружаем модель FRED-T5-Summarizer
MODEL_NAME = "RussianNLP/FRED-T5-Summarizer"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, eos_token='</s>')
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Загружаем SBERT для экстрактивной суммаризации
sbert_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")


# Функция экстрактивной суммаризации с учетом важности начала и конца статьи
def sbert_extractive_summarization(text: str, summary_ratio: float = 0.4) -> str:
    sentences = sent_tokenize(text)
    if len(sentences) <= 2:
        return text

    # Кодируем предложения с помощью SBERT
    sentence_embeddings = sbert_model.encode(sentences, convert_to_numpy=True)

    # Вычисляем попарное косинусное сходство
    similarity_matrix = cosine_similarity(sentence_embeddings)

    # Вычисляем среднее значение сходства
    scores = similarity_matrix.mean(axis=1)

    # **Добавляем вес начальным и конечным предложениям**
    position_weights = np.array(
        [1.2 if i < len(sentences) * 0.2 or i > len(sentences) * 0.8 else 1.0 for i in range(len(sentences))])
    scores = scores * position_weights  # Усиливаем значимость начальных и конечных предложений

    # Выбираем `summary_ratio` самых значимых предложений
    num_summary_sentences = max(1, int(len(sentences) * summary_ratio))
    top_sentence_indices = np.argsort(scores)[-num_summary_sentences:]  # Берем N самых значимых предложений
    top_sentence_indices.sort()  # Восстанавливаем порядок

    return " ".join(sentences[i] for i in top_sentence_indices)


# **Обновленная функция абстрактивной суммаризации с научным стилем**
def abstractive_summarize(text: str) -> str:
    input_text = f"<LM> Напиши аннотацию к научной статье.\n В данной статье рассматривается {text}"
    input_ids = torch.tensor([tokenizer.encode(input_text)]).to(device)

    outputs = model.generate(
        input_ids,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=7,
        min_new_tokens=120,
        max_new_tokens=750,
        do_sample=True,
        temperature=0.65,
        no_repeat_ngram_size=4,
        top_p=0.9
    )

    return tokenizer.decode(outputs[0][1:])


# Функция суммаризации и оценки
def extractive_and_abstractive_summarization(json_file: str, summary_ratio: float = 0.35):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Файл {json_file} не найден.")
        return
    except json.JSONDecodeError:
        print("[ERROR] Ошибка декодирования JSON.")
        return

    # Выбираем 20 случайных записей
    random.seed(42)
    data = random.sample(data, min(50, len(data)))

    results_data = []
    rouge_metric = Rouge()
    bleurt_metric = load("bleurt", "bleurt-20")
    references, predictions = [], []

    for entry in tqdm(data, desc="Summarizing"):
        text, reference = entry.get("text", ""), entry.get("abstract", "")
        if not text.strip() or not reference.strip():
            continue

        # **Экстрактивная суммаризация с SBERT и весами**
        extractive_summary = sbert_extractive_summarization(text, summary_ratio=summary_ratio)
        if not extractive_summary.strip():
            extractive_summary = text

        # **Абстрактивная суммаризация (в научном стиле)**
        abstractive_summary = abstractive_summarize(extractive_summary)

        references.append(reference)
        predictions.append(abstractive_summary)

        # **Оценка метрик**
        rouge_scores = rouge_metric.get_scores(abstractive_summary, reference, avg=True)
        bleurt_results = bleurt_metric.compute(references=[reference], predictions=[abstractive_summary])
        P, R, F1 = bert_score([abstractive_summary], [reference], lang="ru")
        bertscore_f1 = torch.mean(F1).item()

        # **Сохранение результатов**
        results_data.append({
            "Полный текст": text,
            "Оригинальный абстракт": reference,
            "Экстрактивная суммаризация": extractive_summary,
            "Абстрактивная суммаризация": abstractive_summary,
            "ROUGE-1 F1": rouge_scores['rouge-1']['f'],
            "ROUGE-2 F1": rouge_scores['rouge-2']['f'],
            "ROUGE-L F1": rouge_scores['rouge-l']['f'],
            "BLEURT": bleurt_results["scores"][0],
            "BERTScore F1": bertscore_f1
        })

    df = pd.DataFrame(results_data)
    df.to_excel("summarization_results.xlsx", index=False)
    print("[INFO] Результаты сохранены в summarization_results.xlsx")


# **Запуск**
if __name__ == "__main__":
    start_time = time.time()
    extractive_and_abstractive_summarization("abstracts_texts_old.json", summary_ratio=0.1)
    end_time = time.time()
    print(f"\nСуммаризация завершена за {end_time - start_time:.2f} секунд(ы).")
