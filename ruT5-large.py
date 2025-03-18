import time
import json
import torch
import pymorphy3
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge
from bert_score import score as bert_score
from evaluate import load
from tqdm import tqdm

# Инициализация лемматизатора pymorphy3
morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text):
    """
    Лемматизирует текст для улучшения сравнения в ROUGE.
    Приводит к нижнему регистру и удаляет знаки препинания.
    """
    words = re.findall(r'\b\w+\b', text.lower())  # Приведение к lowercase + токенизация по словам
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]  # Лемматизация
    return " ".join(lemmatized_words)

def summarize_and_evaluate(json_file: str,
                           model_name: str = "sberbank-ai/ruT5-large",
                           max_source_length: int = 512,
                           max_summary_length: int = 128):
    """
    Функция выполняет суммаризацию текста и сравнивает с референсами,
    используя ROUGE (с лемматизацией и lowercase), BLEURT и BERTScore.
    """

    # Загружаем токенайзер и модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Определяем устройство (GPU или CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Считываем данные из JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Инициализация метрик
    rouge_metric = Rouge()
    bleurt_metric = load("bleurt", "bleurt-20")

    references = []
    predictions = []

    # Префикс для суммаризации в ruT5
    summarize_prefix = "summarize: "

    # Проходимся по каждому объекту из JSON
    for entry in tqdm(data[:20], desc="Summarizing"):
        text = entry["text"]
        reference = entry["abstract"]

        # Добавляем префикс "summarize: " для ruT5
        prefixed_text = summarize_prefix + text

        # Токенизация входного текста
        inputs = tokenizer(
            prefixed_text,
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # Генерация суммаризации
        summary_ids = model.generate(
            **inputs,
            max_length=max_summary_length,
            num_beams=4,
            no_repeat_ngram_size=2
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        references.append(reference)
        predictions.append(summary)

    # Лемматизируем тексты перед вычислением ROUGE
    lemmatized_references = [lemmatize_text(ref) for ref in references]
    lemmatized_predictions = [lemmatize_text(pred) for pred in predictions]

    # Подсчёт метрик

    # 1. ROUGE (сравнение лемматизированных версий)
    rouge_scores = rouge_metric.get_scores(lemmatized_predictions, lemmatized_references, avg=True)

    # 2. BLEURT
    bleurt_results = bleurt_metric.compute(references=references, predictions=predictions)
    bleurt_score_mean = sum(bleurt_results["scores"]) / len(bleurt_results["scores"])

    # 3. BERTScore (оригинальные тексты, так как он работает с эмбеддингами)
    P, R, F1 = bert_score(predictions, references, lang="ru")
    bertscore_f1 = torch.mean(F1).item()

    # Вывод результатов
    print("==== Результаты метрик ====")
    print("ROUGE (avg):", rouge_scores)
    print(f"BLEURT (mean): {bleurt_score_mean:.4f}")
    print(f"BERTScore F1 (mean): {bertscore_f1:.4f}")

    return {
        "references": references,
        "predictions": predictions,
        "rouge_scores": rouge_scores,
        "bleurt_scores": bleurt_results["scores"],
        "bleurt_score_mean": bleurt_score_mean,
        "bertscore_f1": bertscore_f1
    }

if __name__ == "__main__":
    start_time = time.time()
    results = summarize_and_evaluate("abstracts_texts_old.json")
    end_time = time.time()
    print(f"\nСуммаризация завершена за {end_time - start_time:.2f} секунд(ы).")
