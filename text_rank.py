import time
import json
import torch
import numpy as np
import pymorphy3
import re
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from bert_score import score as bert_score
from evaluate import load

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Загружаем nltk punkt...")
    nltk.download('punkt', quiet=True)

morph = pymorphy3.MorphAnalyzer()


def text_rank(text: str, summary_ratio: float = 0.2, damping: float = 0.85, convergence_threshold: float = 1e-4) -> str:
    sentences = sent_tokenize(text)
    if not sentences:
        return ""
    if len(sentences) <= 2:
        return text

    vectorizer = TfidfVectorizer(stop_words="russian")
    try:
        sentence_vectors = vectorizer.fit_transform(sentences)
    except ValueError:
        return text

    similarity_matrix = cosine_similarity(sentence_vectors)
    num_sentences = len(sentences)
    scores = np.ones(num_sentences)
    prev_scores = np.zeros(num_sentences)
    max_iterations = 100
    iteration = 0

    while np.linalg.norm(scores - prev_scores) > convergence_threshold and iteration < max_iterations:
        prev_scores = scores.copy()
        for i in range(num_sentences):
            summation = sum(similarity_matrix[i, j] * scores[j] for j in range(num_sentences) if i != j)
            scores[i] = (1 - damping) + damping * summation
        iteration += 1

    scores /= np.max(scores)

    ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
    num_summary_sentences = max(1, int(len(sentences) * summary_ratio))

    step = max(1, len(ranked_sentences) // num_summary_sentences)
    selected_sentences = sorted(ranked_sentences[::step], key=lambda x: x[2])

    return " ".join([s for (_, s, _) in selected_sentences])


def extractive_summarize_and_evaluate(json_file: str, summary_ratio: float = 0.2):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("[ERROR] Ошибка открытия JSON файла.")
        return

    rouge_metric = Rouge()
    bleurt_metric = load("bleurt", "bleurt-20")

    all_rouge1, all_rouge2, all_rougeL = [], [], []
    all_bleurt, all_bertscore = [], []

    for entry in tqdm(data[:120], desc="Extractive Summarizing"):
        text, reference = entry.get("text", ""), entry.get("abstract", "")
        if not text.strip() or not reference.strip():
            continue

        summary = text_rank(text, summary_ratio=summary_ratio)
        if not summary.strip():
            summary = text

        rouge_scores = rouge_metric.get_scores(summary, reference, avg=True)
        bleurt_results = bleurt_metric.compute(references=[reference], predictions=[summary])
        _, _, F1 = bert_score([summary], [reference], lang="ru")
        bertscore_f1 = torch.mean(F1).item()

        all_rouge1.append(rouge_scores['rouge-1']['f'])
        all_rouge2.append(rouge_scores['rouge-2']['f'])
        all_rougeL.append(rouge_scores['rouge-l']['f'])
        all_bleurt.append(bleurt_results["scores"][0])
        all_bertscore.append(bertscore_f1)

    print("\n=== Средние метрики ===")
    print(f"ROUGE-1 F1: {np.mean(all_rouge1):.3f}")
    print(f"ROUGE-2 F1: {np.mean(all_rouge2):.3f}")
    print(f"ROUGE-L F1: {np.mean(all_rougeL):.3f}")
    print(f"BLEURT: {np.mean(all_bleurt):.3f}")
    print(f"BERTScore F1: {np.mean(all_bertscore):.3f}")


if __name__ == "__main__":
    start_time = time.time()
    extractive_summarize_and_evaluate("abstracts_texts_old.json", summary_ratio=0.1)
    end_time = time.time()
    print(f"\nСуммаризация завершена за {end_time - start_time:.2f} секунд(ы).")
