import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

JSON_FILE = "abstracts_texts_old.json"

# Загружаем JSON-файл
with open(JSON_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

def count_words(text):
    """ Подсчитывает количество слов в тексте """
    return len(re.findall(r'\b\w+\b', text))

def count_sentences(text):
    """ Подсчитывает количество предложений в тексте """
    return len(re.findall(r'[.!?]', text))

# Списки для анализа
abstract_words = []
text_words = []
abstract_sentences = []
text_sentences = []

# Заполняем списки
for entry in data:
    abs_words = count_words(entry["abstract"])
    txt_words = count_words(entry["text"])
    abs_sentences = count_sentences(entry["abstract"])
    txt_sentences = count_sentences(entry["text"])

    abstract_words.append(abs_words)
    text_words.append(txt_words)
    abstract_sentences.append(abs_sentences)
    text_sentences.append(txt_sentences)

# Вычисляем среднее соотношение (сколько слов в абстракте на слово в тексте)
word_ratio = np.array(abstract_words) / np.array(text_words)
sentence_ratio = np.array(abstract_sentences) / np.array(text_sentences)

mean_word_ratio = np.mean(word_ratio)
median_word_ratio = np.median(word_ratio)

mean_sentence_ratio = np.mean(sentence_ratio)
median_sentence_ratio = np.median(sentence_ratio)

print(f"📊 Среднее соотношение слов (abstract / text): {mean_word_ratio:.3f}")
print(f"📊 Медианное соотношение слов (abstract / text): {median_word_ratio:.3f}")
print(f"📊 Среднее соотношение предложений (abstract / text): {mean_sentence_ratio:.3f}")
print(f"📊 Медианное соотношение предложений (abstract / text): {median_sentence_ratio:.3f}")

# Визуализация данных
plt.figure(figsize=(12, 5))

# Гистограмма соотношения слов
plt.subplot(1, 2, 1)
sns.histplot(word_ratio, bins=30, kde=True, color="blue")
plt.axvline(mean_word_ratio, color="red", linestyle="dashed", label=f"Среднее: {mean_word_ratio:.3f}")
plt.axvline(median_word_ratio, color="green", linestyle="dashed", label=f"Медиана: {median_word_ratio:.3f}")
plt.xlabel("Соотношение слов (abstract / text)")
plt.ylabel("Частота")
plt.title("Распределение соотношения слов")
plt.legend()

# Гистограмма соотношения предложений
plt.subplot(1, 2, 2)
sns.histplot(sentence_ratio, bins=30, kde=True, color="orange")
plt.axvline(mean_sentence_ratio, color="red", linestyle="dashed", label=f"Среднее: {mean_sentence_ratio:.3f}")
plt.axvline(median_sentence_ratio, color="green", linestyle="dashed", label=f"Медиана: {median_sentence_ratio:.3f}")
plt.xlabel("Соотношение предложений (abstract / text)")
plt.ylabel("Частота")
plt.title("Распределение соотношения предложений")
plt.legend()

plt.tight_layout()
plt.show()
