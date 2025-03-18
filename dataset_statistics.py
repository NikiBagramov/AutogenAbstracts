# Улучшенный график с двумя подграфиками (subplots)

import json
import matplotlib.pyplot as plt

# Загружаем данные
file_path = "abstracts_texts.json"
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Функция для подсчета количества слов в тексте
def count_words(text):
    return len(text.split())

# Подсчет статистики
abstract_lengths = [count_words(item['abstract']) for item in data]
text_lengths = [count_words(item['text']) for item in data]

# Создание двух подграфиков на одном полотне
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

# Гистограмма аннотаций
axes[0].hist(abstract_lengths, bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_xlabel("Количество слов в аннотациях")
axes[0].set_ylabel("Частота")
axes[0].set_title("Распределение количества слов в аннотациях")
axes[0].grid(True)

# Гистограмма полных текстов
axes[1].hist(text_lengths, bins=30, alpha=0.7, color='orange', edgecolor='black')
axes[1].set_xlabel("Количество слов в полных текстах")
axes[1].set_title("Распределение количества слов в полных текстах")
axes[1].grid(True)

# Улучшаем компоновку и отображаем график
plt.tight_layout()
plt.show()
