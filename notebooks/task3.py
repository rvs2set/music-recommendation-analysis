import json
from sentence_transformers import SentenceTransformer

print("Задание 3: Запуск пайплайна векторных эмбеддингов...")

# Инициализируем модель извлечения признаков
model = SentenceTransformer('all-MiniLM-L6-v2')

input_file = "data/tracks_with_features.jsonl"
output_file = "data/tracks_pipeline_output.jsonl"

tracks = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        tracks.append(json.loads(line))

# Формируем список текстов для модели
captions = [t.get("caption", "") for t in tracks]

# Генерируем векторы с батчингом
print("Генерация вектора признаков для каждого трека...")
embeddings = model.encode(captions, batch_size=32, show_progress_bar=True)

# Добавляем вектор признаков в данные трека
for i, track in enumerate(tracks):
    track["feature_vector"] = embeddings[i].tolist()

with open(output_file, "w", encoding="utf-8") as f:
    for track in tracks:
        f.write(json.dumps(track, ensure_ascii=False) + "\n")

print(f"Пайплайн завершен. Результаты сохранены в {output_file}")
