import json
from sentence_transformers import SentenceTransformer

print("1. Загружаем модель для извлечения текстовых признаков...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("2. Читаем датасет из 50 треков...")
input_file = "data/sample_tracks.jsonl"
output_file = "data/tracks_with_features.jsonl"

tracks = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        tracks.append(json.loads(line))

captions = [t.get("caption", "") for t in tracks]

print("3. Извлекаем семантические векторы признаков...")
embeddings = model.encode(captions)

for i, track in enumerate(tracks):
    track["feature_vector"] = embeddings[i].tolist()

print("4. Сохраняем обогащенные данные...")
with open(output_file, "w", encoding="utf-8") as f:
    for track in tracks:
        f.write(json.dumps(track, ensure_ascii=False) + "\n")

print(f"Успешно! Извлечены признаки для {len(tracks)} треков и сохранены в {output_file}")
