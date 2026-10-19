import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("Задание 5: Поиск похожих треков на основе вектора признаков...")

input_file = "data/tracks_pipeline_output.jsonl"

tracks = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        tracks.append(json.loads(line))

vectors = np.array([t["feature_vector"] for t in tracks])

similarity_matrix = cosine_similarity(vectors)

target_index = 0
top_indices = np.argsort(similarity_matrix[target_index])[::-1][1:4]

print(f"\nИсходный трек [ID: {tracks[target_index]['ytid']}]:")
print(f"Описание: {tracks[target_index]['caption']}\n")

print("ТОП-3 самых похожих трека:")
for idx in top_indices:
    score = similarity_matrix[target_index][idx]
    print(f"- [Сходство: {score:.4f}] ID: {tracks[idx]['ytid']}")
    print(f"  Описание: {tracks[idx]['caption']}\n")
