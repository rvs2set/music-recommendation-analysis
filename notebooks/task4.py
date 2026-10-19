import json
import matplotlib.pyplot as plt

print("Задание 4: Проведение EDA и построение визуализаций...")

input_file = "data/tracks_with_features.jsonl"

tracks = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        tracks.append(json.loads(line))

feature_counts = {
    "Energetic": sum(1 for t in tracks if t["features"]["is_energetic"]),
    "Calm": sum(1 for t in tracks if t["features"]["is_calm"]),
    "Has Vocals": sum(1 for t in tracks if t["features"]["has_vocals"]),
    "Electronic": sum(1 for t in tracks if t["features"]["is_electronic"])
}

print("Распределение признаков по трекам:")
for feature, count in feature_counts.items():
    print(f"- {feature}: {count}")

plt.figure(figsize=(8, 5))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#c2c2f0']
plt.bar(feature_counts.keys(), feature_counts.values(), color=colors)
plt.title("Распределение семантических признаков в датасете")
plt.xlabel("Признаки")
plt.ylabel("Количество треков")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("data/features_distribution.png")
print("График с результатами сохранен в data/features_distribution.png")
