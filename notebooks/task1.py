import json
from datasets import load_dataset

print("Загружаем датасет с Hugging Face...")
try:
    dataset = load_dataset("google/MusicCaps", split="train", streaming=True)
    
    tracks = []
    for item in dataset:
        tracks.append({
            "ytid": item.get("ytid"),
            "start_s": item.get("start_s"),
            "end_s": item.get("end_s"),
            "caption": item.get("caption")
        })
        if len(tracks) >= 50:
            break

    with open("data/sample_tracks.jsonl", "w", encoding="utf-8") as f:
        for track in tracks:
            f.write(json.dumps(track, ensure_ascii=False) + "\n")

    print(f"Успешно сохранено {len(tracks)} треков в data/sample_tracks.jsonl")
except Exception as e:
    print(f"Ошибка: {e}")
