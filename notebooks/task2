import json

print("Задание 2: Обогащение треков семантическими признаками...")

input_file = "data/sample_tracks.jsonl"
output_file = "data/tracks_with_features.jsonl"

tracks = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        tracks.append(json.loads(line))

# Извлекаем семантические категории из описания
for track in tracks:
    caption = track.get("caption", "").lower()
    
    track["features"] = {
        "is_energetic": any(word in caption for word in ["fast", "energetic", "upbeat", "intense", "drum", "rock"]),
        "is_calm": any(word in caption for word in ["slow", "calm", "relaxing", "soft", "peaceful", "ambient"]),
        "has_vocals": any(word in caption for word in ["vocal", "singing", "voice", "female", "male", "sing"]),
        "is_electronic": any(word in caption for word in ["synth", "electronic", "synthesizer", "edm", "beat", "techno"])
    }

with open(output_file, "w", encoding="utf-8") as f:
    for track in tracks:
        f.write(json.dumps(track, ensure_ascii=False) + "\n")

print(f"Успешно обработано {len(tracks)} треков и сохранено в {output_file}")
