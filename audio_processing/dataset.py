import os


def load_metadata(prompts_path):
    metadata = {}

    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            audio_id, text = parts
            metadata[audio_id] = text

    return metadata


def find_audio_files(waves_root):
    audio_map = {}

    for root, _, files in os.walk(waves_root):
        for file in files:
            if file.endswith(".wav"):
                audio_id = file.replace(".wav", "")
                full_path = os.path.join(root, file)
                audio_map[audio_id] = full_path

    return audio_map


def build_dataset(root_path, split="train"):
    split_path = os.path.join(root_path, split)

    waves_root = os.path.join(split_path, "waves")
    prompts_path = os.path.join(split_path, "prompts.txt")

    metadata = load_metadata(prompts_path)
    audio_map = find_audio_files(waves_root)

    dataset = []

    for audio_id, text in metadata.items():
        if audio_id in audio_map:
            dataset.append({"audio": audio_map[audio_id], "text": text})

    return dataset


if __name__ == "__main__":
    data = build_dataset("../data/vivos", "train")
    print("Total samples:", len(data))
    print(data[0])
