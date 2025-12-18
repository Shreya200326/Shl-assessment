import os
import re
import numpy as np
import pandas as pd
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from sklearn.ensemble import RandomForestRegressor

BASE_PATH = "/kaggle/input/shl-intern-hiring-assessment-2025/dataset"

train_df = pd.read_csv(os.path.join(BASE_PATH, "csvs", "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_PATH, "csvs", "test.csv"))

# load speech to text model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")nasr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
asr_model.eval()

# find correct audio file for an id
def find_audio_file(audio_id, split):
    audio_dir = os.path.join(BASE_PATH, "audios", split)
    for f in os.listdir(audio_dir):
        if f.startswith(audio_id):
            return os.path.join(audio_dir, f)
    return None

# convert speech to text
def convert_audio_to_text(audio_path):
    signal, sr = librosa.load(audio_path, sr=16000, mono=True)
    inputs = processor(signal, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = asr_model(inputs.input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0].lower()

# extract simple text features
def extract_features(text):
    words = text.split()
    sentences = re.split(r"[.!?]", text)
    return {
        "word_count": len(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "sentence_count": len([s for s in sentences if s.strip()]),
        "short_word_ratio": len([w for w in words if len(w) <= 3]) / max(len(words), 1)
    }

# use a subset to keep runtime reasonable
train_subset = train_df.sample(120, random_state=42).reset_index(drop=True)

train_features = []
for _, row in train_subset.iterrows():
    audio_path = find_audio_file(row["filename"], "train")
    text = convert_audio_to_text(audio_path)
    train_features.append(extract_features(text))

X_train = pd.DataFrame(train_features)
y_train = train_subset["label"]

model = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# build test features
test_features = []
for fname in test_df["filename"]:
    audio_path = find_audio_file(fname, "test")
    text = convert_audio_to_text(audio_path)
    test_features.append(extract_features(text))

X_test = pd.DataFrame(test_features)
predictions = model.predict(X_test)
predictions = np.clip(predictions, 0, 5)

submission = pd.DataFrame({
    "filename": test_df["filename"],
    "label": predictions
})

submission.to_csv("submission.csv", index=False)
