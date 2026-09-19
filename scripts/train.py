import os
import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from datasets import Dataset

# Constants
MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
REAL_DIR = Path("dataset/real")
FAKE_DIR = Path("dataset/fake")
OUTPUT_DIR = Path("ai_models/weights_v3")

def load_audio_files(directory, label, max_duration=10.0, sr=16000):
    data = []
    if not directory.exists():
        return data
        
    for file in directory.glob("*.*"):
        if file.suffix.lower() not in ['.wav', '.mp3', '.flac']:
            continue
            
        try:
            # Load and resample
            audio, _ = librosa.load(str(file), sr=sr, mono=True)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=30)
            
            # Create overlapping chunks to increase dataset size from 8 files to many chunks
            chunk_length = int(4.0 * sr)
            hop_length = int(2.0 * sr)
            
            if len(audio) < chunk_length:
                # Pad if too short
                pad_len = chunk_length - len(audio)
                audio = np.pad(audio, (0, pad_len), mode='constant')
                data.append({"audio": audio, "label": label, "file": file.name})
            else:
                # Sliding window
                for i in range(0, len(audio) - chunk_length + 1, hop_length):
                    chunk = audio[i:i + chunk_length]
                    data.append({"audio": chunk, "label": label, "file": f"{file.name}_{i}"})
                    
        except Exception as e:
            print(f"Error loading {file}: {e}")
            
    return data

def main():
    print("Loading feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    
    print(f"Loading files from {REAL_DIR} and {FAKE_DIR}...")
    # Label 1 = REAL, 0 = FAKE (Standardizing to fix the label inversion issue)
    real_data = load_audio_files(REAL_DIR, 1, sr=feature_extractor.sampling_rate)
    fake_data = load_audio_files(FAKE_DIR, 0, sr=feature_extractor.sampling_rate)
    
    all_data = real_data + fake_data
    if not all_data:
        print("No audio files found for training.")
        return
        
    print(f"Created {len(all_data)} chunks for training from the original files.")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(all_data)
    
    # Preprocessing function
    def preprocess_function(examples):
        inputs = feature_extractor(
            examples["audio"], 
            sampling_rate=feature_extractor.sampling_rate, 
            max_length=int(feature_extractor.sampling_rate * 4.0), 
            truncation=True,
            padding="max_length"
        )
        inputs["labels"] = examples["label"]
        return inputs

    print("Tokenizing and extracting features...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["audio", "file"])
    
    # Shuffle and split (using 90% train, 10% eval just to track loss)
    tokenized_dataset = tokenized_dataset.shuffle(seed=42)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    print("Loading base model...")
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_ID, 
        num_labels=2, 
        ignore_mismatched_sizes=True,
        id2label={"0": "FAKE", "1": "REAL"},
        label2id={"FAKE": 0, "REAL": 1}
    )
    
    # Freeze the wav2vec2 feature extractor so we don't overfit to the acoustic environment of the 8 files
    # We only fine-tune the classification head and the top transformer layers
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        warmup_steps=5,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
    )
    
    print("Starting fine-tuning (Phase 11)...")
    trainer.train()
    
    print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    feature_extractor.save_pretrained(str(OUTPUT_DIR))
    print("Training complete!")

if __name__ == "__main__":
    main()
