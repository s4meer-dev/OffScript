from detector import DeepfakeAudioDetector

d_v1 = DeepfakeAudioDetector(model_path_or_id="F:/DeepFake/model")
d_v2 = DeepfakeAudioDetector(model_path_or_id="MelodyMachine/Deepfake-audio-detection-V2")

for name in ["samples/real_human_voice.wav", "samples/ai_voice_clone.wav"]:
    r1 = d_v1.predict(name)
    r2 = d_v2.predict(name)
    print(f"\n--- Testing: {name} ---")
    print(f"V1: {r1['prediction']} (Fake: {r1['probabilities']['fake']*100:.1f}%, Real: {r1['probabilities']['real']*100:.1f}%)")
    print(f"V2: {r2['prediction']} (Fake: {r2['probabilities']['fake']*100:.1f}%, Real: {r2['probabilities']['real']*100:.1f}%)")
