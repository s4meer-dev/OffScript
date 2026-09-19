from huggingface_hub import hf_hub_download
from detector import DeepfakeAudioDetector

d = DeepfakeAudioDetector()
repo = "garystafford/deepfake-audio-detection"

real_samples = [
    "real/yt_0000_part_001.flac",
    "real/yt_0000_part_002.flac",
    "real/yt_0000_part_003.flac",
    "real/yt_0000_part_004.flac",
    "real/yt_0000_p2_part_167.flac",
]

fake_samples = [
    "fake/el_0001_part_001.flac",
    "fake/el_0001_c_part_002.flac",
    "fake/el_0002_c_part_001.flac",
    "fake/el_0003_part_001.flac",
    "fake/el_0003_part_002.flac",
]

print("\n=== TESTING REAL SAMPLES ===")
for s in real_samples:
    p = hf_hub_download(repo_id=repo, filename=s, repo_type="dataset")
    res = d.predict(p)
    print(f"REAL sample {s} -> Predicted: {res['prediction']} (fake: {res['probabilities']['fake']*100:.1f}%, real: {res['probabilities']['real']*100:.1f}%)")

print("\n=== TESTING FAKE SAMPLES ===")
for s in fake_samples:
    p = hf_hub_download(repo_id=repo, filename=s, repo_type="dataset")
    res = d.predict(p)
    print(f"FAKE sample {s} -> Predicted: {res['prediction']} (fake: {res['probabilities']['fake']*100:.1f}%, real: {res['probabilities']['real']*100:.1f}%)")
