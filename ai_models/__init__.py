"""
AI Models Module for Multimodal Deepfake Detection & Localization.
Provides unified access to:
- DeepfakeAudioDetector (Speech Wav2Vec2 transformer)
- DeepfakeVisionDetector (Computer Vision, Facial, Spatial FFT & Temporal Flicker)
- UnifiedDeepfakeDetector (Multimodal fusion & AV-Deepfake1M taxonomy localization)
"""

from ai_models.audio.detector import DeepfakeAudioDetector
from ai_models.vision.vision_detector import DeepfakeVisionDetector
from ai_models.multimodal.multimodal_detector import UnifiedDeepfakeDetector

__all__ = [
    "DeepfakeAudioDetector",
    "DeepfakeVisionDetector",
    "UnifiedDeepfakeDetector",
]
