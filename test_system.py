"""
Test Suite Entrypoint for Multimodal Audio-Visual Deepfake Detection & Localization.
Forwards execution to backend.tests.test_system.
"""

from backend.tests.test_system import (
    test_audio_detector,
    test_vision_detector,
    test_multimodal_detector,
    test_fastapi_endpoints,
    generate_synthetic_audio,
    generate_synthetic_video,
    TEST_AUDIO_PATH,
    TEST_VIDEO_PATH,
)

if __name__ == "__main__":
    import runpy
    runpy.run_module("backend.tests.test_system", run_name="__main__")
