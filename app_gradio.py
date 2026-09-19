"""
Gradio Web Interface for Unified Audio-Visual Deepfake Detection & Localization.
Integrates Wav2Vec2 and AV-Deepfake1M Computer Vision benchmarks.
"""

import gradio as gr
try:
    from ai_models.multimodal.multimodal_detector import UnifiedDeepfakeDetector
except ImportError:
    from multimodal_detector import UnifiedDeepfakeDetector

print("Loading Unified Deepfake Detector for Gradio UI...")
detector = UnifiedDeepfakeDetector()


def classify_media(video_obj, audio_obj, mode_selection, audio_thresh=0.85, visual_thresh=0.65):
    file_obj = video_obj if mode_selection == "Video Defect Detection" else audio_obj
    if not file_obj:
        return "Please upload an audio or video file.", {}, ""

    try:
        mode = "video" if mode_selection == "Video Defect Detection" else "audio"
        file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))

        is_video = detector.is_video_file(file_path)
        if mode == "audio" and is_video:
            return (
                "### ⚠️ Rejection\n\n**Video files are not permitted in Audio Defect Detection mode.** "
                "Please upload an audio file (WAV, MP3, FLAC, M4A, OGG) or switch to 'Video Defect Detection'.",
                {},
                "",
            )
        if mode == "video" and not is_video:
            return (
                "### ⚠️ Rejection\n\n**Audio files are not permitted in Video Defect Detection mode.** "
                "Please upload a video file (MP4, WebM, AVI, MOV).",
                {},
                "",
            )

        result = detector.predict(
            file_path,
            mode=mode,
            audio_threshold=float(audio_thresh),
            visual_threshold=float(visual_thresh),
        )

        media_type = result.get("media_type", "unknown").upper()
        verdict_str = result.get("overall_verdict", "real").upper()
        is_fake = result.get("is_fake", False)
        verdict_title = result.get("verdict_title", "")
        conf = result.get("overall_confidence", 0.0) * 100

        audio_res = result.get("audio_analysis")
        visual_res = result.get("visual_analysis")

        probs = {}
        forensic_md = "### Detailed Forensic Metrics:\n"

        if mode == "audio" and audio_res:
            if audio_res.get("probabilities"):
                probs["Audio: Authentic Voice"] = audio_res["probabilities"].get("real", 0.0)
                probs["Audio: Synthetic / Fake"] = audio_res["probabilities"].get("fake", 0.0)
            forensic_md += f"- **Acoustic Verdict:** {audio_res.get('prediction', 'N/A')}\n"
            if audio_res.get("bandwidth_analysis"):
                forensic_md += f"- **Narrowband (<4kHz):** {audio_res['bandwidth_analysis'].get('is_narrowband')}\n"
                forensic_md += f"- **High-Frequency Ratio:** {audio_res['bandwidth_analysis'].get('high_freq_ratio')}\n"
        elif mode == "video" and visual_res:
            if visual_res.get("probabilities"):
                probs["Visual: Authentic Video"] = visual_res["probabilities"].get("real", 0.0)
                probs["Visual: Facial Deepfake"] = visual_res["probabilities"].get("fake", 0.0)
            v_met = visual_res.get("forensic_metrics", {})
            forensic_md += (
                f"- **Visual Frames / Faces:** {visual_res.get('faces_detected', 0)} faces in {visual_res.get('frames_analyzed', 0)} frames\n"
                f"- **Boundary Artifact Score:** {v_met.get('boundary_artifact_score', 0.0) * 100:.1f}%\n"
                f"- **2D FFT Frequency Anomaly:** {v_met.get('fft_frequency_score', 0.0) * 100:.1f}%\n"
                f"- **Temporal Motion Flicker:** {v_met.get('temporal_flicker_score', 0.0) * 100:.1f}%\n"
            )

        verdict_icon = "🚨" if is_fake else "✅"
        summary = (
            f"## {verdict_icon} {verdict_title}\n\n"
            f"- **Detection Mode:** `{mode_selection}`\n"
            f"- **Verdict:** `{verdict_str}`\n"
            f"- **Confidence:** **{conf:.2f}%**\n"
            f"- **Media Container:** {media_type}\n"
            f"- **Duration:** {result.get('duration_seconds', 0.0):.2f}s\n"
            f"- **Inference Latency:** {result.get('inference_time_seconds', 0.0):.3f}s\n\n"
        )

        fake_segments = result.get("fake_segments", [])
        if fake_segments:
            summary += f"### ⏱️ Localized Tampering Intervals ({len(fake_segments)} detected):\n"
            for seg in fake_segments:
                summary += f"- `[{seg[0]}s -> {seg[1]}s]`\n"
        else:
            summary += "### ⏱️ Temporal Localization:\n- No tampering intervals detected.\n"

        return summary, probs, forensic_md

    except Exception as e:
        return f"### ⚠️ Error Analyzing Media\n\n**{str(e)}**", {}, ""


with gr.Blocks(title="Audio / Video Deepfake Detector") as demo:
    gr.Markdown("# 🛡️ Deepfake Defect Detection & Localization Platform")
    gr.Markdown(
        "Forensics powered by **Video Defect Detection** (AV-Deepfake1M facial, FFT frequency, and temporal flicker analysis) "
        "and **Audio Defect Detection** (Wav2Vec2 speech transformer forensics)."
    )

    with gr.Row():
        with gr.Column():
            mode_selector = gr.Radio(
                ["Video Defect Detection", "Audio Defect Detection"],
                value="Video Defect Detection",
                label="Select Detection Mode",
            )
            video_input = gr.Video(
                label="Upload & Preview Video (MP4, WebM, AVI, MOV)",
                visible=True,
            )
            audio_input = gr.Audio(
                label="Upload & Preview Audio (WAV, MP3, FLAC, M4A, OGG)",
                type="filepath",
                visible=False,
            )

            def update_mode_visibility(selected_mode):
                is_vid = selected_mode == "Video Defect Detection"
                return gr.update(visible=is_vid), gr.update(visible=not is_vid)

            mode_selector.change(
                fn=update_mode_visibility,
                inputs=[mode_selector],
                outputs=[video_input, audio_input],
            )

            with gr.Accordion("Advanced Calibration Thresholds", open=False):
                audio_slider = gr.Slider(
                    minimum=0.50, maximum=0.99, value=0.85, step=0.05, label="Audio Fake Threshold"
                )
                visual_slider = gr.Slider(
                    minimum=0.40, maximum=0.95, value=0.65, step=0.05, label="Visual Fake Threshold"
                )
            submit_btn = gr.Button("Run Defect Detection", variant="primary")

        with gr.Column():
            output_verdict = gr.Markdown(label="Verdict")
            output_probs = gr.Label(label="Confidence Distribution", num_top_classes=2)
            output_forensics = gr.Markdown(label="Forensic Metrics")

    submit_btn.click(
        fn=classify_media,
        inputs=[video_input, audio_input, mode_selector, audio_slider, visual_slider],
        outputs=[output_verdict, output_probs, output_forensics],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
