"""
Gradio Web Interface for Unified Audio-Visual Deepfake Detection & Localization.
Integrates Wav2Vec2 and AV-Deepfake1M Computer Vision benchmarks.
"""

import gradio as gr
from multimodal_detector import UnifiedDeepfakeDetector

print("Loading Unified Deepfake Detector for Gradio UI...")
detector = UnifiedDeepfakeDetector()


def classify_media(file_obj, audio_thresh=0.85, visual_thresh=0.65):
    if not file_obj:
        return "Please upload an audio or video file.", {}, ""

    try:
        file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))
        result = detector.predict(
            file_path,
            audio_threshold=float(audio_thresh),
            visual_threshold=float(visual_thresh),
        )

        media_type = result.get("media_type", "unknown").upper()
        av_class = result.get("overall_verdict", "real").upper()
        is_fake = result.get("is_fake", False)
        verdict_title = result.get("verdict_title", "")
        conf = result.get("overall_confidence", 0.0) * 100

        audio_res = result.get("audio_analysis", {})
        visual_res = result.get("visual_analysis", {})

        probs = {}
        if audio_res.get("probabilities"):
            probs["Audio: Authentic Voice"] = audio_res["probabilities"]["real"]
            probs["Audio: Synthetic / Fake"] = audio_res["probabilities"]["fake"]
        if visual_res.get("probabilities"):
            probs["Visual: Authentic Video"] = visual_res["probabilities"]["real"]
            probs["Visual: Facial Deepfake"] = visual_res["probabilities"]["fake"]

        verdict_icon = "🚨" if is_fake else "✅"
        summary = (
            f"## {verdict_icon} {verdict_title}\n\n"
            f"- **AV-Deepfake1M Class:** `{av_class}`\n"
            f"- **Overall Confidence:** **{conf:.2f}%**\n"
            f"- **Media Format:** {media_type}\n"
            f"- **Total Duration:** {result.get('duration_seconds', 0.0):.2f}s\n"
            f"- **Inference Latency:** {result.get('inference_time_seconds', 0.0):.3f}s\n\n"
        )

        fake_segments = result.get("fake_segments", [])
        if fake_segments:
            summary += f"### ⏱️ Localized Tampering Intervals ({len(fake_segments)} detected):\n"
            for seg in fake_segments:
                summary += f"- `[{seg[0]}s -> {seg[1]}s]`\n"
        else:
            summary += "### ⏱️ Temporal Localization:\n- No tampering intervals detected.\n"

        # Forensic details
        forensic_md = "### Detailed Forensic Metrics:\n"
        if audio_res.get("status") != "no_audio_stream":
            forensic_md += f"- **Acoustic Verdict:** {audio_res.get('prediction', 'N/A')}\n"
            if audio_res.get("bandwidth_analysis"):
                forensic_md += f"- **Narrowband (<4kHz):** {audio_res['bandwidth_analysis'].get('is_narrowband')}\n"
        if visual_res.get("status") != "skipped":
            v_met = visual_res.get("forensic_metrics", {})
            forensic_md += (
                f"- **Visual Frames / Faces:** {visual_res.get('faces_detected', 0)} faces in {visual_res.get('frames_analyzed', 0)} frames\n"
                f"- **Boundary Artifact Score:** {v_met.get('boundary_artifact_score', 0.0) * 100:.1f}%\n"
                f"- **2D FFT Frequency Anomaly:** {v_met.get('fft_frequency_score', 0.0) * 100:.1f}%\n"
                f"- **Temporal Motion Flicker:** {v_met.get('temporal_flicker_score', 0.0) * 100:.1f}%\n"
            )

        return summary, probs, forensic_md

    except Exception as e:
        return f"Error analyzing media: {str(e)}", {}, ""


with gr.Blocks(title="Audio-Visual Deepfake Detector") as demo:
    gr.Markdown("# 🛡️ Audio-Visual Deepfake Detection & Localization")
    gr.Markdown(
        "Unified deepfake detection integrating **Wav2Vec2** speech representations with the "
        "**AV-Deepfake1M** computer vision temporal benchmark. Classifies media into the standard taxonomy: "
        "`real`, `audio_modified`, `visual_modified`, or `both_modified`."
    )

    with gr.Row():
        with gr.Column():
            input_file = gr.File(
                label="Upload Audio or Video Media",
                file_types=["audio", "video"],
            )
            with gr.Accordion("Advanced Calibration Thresholds", open=False):
                audio_slider = gr.Slider(
                    minimum=0.50, maximum=0.99, value=0.85, step=0.05, label="Audio Fake Threshold"
                )
                visual_slider = gr.Slider(
                    minimum=0.40, maximum=0.95, value=0.65, step=0.05, label="Visual Fake Threshold"
                )
            submit_btn = gr.Button("Run Forensic Detection", variant="primary")

        with gr.Column():
            output_verdict = gr.Markdown(label="Multimodal Verdict")
            output_probs = gr.Label(label="Modality Probabilities", num_top_classes=4)
            output_forensics = gr.Markdown(label="Forensic Metrics")

    submit_btn.click(
        fn=classify_media,
        inputs=[input_file, audio_slider, visual_slider],
        outputs=[output_verdict, output_probs, output_forensics],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
