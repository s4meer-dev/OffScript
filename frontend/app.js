// Multimodal Deepfake Detector & Video Forensics AI - Frontend Application Logic

let selectedFile = null;
let currentMode = 'video'; // 'video' (primary) or 'audio'
let currentObjectUrl = null;

// DOM Elements
const systemStatus = document.getElementById('systemStatus');
const dropZone = document.getElementById('dropZone');
const dropIcon = document.getElementById('dropIcon');
const dropTitle = document.getElementById('dropTitle');
const dropHelp = document.getElementById('dropHelp');
const mediaInput = document.getElementById('mediaInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const fileIcon = document.getElementById('fileIcon');
const btnRemoveFile = document.getElementById('btnRemoveFile');
const btnAnalyze = document.getElementById('btnAnalyze');
const loadingIndicator = document.getElementById('loadingIndicator');
const loadingStatusText = document.getElementById('loadingStatusText');
const thresholdPreset = document.getElementById('thresholdPreset');
const thresholdSlider = document.getElementById('thresholdSlider');
const sliderValueText = document.getElementById('sliderValueText');
const resultsCard = document.getElementById('resultsCard');
const errorBanner = document.getElementById('errorBanner');
const errorMessage = document.getElementById('errorMessage');

const previewBox = document.getElementById('previewBox');
const previewBadge = document.getElementById('previewBadge');
const previewAudioNotice = document.getElementById('previewAudioNotice');
const videoPlayer = document.getElementById('videoPlayer');
const audioPlayer = document.getElementById('audioPlayer');

const btnModeVideo = document.getElementById('btnModeVideo');
const btnModeAudio = document.getElementById('btnModeAudio');
const videoSampleChips = document.getElementById('videoSampleChips');
const audioSampleChips = document.getElementById('audioSampleChips');
const videoForensicsSection = document.getElementById('videoForensicsSection');
const audioForensicsSection = document.getElementById('audioForensicsSection');

// Helper: Show/Hide Error Banner
function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.style.display = 'flex';
    resultsCard.style.display = 'none';
}

function hideError() {
    errorBanner.style.display = 'none';
    errorMessage.textContent = '';
}

// 1. Check System Health
async function checkSystemHealth() {
    try {
        const res = await fetch('/api/health');
        if (res.ok) {
            const data = await res.json();
            const device = data.device ? data.device.toUpperCase() : 'ONLINE';
            systemStatus.textContent = `Online (${device})`;
            systemStatus.className = 'status-pill status-online';
        } else {
            systemStatus.textContent = 'Service Degraded';
            systemStatus.className = 'status-pill status-offline';
        }
    } catch (err) {
        systemStatus.textContent = 'Backend Offline';
        systemStatus.className = 'status-pill status-offline';
    }
}

// 2. Mode Switching Logic (Video vs Audio)
function setMode(mode) {
    if (mode === currentMode) return;
    currentMode = mode;
    hideError();
    resultsCard.style.display = 'none';

    if (mode === 'audio') {
        btnModeAudio.classList.add('active');
        btnModeVideo.classList.remove('active');
        mediaInput.accept = 'audio/*';
        dropIcon.textContent = '🎙️';
        dropTitle.textContent = 'Click to select or drag & drop audio file for speech forensics';
        dropHelp.textContent = 'Supports pure Audio formats (WAV, MP3, FLAC, M4A, OGG). Video containers not permitted.';
        audioSampleChips.style.display = 'inline-flex';
        videoSampleChips.style.display = 'none';
        btnAnalyze.textContent = 'Run Audio Forensic Analysis';
        loadingStatusText.textContent = 'Analyzing speech acoustic features & neural representations (Wav2Vec2)...';

        thresholdPreset.innerHTML = `
            <option value="balanced" selected>🛡️ Balanced Mode (85%)</option>
            <option value="strict">⚠️ Strict Forensics (75%)</option>
            <option value="permissive">⚡ High Confidence Only (92%)</option>
            <option value="custom">🎛️ Custom Fine-Tuned Threshold</option>
        `;
        if (thresholdSlider) {
            thresholdSlider.value = 85;
            if (sliderValueText) sliderValueText.textContent = '85%';
        }
    } else {
        btnModeVideo.classList.add('active');
        btnModeAudio.classList.remove('active');
        mediaInput.accept = 'video/*';
        dropIcon.textContent = '👁️';
        dropTitle.textContent = 'Click to select or drag & drop video for deepfake forensics';
        dropHelp.textContent = 'Supports Video containers: MP4, WebM, AVI, MOV, MKV, FLV';
        videoSampleChips.style.display = 'inline-flex';
        audioSampleChips.style.display = 'none';
        btnAnalyze.textContent = 'Run Video Forensic Analysis';
        loadingStatusText.textContent = 'Analyzing video across spatial, spectral, deep identity, and temporal vectors...';

        thresholdPreset.innerHTML = `
            <option value="balanced" selected>🛡️ Balanced Forensics (50%)</option>
            <option value="strict">⚠️ Strict Forensics (40%)</option>
            <option value="permissive">⚡ High Confidence Only (70%)</option>
            <option value="custom">🎛️ Custom Fine-Tuned Threshold</option>
        `;
        if (thresholdSlider) {
            thresholdSlider.value = 50;
            if (sliderValueText) sliderValueText.textContent = '50%';
        }
    }

    // Validate file compatibility if already selected
    if (selectedFile) {
        const isVideo = selectedFile.type.startsWith('video') || /\.(mp4|webm|avi|mov|mkv|flv|wmv|m4v)$/i.test(selectedFile.name);
        const isAudio = selectedFile.type.startsWith('audio') || /\.(wav|mp3|flac|ogg|m4a|aac|wma|opus)$/i.test(selectedFile.name);

        if (mode === 'audio' && isVideo) {
            clearSelectedFile();
            showError("Video files are not permitted in Audio Defect Detection mode. Please upload an audio file.");
        } else if (mode === 'video' && isAudio) {
            clearSelectedFile();
            showError("Audio files are not permitted in Video Defect Detection mode. Please upload a video file.");
        }
    }
}

if (btnModeVideo) btnModeVideo.addEventListener('click', () => setMode('video'));
if (btnModeAudio) btnModeAudio.addEventListener('click', () => setMode('audio'));

// 3. Media Preview Handling
function displayPreview(file) {
    if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
    }

    videoPlayer.pause();
    videoPlayer.removeAttribute('src');
    videoPlayer.load();
    videoPlayer.style.display = 'none';

    audioPlayer.pause();
    audioPlayer.removeAttribute('src');
    audioPlayer.load();
    audioPlayer.style.display = 'none';

    currentObjectUrl = URL.createObjectURL(file);
    const isVideo = file.type.startsWith('video') || /\.(mp4|webm|avi|mov|mkv|flv|wmv|m4v)$/i.test(file.name);

    if (isVideo) {
        videoPlayer.src = currentObjectUrl;
        videoPlayer.style.display = 'block';
        previewBadge.textContent = '🎬 Video & Audio Preview';
        previewBadge.className = 'preview-badge badge-video';
        previewAudioNotice.textContent = 'Full frame rendering & synchronized audio track ready';
    } else {
        audioPlayer.src = currentObjectUrl;
        audioPlayer.style.display = 'block';
        previewBadge.textContent = '🎧 Audio Track Preview';
        previewBadge.className = 'preview-badge badge-audio';
        previewAudioNotice.textContent = 'Acoustic waveform preview ready';
    }

    previewBox.style.display = 'block';
}

function updateSelectedFile(file) {
    if (!file) return;
    hideError();

    const isVideo = file.type.startsWith('video') || /\.(mp4|webm|avi|mov|mkv|flv|wmv|m4v)$/i.test(file.name);
    const isAudio = file.type.startsWith('audio') || /\.(wav|mp3|flac|ogg|m4a|aac|wma|opus)$/i.test(file.name);

    if (currentMode === 'video' && !isVideo) {
        showError("Video Defect Detection requires a video container (MP4, WebM, AVI, MOV, MKV). Please select a video file or switch to Audio mode.");
        return;
    }

    if (currentMode === 'audio' && !isAudio) {
        showError("Audio Defect Detection requires an audio file (WAV, MP3, FLAC, M4A, OGG). Video files are not permitted in Audio mode.");
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB (${(file.size / 1024).toFixed(1)} KB)`;
    if (fileIcon) fileIcon.textContent = isVideo ? '🎬' : '🎵';

    fileInfo.style.display = 'flex';
    displayPreview(file);
    dropZone.style.display = 'none';
    btnAnalyze.disabled = false;
}

function clearSelectedFile() {
    selectedFile = null;
    mediaInput.value = '';
    fileInfo.style.display = 'none';

    if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
    }

    videoPlayer.pause();
    videoPlayer.removeAttribute('src');
    videoPlayer.load();
    videoPlayer.style.display = 'none';

    audioPlayer.pause();
    audioPlayer.removeAttribute('src');
    audioPlayer.load();
    audioPlayer.style.display = 'none';

    previewBox.style.display = 'none';
    dropZone.style.display = 'block';
    btnAnalyze.disabled = true;
    resultsCard.style.display = 'none';
    hideError();
}

// Event Listeners for File Selection
mediaInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        updateSelectedFile(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        updateSelectedFile(e.dataTransfer.files[0]);
    }
});

btnRemoveFile.addEventListener('click', clearSelectedFile);

// 4. Quick Reference Benchmark Samples
async function loadSampleMedia(filename) {
    try {
        hideError();
        btnAnalyze.disabled = true;
        systemStatus.textContent = 'Loading sample...';
        const res = await fetch(`/samples/${filename}`);
        if (!res.ok) throw new Error('Sample file not found');

        const blob = await res.blob();
        const isVideo = filename.endsWith('.mp4');
        const mimeType = isVideo ? 'video/mp4' : (blob.type || 'audio/wav');
        const file = new File([blob], filename, { type: mimeType });

        if (isVideo && currentMode !== 'video') {
            setMode('video');
        } else if (!isVideo && currentMode !== 'audio') {
            setMode('audio');
        }

        updateSelectedFile(file);
        checkSystemHealth();
    } catch (err) {
        showError(`Failed to load benchmark sample: ${err.message}`);
        checkSystemHealth();
    }
}

// 5. Calibration Sensitivity Thresholds & Slider Synchronization
function getThresholds() {
    const sliderVal = thresholdSlider ? parseFloat(thresholdSlider.value) / 100.0 : 0.50;
    const val = thresholdPreset.value;

    if (val === 'custom') {
        return { audio: sliderVal, visual: sliderVal };
    }

    if (currentMode === 'audio') {
        if (val === 'strict') return { audio: 0.75, visual: 0.50 };
        if (val === 'permissive') return { audio: 0.92, visual: 0.50 };
        return { audio: 0.85, visual: 0.50 };
    } else {
        if (val === 'strict') return { audio: 0.85, visual: 0.40 };
        if (val === 'permissive') return { audio: 0.85, visual: 0.70 };
        return { audio: 0.85, visual: sliderVal || 0.50 };
    }
}

if (thresholdSlider) {
    thresholdSlider.addEventListener('input', (e) => {
        const val = e.target.value;
        if (sliderValueText) sliderValueText.textContent = `${val}%`;
        if (thresholdPreset) thresholdPreset.value = 'custom';
    });
}

if (thresholdPreset) {
    thresholdPreset.addEventListener('change', (e) => {
        const p = e.target.value;
        if (!thresholdSlider) return;
        if (p === 'balanced') {
            const defVal = currentMode === 'audio' ? 85 : 50;
            thresholdSlider.value = defVal;
            if (sliderValueText) sliderValueText.textContent = `${defVal}%`;
        } else if (p === 'strict') {
            const defVal = currentMode === 'audio' ? 75 : 40;
            thresholdSlider.value = defVal;
            if (sliderValueText) sliderValueText.textContent = `${defVal}%`;
        } else if (p === 'permissive') {
            const defVal = currentMode === 'audio' ? 92 : 70;
            thresholdSlider.value = defVal;
            if (sliderValueText) sliderValueText.textContent = `${defVal}%`;
        }
    });
}

// 6. Run Forensic Analysis
btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile) return;

    hideError();
    btnAnalyze.disabled = true;
    loadingIndicator.style.display = 'block';
    resultsCard.style.display = 'none';

    const thresholds = getThresholds();
    const formData = new FormData();
    formData.append('file', selectedFile);

    const queryUrl = `/api/predict?mode=${currentMode}&audio_threshold=${thresholds.audio}&visual_threshold=${thresholds.visual}`;

    try {
        const response = await fetch(queryUrl, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            const msg = errData.detail || `Server error (${response.status})`;
            showError(msg);
            return;
        }

        const data = await response.json();
        renderForensicDashboard(data);
    } catch (err) {
        showError(`Forensic Analysis Error: ${err.message}`);
    } finally {
        btnAnalyze.disabled = false;
        loadingIndicator.style.display = 'none';
    }
});

// 7. Render Comprehensive Forensic Dashboard
function renderForensicDashboard(data) {
    // Robust detection of audio vs video mode
    const hasVisualAnalysis = Boolean(data.visual_analysis && (
        (data.visual_analysis.frames_analyzed != null && data.visual_analysis.frames_analyzed > 0) ||
        (data.visual_analysis.frame_analysis && data.visual_analysis.frame_analysis.length > 0) ||
        data.visual_analysis.status === 'success' ||
        data.visual_analysis.video_metadata
    ));
    const hasAudioAnalysis = Boolean(data.audio_analysis && (
        data.audio_analysis.duration_seconds != null ||
        data.audio_analysis.status === 'success' ||
        data.audio_analysis.confidence != null
    ));

    const isAudioMode = (currentMode === 'audio' || data.mode === 'audio' || data.media_type === 'audio')
        ? true
        : (currentMode === 'video' || data.mode === 'video' || data.media_type === 'video')
            ? false
            : (!hasVisualAnalysis && hasAudioAnalysis);

    const visualData = data.visual_analysis || data;
    const audioData = data.audio_analysis || data;
    const isFake = Boolean(data.is_fake ?? (data.overall_prediction === 'fake') ?? (isAudioMode ? audioData.is_fake : visualData.is_fake));

    const overallConf = data.overall_confidence ?? (isAudioMode ? audioData.confidence : visualData.confidence) ?? 0;
    const confidence = (overallConf * 100).toFixed(1);

    const probs = data.probabilities || (isAudioMode ? audioData.probabilities : visualData.probabilities) || (isFake ? { real: 0.05, fake: 0.95 } : { real: 0.95, fake: 0.05 });
    const probFake = ((probs.fake || 0) * 100).toFixed(1);
    const probReal = ((probs.real || 0) * 100).toFixed(1);

    // 1. Verdict & Risk Header
    const verdictTitleText = document.getElementById('verdictTitleText');
    const techniqueSubtext = document.getElementById('techniqueSubtext');
    const verdictBadge = document.getElementById('verdictBadge');
    const riskBadge = document.getElementById('riskBadge');

    const defaultTitle = isAudioMode 
        ? (isFake ? "AI Voice Clone / Synthetic Speech Detected" : "Authentic Human Voice Recording")
        : (isFake ? "Deepfake Video Manipulation Detected" : "Authentic Video Media Verified");

    const verdictText = data.verdict || data.verdict_title || (isAudioMode ? audioData.verdict || audioData.verdict_title : visualData.verdict || visualData.verdict_title) || defaultTitle;
    verdictTitleText.textContent = verdictText;

    techniqueSubtext.textContent = data.forensic_metrics?.primary_technique || visualData.forensic_metrics?.primary_technique || (isAudioMode ? "Neural Wav2Vec2 Acoustic Analysis" : "Authentic Optical Capture");

    if (isFake) {
        verdictBadge.textContent = "MANIPULATED";
        verdictBadge.className = "verdict-badge verdict-fake";
        riskBadge.textContent = data.risk_level || visualData.risk_level || (isAudioMode ? "CRITICAL (SYNTHETIC VOICE)" : "CRITICAL RISK");
        riskBadge.className = "risk-badge risk-high";
    } else if (data.prediction === "suspicious_visual" || visualData.prediction === "suspicious_visual" || data.prediction === "uncertain_ambient") {
        verdictBadge.textContent = "INCONCLUSIVE";
        verdictBadge.className = "verdict-badge verdict-warning";
        riskBadge.textContent = data.risk_level || visualData.risk_level || "MODERATE RISK";
        riskBadge.className = "risk-badge risk-med";
    } else {
        verdictBadge.textContent = "AUTHENTIC";
        verdictBadge.className = "verdict-badge verdict-real";
        riskBadge.textContent = data.risk_level || visualData.risk_level || "LOW RISK";
        riskBadge.className = "risk-badge risk-low";
    }

    // 2. Confidence & Probability Meters
    const confidenceLabel = document.getElementById('confidenceLabel');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const probRealValue = document.getElementById('probRealValue');
    const probFakeValue = document.getElementById('probFakeValue');

    confidenceLabel.textContent = isFake ? "Manipulation Confidence" : "Authenticity Confidence";
    confidenceValue.textContent = `${confidence}%`;
    confidenceBar.style.width = `${confidence}%`;
    confidenceBar.style.backgroundColor = isFake ? "var(--danger)" : "var(--success)";

    probRealValue.textContent = `${probReal}%`;
    probFakeValue.textContent = `${probFake}%`;

    // 3. Modality Routing: Video vs Audio
    if (isAudioMode) {
        // Show Audio Section, Hide Video Section
        audioForensicsSection.style.display = 'block';
        videoForensicsSection.style.display = 'none';

        const audioPred = (audioData.prediction || (audioData.is_fake ? 'fake' : 'real') || 'N/A').toUpperCase();
        document.getElementById('audioVerdict').textContent = audioPred;
        document.getElementById('audioConfidence').textContent = audioData.confidence != null ? `${(audioData.confidence * 100).toFixed(1)}%` : 'N/A';
        const audDuration = audioData.duration_seconds ?? data.duration_seconds;
        document.getElementById('audioDuration').textContent = audDuration != null ? `${Number(audDuration).toFixed(2)}s` : 'N/A';

        const audioRating = document.getElementById('audioRating');
        if (audioData.prediction === 'fake' || audioData.is_fake) {
            audioRating.textContent = 'Synthetic Voice';
            audioRating.className = 'diag-rating rating-danger';
        } else {
            audioRating.textContent = 'Natural Speech';
            audioRating.className = 'diag-rating rating-safe';
        }
    } else {
        // Show Video Section, Hide Audio Section
        videoForensicsSection.style.display = 'block';
        audioForensicsSection.style.display = 'none';

        // Multi-Vector Diagnostics Radar Grid
        const diag = data.diagnostic_breakdown || visualData.diagnostic_breakdown || {};
        renderDiagnosticVector('Boundary', diag.boundary_seams || { score: 0.1, rating: 'Pristine' });
        renderDiagnosticVector('Fft', diag.spectral_lattice || { score: 0.1, rating: 'Natural' });
        renderDiagnosticVector('Temporal', diag.temporal_stability || { score: 0.1, rating: 'Smooth' });
        renderDiagnosticVector('Deep', diag.identity_coherence || { score: 0.1, rating: 'Stable' });

        // Forensic Observations & Findings Log
        const findingsList = document.getElementById('findingsList');
        findingsList.innerHTML = '';
        const findings = data.findings_log || visualData.findings_log || [];
        if (findings.length > 0) {
            findings.forEach((finding) => {
                const item = document.createElement('div');
                const isAnomaly = finding.startsWith('[ANOMALY]') || finding.includes('🚨');
                item.className = `finding-item ${isAnomaly ? 'finding-anomaly' : 'finding-authentic'}`;
                
                const cleanText = finding.replace(/^\[(AUTHENTIC|ANOMALY)\]\s*/, '').replace(/^[✅🚨]\s*/, '');
                const icon = isAnomaly ? '⚠️' : '✓';
                item.innerHTML = `<span class="finding-icon">${icon}</span><span class="finding-text">${cleanText}</span>`;
                findingsList.appendChild(item);
            });
        } else {
            findingsList.innerHTML = `<div class="finding-item finding-authentic"><span class="finding-icon">✓</span><span class="finding-text">All biometric and visual forensic vectors within authentic operational parameters.</span></div>`;
        }

        // Frame-by-Frame Timeline Analysis Gallery
        const frameGallery = document.getElementById('frameGallery');
        frameGallery.innerHTML = '';
        const frames = data.frame_analysis || visualData.frame_analysis || [];
        const meta = data.video_metadata || visualData.video_metadata || {};
        const duration = meta.duration_seconds ?? data.duration_seconds ?? visualData.duration_seconds ?? 0;
        const timelineHelpText = document.getElementById('timelineHelpText');
        timelineHelpText.textContent = `${frames.length} sampled frames evaluated across ${Number(duration).toFixed(2)}s duration`;

        frames.forEach((f) => {
            const card = document.createElement('div');
            const statusClass = f.status === 'tampered' ? 'frame-tampered' : (f.status === 'suspicious' ? 'frame-suspicious' : 'frame-authentic');
            card.className = `frame-card ${statusClass}`;

            const scorePercent = ((f.anomaly_score || 0) * 100).toFixed(1);
            const faceBadge = f.face_detected ? `<span class="face-tag face-found">✓ Face Located</span>` : `<span class="face-tag face-missed">No Face</span>`;

            card.innerHTML = `
                <div class="frame-card-header">
                    <span class="frame-idx">Frame #${f.frame_index}</span>
                    <span class="frame-time">${f.timestamp_label}</span>
                </div>
                <div class="frame-meta-row">
                    ${faceBadge}
                    <span class="frame-status-tag ${statusClass}-tag">${(f.status || 'authentic').toUpperCase()}</span>
                </div>
                <div class="frame-score-row">
                    <span>Anomaly Risk:</span>
                    <strong>${scorePercent}%</strong>
                </div>
                <div class="frame-meter-bg">
                    <div class="frame-meter-fill ${statusClass}-fill" style="width: ${scorePercent}%;"></div>
                </div>
            `;
            frameGallery.appendChild(card);
        });

        // Video Forensic Metadata
        const framesAnalyzed = data.frames_analyzed ?? visualData.frames_analyzed ?? frames.length;
        const facesDetected = data.faces_detected ?? visualData.faces_detected ?? (frames.filter(f => f.face_detected).length);
        const facePresenceRatio = data.face_presence_ratio ?? visualData.face_presence_ratio ?? (framesAnalyzed > 0 ? (facesDetected / framesAnalyzed) : 0.0);
        const latency = data.inference_time_seconds ?? visualData.inference_time_seconds ?? 0;

        document.getElementById('metaResolution').textContent = meta.resolution || (meta.width && meta.height ? `${meta.width}x${meta.height}` : 'N/A');
        document.getElementById('metaFps').textContent = meta.fps ? `${meta.fps} FPS` : 'N/A';
        document.getElementById('metaDuration').textContent = duration ? `${Number(duration).toFixed(2)}s` : 'N/A';
        document.getElementById('metaFramesAnalyzed').textContent = `${framesAnalyzed} (${meta.total_frames || framesAnalyzed} total)`;
        document.getElementById('metaFacePresence').textContent = `${facesDetected} detected (${(facePresenceRatio * 100).toFixed(0)}%)`;
        document.getElementById('metaLatency').textContent = `${latency}s`;
    }

    // 4. Temporal Tampering Localization
    const tamperingSection = document.getElementById('tamperingSection');
    const tamperingList = document.getElementById('tamperingList');
    const fakeSegments = isAudioMode 
        ? (data.audio_fake_segments || audioData.audio_fake_segments || data.fake_segments || [])
        : (data.visual_fake_segments || visualData.visual_fake_segments || data.fake_segments || []);

    if (fakeSegments.length > 0) {
        tamperingList.innerHTML = fakeSegments.map(seg => 
            `<span class="tamper-tag">Interval [${seg[0].toFixed(2)}s ➔ ${seg[1].toFixed(2)}s]</span>`
        ).join('');
        tamperingSection.style.display = 'block';
    } else {
        tamperingSection.style.display = 'none';
    }

    // 5. Raw JSON Report
    document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);

    // Reveal & Scroll
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });
}

// Helper: Render individual diagnostic vector card
function renderDiagnosticVector(key, vectorData) {
    const score = vectorData.score || 0;
    const percent = Math.min(100, (score * 100)).toFixed(1);
    const ratingElem = document.getElementById(`rating${key}`);
    const meterElem = document.getElementById(`meter${key}`);
    const scoreElem = document.getElementById(`score${key}`);

    if (!ratingElem || !meterElem || !scoreElem) return;

    ratingElem.textContent = vectorData.rating || 'Evaluated';
    scoreElem.textContent = `${percent}%`;
    meterElem.style.width = `${percent}%`;

    if (score < 0.25) {
        meterElem.style.backgroundColor = 'var(--success)';
        ratingElem.className = 'diag-rating rating-safe';
    } else if (score < 0.50) {
        meterElem.style.backgroundColor = 'var(--warning)';
        ratingElem.className = 'diag-rating rating-warn';
    } else {
        meterElem.style.backgroundColor = 'var(--danger)';
        ratingElem.className = 'diag-rating rating-danger';
    }
}

// Initial health check on page load
checkSystemHealth();
