// Multimodal Deepfake Detector - Frontend Application Logic

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
const thresholdPreset = document.getElementById('thresholdPreset');
const resultsCard = document.getElementById('resultsCard');
const errorBanner = document.getElementById('errorBanner');
const errorMessage = document.getElementById('errorMessage');

const previewBox = document.getElementById('previewBox');
const previewBadge = document.getElementById('previewBadge');
const previewAudioNotice = document.getElementById('previewAudioNotice');
const videoPlayer = document.getElementById('videoPlayer');
const audioPlayer = document.getElementById('audioPlayer');

const btnModeAudio = document.getElementById('btnModeAudio');
const btnModeVideo = document.getElementById('btnModeVideo');
const audioSampleChips = document.getElementById('audioSampleChips');
const videoSampleChips = document.getElementById('videoSampleChips');
const audioCard = document.getElementById('audioCard');
const visualCard = document.getElementById('visualCard');

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

// 1. Mode Switching Logic
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
        dropTitle.textContent = 'Click to select or drag & drop audio file';
        dropHelp.textContent = 'Supports pure Audio formats only (WAV, MP3, FLAC, M4A, OGG). Video files not permitted.';
        audioSampleChips.style.display = 'inline-flex';
        videoSampleChips.style.display = 'none';

        thresholdPreset.innerHTML = `
            <option value="balanced" selected>🛡️ Balanced Mode (Threshold 85%)</option>
            <option value="strict">⚠️ Strict Forensics (Threshold 75%)</option>
            <option value="permissive">⚡ High Confidence Only (Threshold 92%)</option>
        `;
    } else {
        btnModeVideo.classList.add('active');
        btnModeAudio.classList.remove('active');
        mediaInput.accept = 'video/*';
        dropIcon.textContent = '👁️';
        dropTitle.textContent = 'Click to select or drag & drop video media';
        dropHelp.textContent = 'Supports Video containers (MP4, WebM, AVI, MOV, MKV)';
        audioSampleChips.style.display = 'none';
        videoSampleChips.style.display = 'inline-flex';

        thresholdPreset.innerHTML = `
            <option value="balanced" selected>🛡️ Balanced Mode (Threshold 65%)</option>
            <option value="strict">⚠️ Strict Forensics (Threshold 50%)</option>
            <option value="permissive">⚡ High Confidence Only (Threshold 80%)</option>
        `;
    }

    // Incompatible file checks on mode switch
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

btnModeAudio.addEventListener('click', () => setMode('audio'));
btnModeVideo.addEventListener('click', () => setMode('video'));

// 2. Check System Health
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

// 3. File Selection, Media Preview & Drag-and-Drop
function displayPreview(file) {
    if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
    }

    // Pause and reset existing players
    videoPlayer.pause();
    videoPlayer.removeAttribute('src');
    videoPlayer.load();
    videoPlayer.style.display = 'none';

    audioPlayer.pause();
    audioPlayer.removeAttribute('src');
    audioPlayer.load();
    audioPlayer.style.display = 'none';

    const isVideo = file.type.startsWith('video') || /\.(mp4|webm|avi|mov|mkv|flv|wmv|m4v)$/i.test(file.name);
    currentObjectUrl = URL.createObjectURL(file);

    if (isVideo) {
        videoPlayer.src = currentObjectUrl;
        videoPlayer.style.display = 'block';
        previewBadge.textContent = '🎬 Video & Audio Preview';
        previewBadge.className = 'preview-badge badge-video';
        previewAudioNotice.textContent = 'Synchronized video & audio playback ready';
        previewAudioNotice.style.display = 'inline';
    } else {
        audioPlayer.src = currentObjectUrl;
        audioPlayer.style.display = 'block';
        previewBadge.textContent = '🎵 Audio Track Preview';
        previewBadge.className = 'preview-badge badge-audio';
        previewAudioNotice.textContent = 'Speech waveform & acoustic player ready';
        previewAudioNotice.style.display = 'inline';
    }

    previewBox.style.display = 'block';
}

function updateSelectedFile(file) {
    if (!file) return;
    hideError();

    const isVideo = file.type.startsWith('video') || /\.(mp4|webm|avi|mov|mkv|flv|wmv|m4v)$/i.test(file.name);
    const isAudio = file.type.startsWith('audio') || /\.(wav|mp3|flac|ogg|m4a|aac|wma|opus)$/i.test(file.name);

    if (currentMode === 'audio' && isVideo) {
        showError("Video files are not permitted in Audio Defect Detection mode. Please upload an audio file (WAV, MP3, FLAC, etc.).");
        return;
    }
    if (currentMode === 'video' && isAudio) {
        showError("Video Defect Detection requires a video file (MP4, WebM, AVI, MOV). Please upload a video file.");
        return;
    }

    selectedFile = file;

    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    fileIcon.textContent = isVideo ? '🎬' : '🎵';

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

// 4. Quick Reference Samples
async function loadSampleMedia(filename) {
    try {
        hideError();
        btnAnalyze.disabled = true;
        systemStatus.textContent = 'Loading sample...';
        const res = await fetch(`/samples/${filename}`);
        if (!res.ok) throw new Error('Sample not found');

        const blob = await res.blob();
        const mimeType = filename.endsWith('.mp4') ? 'video/mp4' : (blob.type || 'audio/wav');
        const file = new File([blob], filename, { type: mimeType });
        updateSelectedFile(file);
        checkSystemHealth();
    } catch (err) {
        showError(`Failed to load sample: ${err.message}`);
        checkSystemHealth();
    }
}

// 5. Threshold Calibration Presets
function getThresholds() {
    const preset = thresholdPreset.value;
    if (currentMode === 'audio') {
        if (preset === 'strict') return { audio: 0.75, visual: 0.65 };
        if (preset === 'permissive') return { audio: 0.92, visual: 0.65 };
        return { audio: 0.85, visual: 0.65 };
    } else {
        if (preset === 'strict') return { audio: 0.85, visual: 0.50 };
        if (preset === 'permissive') return { audio: 0.85, visual: 0.80 };
        return { audio: 0.85, visual: 0.65 };
    }
}

// 6. Run Forensic Analysis
btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile) return;

    hideError();

    // Mode-specific pre-flight checks
    const isVideo = selectedFile.type.startsWith('video') || /\.(mp4|webm|avi|mov|mkv|flv|wmv|m4v)$/i.test(selectedFile.name);
    const isAudio = selectedFile.type.startsWith('audio') || /\.(wav|mp3|flac|ogg|m4a|aac|wma|opus)$/i.test(selectedFile.name);

    if (currentMode === 'audio' && isVideo) {
        showError("Video files are not permitted in Audio Defect Detection mode. Please upload an audio file (WAV, MP3, FLAC, etc.).");
        return;
    }
    if (currentMode === 'video' && isAudio) {
        showError("Video Defect Detection requires a video container (MP4, WebM, AVI, MOV). Please upload a video file.");
        return;
    }

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
        renderResults(data);
    } catch (err) {
        showError(`Analysis Error: ${err.message}`);
    } finally {
        btnAnalyze.disabled = false;
        loadingIndicator.style.display = 'none';
    }
});

// 7. Render Prediction Results
function renderResults(data) {
    const verdictBadge = document.getElementById('verdictBadge');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');

    const verdict = data.overall_verdict || 'unknown';
    const confidence = ((data.overall_confidence || 0) * 100).toFixed(1);

    // Set Verdict Badge
    verdictBadge.textContent = verdict.replace(/_/g, ' ').toUpperCase();
    if (verdict === 'real') {
        verdictBadge.className = 'verdict-badge verdict-real';
        confidenceBar.style.backgroundColor = 'var(--success)';
    } else {
        verdictBadge.className = 'verdict-badge verdict-fake';
        confidenceBar.style.backgroundColor = 'var(--danger)';
    }

    confidenceValue.textContent = `${confidence}%`;
    confidenceBar.style.width = `${confidence}%`;

    // Strict Modality Isolation: Display ONLY the active modality
    if (data.mode === 'audio' || data.audio_analysis) {
        // Show Audio Card, Hide Visual Card
        audioCard.style.display = 'block';
        visualCard.style.display = 'none';

        const audioData = data.audio_analysis || {};
        document.getElementById('audioVerdict').textContent = (audioData.prediction || 'N/A').toUpperCase();
        document.getElementById('audioConfidence').textContent = audioData.confidence != null ? `${(audioData.confidence * 100).toFixed(1)}%` : 'N/A';
        document.getElementById('audioDuration').textContent = audioData.duration_seconds != null ? `${audioData.duration_seconds.toFixed(2)}s` : 'N/A';
    } else if (data.mode === 'video' || data.visual_analysis) {
        // Show Visual Card, Hide Audio Card
        visualCard.style.display = 'block';
        audioCard.style.display = 'none';

        const visualData = data.visual_analysis || {};
        document.getElementById('visualVerdict').textContent = (visualData.prediction || 'N/A').toUpperCase();
        document.getElementById('visualConfidence').textContent = visualData.confidence !== undefined ? `${(visualData.confidence * 100).toFixed(1)}%` : 'N/A';
        document.getElementById('visualFrames').textContent = visualData.frames_analyzed !== undefined ? `${visualData.faces_detected || 0} faces / ${visualData.frames_analyzed} frames` : 'N/A';
    }

    // Tampering Segments
    const tamperingSection = document.getElementById('tamperingSection');
    const tamperingList = document.getElementById('tamperingList');
    const fakeSegments = data.fake_segments || [];

    if (fakeSegments.length > 0) {
        tamperingList.innerHTML = fakeSegments.map(seg => 
            `<span class="tamper-tag">${seg[0].toFixed(2)}s - ${seg[1].toFixed(2)}s</span>`
        ).join('');
        tamperingSection.style.display = 'block';
    } else {
        tamperingSection.style.display = 'none';
    }

    // Raw JSON Output (Contains only the analyzed modality)
    document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);

    // Show Card & Scroll
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });
}

// Initial health check on page load
checkSystemHealth();
