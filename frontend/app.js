// Multimodal Deepfake Detector - Frontend Application Logic

let selectedFile = null;

// DOM Elements
const systemStatus = document.getElementById('systemStatus');
const dropZone = document.getElementById('dropZone');
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

// 2. File Selection & Drag-and-Drop
function updateSelectedFile(file) {
    if (!file) return;
    selectedFile = file;

    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    fileIcon.textContent = file.type.startsWith('video') ? '🎬' : '🎵';

    fileInfo.style.display = 'flex';
    dropZone.style.display = 'none';
    btnAnalyze.disabled = false;
}

function clearSelectedFile() {
    selectedFile = null;
    mediaInput.value = '';
    fileInfo.style.display = 'none';
    dropZone.style.display = 'block';
    btnAnalyze.disabled = true;
    resultsCard.style.display = 'none';
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

// 3. Quick Reference Samples
async function loadSampleMedia(filename) {
    try {
        btnAnalyze.disabled = true;
        systemStatus.textContent = 'Loading sample...';
        const res = await fetch(`/samples/${filename}`);
        if (!res.ok) throw new Error('Sample not found');

        const blob = await res.blob();
        const file = new File([blob], filename, { type: blob.type || 'audio/wav' });
        updateSelectedFile(file);
        checkSystemHealth();
    } catch (err) {
        alert(`Failed to load sample: ${err.message}`);
        checkSystemHealth();
    }
}

// 4. Threshold Calibration Presets
function getThresholds() {
    const preset = thresholdPreset.value;
    if (preset === 'strict') {
        return { audio: 0.75, visual: 0.50 };
    } else if (preset === 'permissive') {
        return { audio: 0.92, visual: 0.80 };
    }
    return { audio: 0.85, visual: 0.65 }; // balanced / mobile
}

// 5. Run Forensic Analysis
btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile) return;

    btnAnalyze.disabled = true;
    loadingIndicator.style.display = 'block';
    resultsCard.style.display = 'none';

    const thresholds = getThresholds();
    const formData = new FormData();
    formData.append('file', selectedFile);

    const queryUrl = `/api/predict?audio_threshold=${thresholds.audio}&visual_threshold=${thresholds.visual}`;

    try {
        const response = await fetch(queryUrl, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        alert(`Analysis Error: ${err.message}`);
    } finally {
        btnAnalyze.disabled = false;
        loadingIndicator.style.display = 'none';
    }
});

// 6. Render Prediction Results
function renderResults(data) {
    const verdictBadge = document.getElementById('verdictBadge');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');

    const verdict = data.overall_verdict || 'unknown';
    const confidence = ((data.overall_confidence || 0) * 100).toFixed(1);

    // Set Verdict
    verdictBadge.textContent = verdict.replace(/_/g, ' ').toUpperCase();
    if (verdict === 'real') {
        verdictBadge.className = 'verdict-badge verdict-real';
        confidenceBar.style.backgroundColor = 'var(--success)';
    } else if (verdict === 'both_modified') {
        verdictBadge.className = 'verdict-badge verdict-fake';
        confidenceBar.style.backgroundColor = 'var(--danger)';
    } else {
        verdictBadge.className = 'verdict-badge verdict-warning';
        confidenceBar.style.backgroundColor = 'var(--warning)';
    }

    confidenceValue.textContent = `${confidence}%`;
    confidenceBar.style.width = `${confidence}%`;

    // Audio Breakdown
    const audioData = data.audio_analysis || {};
    document.getElementById('audioVerdict').textContent = (audioData.prediction || 'N/A').toUpperCase();
    document.getElementById('audioConfidence').textContent = audioData.confidence !== undefined ? `${(audioData.confidence * 100).toFixed(1)}%` : 'N/A';
    document.getElementById('audioDuration').textContent = audioData.duration_seconds !== undefined ? `${audioData.duration_seconds.toFixed(2)}s` : 'N/A';

    // Visual Breakdown
    const visualData = data.visual_analysis || {};
    document.getElementById('visualVerdict').textContent = (visualData.prediction || 'N/A').toUpperCase();
    document.getElementById('visualConfidence').textContent = visualData.confidence !== undefined ? `${(visualData.confidence * 100).toFixed(1)}%` : 'N/A';
    document.getElementById('visualFrames').textContent = visualData.frames_analyzed !== undefined ? `${visualData.faces_detected || 0} faces / ${visualData.frames_analyzed} frames` : 'N/A';

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

    // Raw JSON Output
    document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);

    // Show Card & Scroll
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });
}

// Initial health check on page load
checkSystemHealth();
