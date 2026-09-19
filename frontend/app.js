// Video Deepfake Forensics & AI Defect Detection - Frontend Controller

let selectedFile = null;
let currentObjectUrl = null;

// DOM Elements
const systemStatus = document.getElementById('systemStatus');
const dropZone = document.getElementById('dropZone');
const mediaInput = document.getElementById('mediaInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
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

// 2. Video Preview Handling
function displayPreview(file) {
    if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
    }

    videoPlayer.pause();
    videoPlayer.removeAttribute('src');
    videoPlayer.load();
    videoPlayer.style.display = 'none';

    if (audioPlayer) {
        audioPlayer.pause();
        audioPlayer.removeAttribute('src');
        audioPlayer.style.display = 'none';
    }

    currentObjectUrl = URL.createObjectURL(file);
    videoPlayer.src = currentObjectUrl;
    videoPlayer.style.display = 'block';

    previewBadge.textContent = '🎬 Video & Audio Preview';
    previewAudioNotice.textContent = 'Full frame rendering & synchronized audio track ready';

    previewBox.style.display = 'block';
}

function updateSelectedFile(file) {
    if (!file) return;
    hideError();

    const isVideo = file.type.startsWith('video') || /\.(mp4|webm|avi|mov|mkv|flv|wmv|m4v)$/i.test(file.name);
    if (!isVideo) {
        showError("Video Defect Detection requires a video container (MP4, WebM, AVI, MOV, MKV). Please select a video file.");
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB (${(file.size / 1024).toFixed(1)} KB)`;

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

// 3. Quick Reference Benchmark Samples
async function loadSampleMedia(filename) {
    try {
        hideError();
        btnAnalyze.disabled = true;
        systemStatus.textContent = 'Loading sample...';
        const res = await fetch(`/samples/${filename}`);
        if (!res.ok) throw new Error('Sample file not found');

        const blob = await res.blob();
        const file = new File([blob], filename, { type: 'video/mp4' });
        updateSelectedFile(file);
        checkSystemHealth();
    } catch (err) {
        showError(`Failed to load benchmark sample: ${err.message}`);
        checkSystemHealth();
    }
}

// 4. Calibration Sensitivity
function getVisualThreshold() {
    const val = thresholdPreset.value;
    if (val === 'strict') return 0.40;
    if (val === 'permissive') return 0.70;
    return 0.50; // balanced default
}

// 5. Run Video Forensic Analysis
btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile) return;

    hideError();
    btnAnalyze.disabled = true;
    loadingIndicator.style.display = 'block';
    resultsCard.style.display = 'none';

    const threshold = getVisualThreshold();
    const formData = new FormData();
    formData.append('file', selectedFile);

    const queryUrl = `/api/predict?mode=video&visual_threshold=${threshold}`;

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

// 6. Render Comprehensive Forensic Dashboard
function renderForensicDashboard(data) {
    const isFake = Boolean(data.is_fake);
    const confidence = ((data.overall_confidence || 0) * 100).toFixed(1);
    const probFake = ((data.probabilities?.fake || 0) * 100).toFixed(1);
    const probReal = ((data.probabilities?.real || 0) * 100).toFixed(1);

    // 1. Verdict & Risk Header
    const verdictTitleText = document.getElementById('verdictTitleText');
    const techniqueSubtext = document.getElementById('techniqueSubtext');
    const verdictBadge = document.getElementById('verdictBadge');
    const riskBadge = document.getElementById('riskBadge');

    verdictTitleText.textContent = data.verdict || (isFake ? "Deepfake Video Manipulation Detected" : "Authentic Video Media Verified");
    techniqueSubtext.textContent = data.forensic_metrics?.primary_technique || "Authentic Optical Capture";

    if (isFake) {
        verdictBadge.textContent = "MANIPULATED";
        verdictBadge.className = "verdict-badge verdict-fake";
        riskBadge.textContent = data.risk_level || "CRITICAL RISK";
        riskBadge.className = "risk-badge risk-high";
    } else if (data.prediction === "suspicious_visual") {
        verdictBadge.textContent = "INCONCLUSIVE";
        verdictBadge.className = "verdict-badge verdict-warning";
        riskBadge.textContent = data.risk_level || "MODERATE RISK";
        riskBadge.className = "risk-badge risk-med";
    } else {
        verdictBadge.textContent = "AUTHENTIC";
        verdictBadge.className = "verdict-badge verdict-real";
        riskBadge.textContent = data.risk_level || "LOW RISK";
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

    // 3. Multi-Vector Diagnostics Radar Grid
    const diag = data.diagnostic_breakdown || {};
    renderDiagnosticVector('Boundary', diag.boundary_seams || { score: 0.1, rating: 'Pristine' });
    renderDiagnosticVector('Fft', diag.spectral_lattice || { score: 0.1, rating: 'Natural' });
    renderDiagnosticVector('Temporal', diag.temporal_stability || { score: 0.1, rating: 'Smooth' });
    renderDiagnosticVector('Deep', diag.identity_coherence || { score: 0.1, rating: 'Stable' });

    // 4. Forensic Observations & Findings Log
    const findingsList = document.getElementById('findingsList');
    findingsList.innerHTML = '';
    const findings = data.findings_log || [];
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

    // 5. Frame-by-Frame Timeline Analysis Gallery
    const frameGallery = document.getElementById('frameGallery');
    frameGallery.innerHTML = '';
    const frames = data.frame_analysis || [];
    const timelineHelpText = document.getElementById('timelineHelpText');
    timelineHelpText.textContent = `${frames.length} sampled frames evaluated across ${data.video_metadata?.duration_seconds || 0}s duration`;

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
                <span class="frame-status-tag ${statusClass}-tag">${f.status.toUpperCase()}</span>
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

    // 6. Temporal Tampering Localization
    const tamperingSection = document.getElementById('tamperingSection');
    const tamperingList = document.getElementById('tamperingList');
    const fakeSegments = data.visual_fake_segments || data.fake_segments || [];

    if (fakeSegments.length > 0) {
        tamperingList.innerHTML = fakeSegments.map(seg => 
            `<span class="tamper-tag">Interval [${seg[0].toFixed(2)}s ➔ ${seg[1].toFixed(2)}s]</span>`
        ).join('');
        tamperingSection.style.display = 'block';
    } else {
        tamperingSection.style.display = 'none';
    }

    // 7. Video Forensic Metadata
    const meta = data.video_metadata || {};
    document.getElementById('metaResolution').textContent = meta.resolution || 'N/A';
    document.getElementById('metaFps').textContent = meta.fps ? `${meta.fps} FPS` : 'N/A';
    document.getElementById('metaDuration').textContent = meta.duration_seconds != null ? `${meta.duration_seconds.toFixed(2)}s` : 'N/A';
    document.getElementById('metaFramesAnalyzed').textContent = `${data.frames_analyzed || 0} (${meta.total_frames || 0} total)`;
    document.getElementById('metaFacePresence').textContent = `${data.faces_detected || 0} detected (${((data.face_presence_ratio || 0) * 100).toFixed(0)}%)`;
    document.getElementById('metaLatency').textContent = `${data.inference_time_seconds || 0}s`;

    // 8. Raw JSON Report
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
