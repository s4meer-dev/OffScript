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
const thresholdSliderGroup = document.getElementById('thresholdSliderGroup');
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
const visualForensicsSection = document.getElementById('visualForensicsSection');
const audioForensicsSection = document.getElementById('audioForensicsSection');

// Voice Live Talk Card
const liveTalkCard = document.getElementById('liveTalkCard');

// Camera Recording Elements (Video Only)
const cameraActionBar = document.getElementById('cameraActionBar');
const btnRecordCamera = document.getElementById('btnRecordCamera');
const chkCameraRecording = document.getElementById('chkCameraRecording');
const cameraBadge = document.getElementById('cameraBadge');
const cameraModal = document.getElementById('cameraModal');
const btnCloseCameraModal = document.getElementById('btnCloseCameraModal');
const webcamLiveFeed = document.getElementById('webcamLiveFeed');
const btnStartRec = document.getElementById('btnStartRec');
const btnStopRec = document.getElementById('btnStopRec');
const recordingTimerBadge = document.getElementById('recordingTimerBadge');
const recTimerText = document.getElementById('recTimerText');

let activeStream = null;
let activeRecorder = null;
let recordedBlobs = [];
let recTimerInterval = null;
let recStartTime = 0;

// Helper: Show/Hide Error Banner
function showError(msg) {
    if (errorMessage) errorMessage.textContent = msg;
    if (errorBanner) errorBanner.style.display = 'flex';
}

function hideError() {
    if (errorBanner) errorBanner.style.display = 'none';
}

// 1. Health Check on Backend
async function checkSystemHealth() {
    try {
        const res = await fetch('/api/health');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        if (data.status === 'online' || data.status === 'healthy') {
            if (systemStatus) {
                systemStatus.textContent = '● System Operational';
                systemStatus.className = 'status-pill status-online';
            }
        } else {
            if (systemStatus) {
                systemStatus.textContent = '▲ Models Initializing';
                systemStatus.className = 'status-pill status-loading';
            }
        }
    } catch (err) {
        if (systemStatus) {
            systemStatus.textContent = '✕ Server Offline';
            systemStatus.className = 'status-pill status-offline';
        }
    }
}

// Helper: Update Preset Dropdown Options per Mode
function updatePresetsForMode(mode) {
    if (!thresholdPreset) return;
    if (mode === 'video') {
        thresholdPreset.innerHTML = `
            <option value="balanced" selected>🛡️ Balanced Forensics (50%)</option>
            <option value="strict">⚠️ Strict Forensics (40%)</option>
            <option value="permissive">⚡ High Confidence Only (70%)</option>
            <option value="custom">🎛️ Custom Fine-Tuned Threshold</option>
        `;
        if (thresholdSlider) thresholdSlider.value = 50;
        if (sliderValueText) sliderValueText.textContent = '50%';
        if (thresholdSliderGroup) thresholdSliderGroup.style.display = 'flex';
    } else {
        thresholdPreset.innerHTML = `
            <option value="balanced" selected>🛡️ Balanced Mode (Threshold 85%)</option>
            <option value="strict">⚠️ Strict Forensics (Threshold 75%)</option>
            <option value="permissive">⚡ High Confidence Only (Threshold 92%)</option>
        `;
        if (thresholdSliderGroup) thresholdSliderGroup.style.display = 'none';
    }
}

// 2. Strict Modality Isolation (Switching Mode)
function setMode(mode) {
    currentMode = mode;
    hideError();

    // Hide any existing results so voice and video do not cross-contaminate
    if (resultsCard) resultsCard.style.display = 'none';

    if (mode === 'video') {
        if (btnModeVideo) btnModeVideo.classList.add('active');
        if (btnModeAudio) btnModeAudio.classList.remove('active');

        // Hide voice-specific live card, show video camera bar
        if (liveTalkCard) liveTalkCard.style.display = 'none';
        if (cameraActionBar) cameraActionBar.style.display = 'flex';

        // Sample benchmark chips
        if (videoSampleChips) videoSampleChips.style.display = 'inline';
        if (audioSampleChips) audioSampleChips.style.display = 'none';

        // Drop zone styling & accept filters
        if (dropIcon) dropIcon.textContent = '🎬';
        if (dropTitle) dropTitle.textContent = 'Click to select or drag & drop video for deepfake forensics';
        if (dropHelp) dropHelp.textContent = 'Supports Video containers: MP4, WebM, AVI, MOV, MKV, FLV';
        if (mediaInput) mediaInput.accept = 'video/*';

        updatePresetsForMode('video');
    } else {
        if (btnModeAudio) btnModeAudio.classList.add('active');
        if (btnModeVideo) btnModeVideo.classList.remove('active');

        // Show voice-specific live card, hide video camera bar
        if (liveTalkCard) liveTalkCard.style.display = 'block';
        if (cameraActionBar) cameraActionBar.style.display = 'none';

        // Sample benchmark chips
        if (videoSampleChips) videoSampleChips.style.display = 'none';
        if (audioSampleChips) audioSampleChips.style.display = 'inline';

        // Drop zone styling & accept filters
        if (dropIcon) dropIcon.textContent = '🎙️';
        if (dropTitle) dropTitle.textContent = 'Click to select or drag & drop audio for voice clone detection';
        if (dropHelp) dropHelp.textContent = 'Supports pure Audio formats only (WAV, MP3, FLAC, M4A, OGG). Video files not permitted.';
        if (mediaInput) mediaInput.accept = 'audio/*';

        updatePresetsForMode('audio');
    }

    // If an existing file belongs to the other modality, clear it
    if (selectedFile) {
        const lowerName = selectedFile.name.toLowerCase();
        const isVideo = lowerName.endsWith('.mp4') || lowerName.endsWith('.webm') || lowerName.endsWith('.avi') || lowerName.endsWith('.mov') || lowerName.endsWith('.mkv') || (selectedFile.type && selectedFile.type.startsWith('video/'));
        if ((mode === 'video' && !isVideo) || (mode === 'audio' && isVideo)) {
            clearSelectedFile();
        }
    }
}

if (btnModeVideo) btnModeVideo.addEventListener('click', () => setMode('video'));
if (btnModeAudio) btnModeAudio.addEventListener('click', () => setMode('audio'));

// 3. File Input & Drag and Drop Handling
if (dropZone) {
    dropZone.addEventListener('click', () => {
        if (mediaInput) mediaInput.click();
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            updateSelectedFile(e.dataTransfer.files[0]);
        }
    });
}

if (mediaInput) {
    mediaInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            updateSelectedFile(e.target.files[0]);
        }
    });
}

function updateSelectedFile(file) {
    if (!file) return;

    hideError();
    const lowerName = file.name.toLowerCase();
    const isVideo = lowerName.endsWith('.mp4') || lowerName.endsWith('.webm') || lowerName.endsWith('.avi') || lowerName.endsWith('.mov') || lowerName.endsWith('.mkv') || (file.type && file.type.startsWith('video/'));

    // Enforce strict modality separation on selection
    if (currentMode === 'video' && !isVideo) {
        showError("Audio files cannot be analyzed under Video Defect Detection. Please switch to Audio Defect Detection or choose a video file.");
        clearSelectedFile();
        return;
    }
    if (currentMode === 'audio' && isVideo) {
        showError("Video files are strictly prohibited under Audio Defect Detection. Please switch to Video Defect Detection or choose an audio file.");
        clearSelectedFile();
        return;
    }

    selectedFile = file;

    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatBytes(file.size);
    if (fileIcon) fileIcon.textContent = isVideo ? '🎬' : '🎙️';
    if (fileInfo) fileInfo.style.display = 'flex';
    if (btnAnalyze) btnAnalyze.disabled = false;

    // Media Preview
    if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
    }
    currentObjectUrl = URL.createObjectURL(file);

    if (previewBox && videoPlayer && audioPlayer) {
        previewBox.style.display = 'block';

        if (isVideo) {
            videoPlayer.style.display = 'block';
            audioPlayer.style.display = 'none';
            videoPlayer.src = currentObjectUrl;
            videoPlayer.load();
            if (previewBadge) previewBadge.textContent = 'Video Preview';
            if (previewAudioNotice) previewAudioNotice.textContent = 'Synchronized audio & video stream playback';
        } else {
            videoPlayer.style.display = 'none';
            audioPlayer.style.display = 'block';
            audioPlayer.src = currentObjectUrl;
            audioPlayer.load();
            if (previewBadge) previewBadge.textContent = 'Audio Waveform Playback';
            if (previewAudioNotice) previewAudioNotice.textContent = 'Acoustic timeline playback';
        }
    }

    if (resultsCard) resultsCard.style.display = 'none';
}

function clearSelectedFile() {
    selectedFile = null;
    if (mediaInput) mediaInput.value = '';
    if (fileInfo) fileInfo.style.display = 'none';
    if (btnAnalyze) btnAnalyze.disabled = true;
    if (previewBox) previewBox.style.display = 'none';
    if (videoPlayer) {
        videoPlayer.pause();
        videoPlayer.src = '';
    }
    if (audioPlayer) {
        audioPlayer.pause();
        audioPlayer.src = '';
    }
    if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
    }
    if (resultsCard) resultsCard.style.display = 'none';
    if (chkCameraRecording) chkCameraRecording.checked = false;
    hideError();
}

if (btnRemoveFile) {
    btnRemoveFile.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelectedFile();
    });
}

function formatBytes(bytes, decimals = 1) {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// 4. Load Sample Media
async function loadSampleMedia(filename) {
    hideError();
    try {
        const res = await fetch(`/samples/${filename}`);
        if (!res.ok) throw new Error(`Could not load test sample file (${res.status})`);
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
    const val = thresholdPreset ? thresholdPreset.value : 'balanced';

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
if (btnAnalyze) {
    btnAnalyze.addEventListener('click', async () => {
        if (!selectedFile) return;

        hideError();
        btnAnalyze.disabled = true;
        if (loadingIndicator) loadingIndicator.style.display = 'block';
        if (resultsCard) resultsCard.style.display = 'none';

        const thresholds = getThresholds();
        const formData = new FormData();
        formData.append('file', selectedFile);

        const isCam = chkCameraRecording ? chkCameraRecording.checked : false;
        let queryUrl = `/api/predict?mode=${currentMode}&audio_threshold=${thresholds.audio}&visual_threshold=${thresholds.visual}`;
        if (isCam) {
            queryUrl += `&is_camera=true`;
        }
        if (selectedFile && selectedFile._mediaDuration && Number(selectedFile._mediaDuration) > 0) {
            queryUrl += `&duration=${Number(selectedFile._mediaDuration).toFixed(2)}`;
        }

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
            if (loadingIndicator) loadingIndicator.style.display = 'none';
        }
    });
}

// 7. Render Forensic Dashboard with Strict Isolation
function renderForensicDashboard(data) {
    const isAudioMode = (currentMode === 'audio' || data.mode === 'audio' || data.media_type === 'audio');
    const visualData = data.visual_analysis || data;
    const audioData = data.audio_analysis || data;

    const verdictTitleText = document.getElementById('verdictTitleText');
    const techniqueSubtext = document.getElementById('techniqueSubtext');
    const verdictEyebrow = document.getElementById('verdictEyebrow');
    const verdictBadge = document.getElementById('verdictBadge');
    const riskBadge = document.getElementById('riskBadge');
    const confidenceLabel = document.getElementById('confidenceLabel');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const probRealValue = document.getElementById('probRealValue');
    const probFakeValue = document.getElementById('probFakeValue');

    if (isAudioMode) {
        // ==========================================
        // VOICE / AUDIO RESULTS ONLY
        // ==========================================
        if (visualForensicsSection) visualForensicsSection.style.display = 'none';
        if (audioForensicsSection) audioForensicsSection.style.display = 'block';
        if (cameraBadge) cameraBadge.style.display = 'none';

        if (verdictEyebrow) verdictEyebrow.textContent = 'VOICE CLONE FORENSIC REPORT';

        const isFake = Boolean(
            data.is_fake ?? 
            (data.overall_prediction === 'fake') ?? 
            (data.overall_verdict === 'audio_modified') ?? 
            (audioData.prediction === 'FAKE') ??
            (data.speech_verdict === 'FAKE')
        );

        const defaultTitle = isFake ? "AI Voice Clone / Synthetic Speech Detected" : "Authentic Human Voice Recording";
        const titleText = data.speech_verdict ? (data.speech_verdict === "FAKE" ? "AI Voice Clone / Synthetic Speech Detected" : "Authentic Human Voice Verified") : (data.verdict || data.verdict_title || defaultTitle);
        if (verdictTitleText) verdictTitleText.textContent = titleText;

        if (techniqueSubtext) {
            techniqueSubtext.textContent = "Neural Wav2Vec2 Latent Forensics & Acoustic Analysis";
        }

        if (verdictBadge) {
            if (isFake) {
                verdictBadge.textContent = "MANIPULATED";
                verdictBadge.className = "verdict-badge verdict-fake";
                if (riskBadge) {
                    riskBadge.textContent = "CRITICAL (SYNTHETIC VOICE)";
                    riskBadge.className = "risk-badge risk-high";
                }
            } else {
                verdictBadge.textContent = "AUTHENTIC";
                verdictBadge.className = "verdict-badge verdict-real";
                if (riskBadge) {
                    riskBadge.textContent = "LOW (AUTHENTIC)";
                    riskBadge.className = "risk-badge risk-low";
                }
            }
        }

        const overallConf = data.overall_confidence ?? audioData.calibrated_confidence ?? audioData.confidence ?? 0;
        const confidence = (overallConf * 100).toFixed(1);

        if (confidenceLabel) {
            confidenceLabel.textContent = isFake ? "Manipulation Confidence" : "Authenticity Confidence";
        }
        if (confidenceValue) confidenceValue.textContent = `${confidence}%`;
        if (confidenceBar) {
            confidenceBar.style.width = `${confidence}%`;
            confidenceBar.style.backgroundColor = isFake ? 'var(--danger)' : 'var(--success)';
        }

        const probs = data.probabilities || audioData.probabilities || (isFake ? { real: 0.05, fake: 0.95 } : { real: 0.95, fake: 0.05 });
        const probFake = ((probs.fake ?? (isFake ? 0.95 : 0.05)) * 100).toFixed(1);
        const probReal = ((probs.real ?? (isFake ? 0.05 : 0.95)) * 100).toFixed(1);
        if (probRealValue) probRealValue.textContent = `${probReal}%`;
        if (probFakeValue) probFakeValue.textContent = `${probFake}%`;

        // Audio Metrics Grid
        const audioRating = document.getElementById('audioRating');
        const audioVerdict = document.getElementById('audioVerdict');
        const audioConfidence = document.getElementById('audioConfidence');
        const audioDuration = document.getElementById('audioDuration');

        if (audioRating) {
            audioRating.textContent = isFake ? "Synthetic Voice" : "Natural Speech";
            audioRating.className = `diag-rating ${isFake ? 'rating-danger' : 'rating-safe'}`;
        }
        if (audioVerdict) {
            audioVerdict.textContent = data.speech_verdict ? (data.speech_verdict === "FAKE" ? "AI SYNTHETIC / CLONED VOICE" : "AUTHENTIC HUMAN SPEECH") : (isFake ? "AI SYNTHETIC / CLONED VOICE" : "AUTHENTIC HUMAN SPEECH");
        }
        if (audioConfidence) {
            audioConfidence.textContent = `${confidence}%`;
        }
        if (audioDuration) {
            const dur = audioData.duration != null ? audioData.duration : (audioData.duration_seconds || data.duration_seconds || 0);
            audioDuration.textContent = `${Number(dur).toFixed(2)}s`;
        }

        // Tampering Segments for Audio
        const tamperingSection = document.getElementById('tamperingSection');
        const tamperingList = document.getElementById('tamperingList');
        const fakeSegments = data.audio_fake_segments || audioData.audio_fake_segments || data.fake_segments || [];

        if (tamperingSection && tamperingList) {
            if (fakeSegments.length > 0) {
                tamperingList.innerHTML = fakeSegments.map(seg => 
                    `<span class="tamper-tag">Interval [${seg[0].toFixed(2)}s ➔ ${seg[1].toFixed(2)}s]</span>`
                ).join('');
                tamperingSection.style.display = 'block';
            } else {
                tamperingSection.style.display = 'none';
            }
        }
    } else {
        // ==========================================
        // VIDEO FORENSIC RESULTS ONLY
        // ==========================================
        if (audioForensicsSection) audioForensicsSection.style.display = 'none';
        if (visualForensicsSection) visualForensicsSection.style.display = 'block';

        if (verdictEyebrow) verdictEyebrow.textContent = 'VIDEO FORENSIC ANALYSIS REPORT';

        const isFake = Boolean(data.is_fake ?? (data.overall_prediction === 'fake') ?? (visualData.is_fake));

        const defaultTitle = isFake ? "Deepfake Video Manipulation Detected" : "Authentic Video Media Verified";
        const verdictText = data.verdict || data.verdict_title || visualData.verdict || visualData.verdict_title || defaultTitle;
        if (verdictTitleText) verdictTitleText.textContent = verdictText;

        if (techniqueSubtext) {
            techniqueSubtext.textContent = data.forensic_metrics?.primary_technique || visualData.forensic_metrics?.primary_technique || "Computer Vision & Spatial-Temporal Biometrics";
        }

        const isCameraRec = Boolean(
            data.is_camera_recording ||
            visualData.is_camera_recording ||
            data.forensic_metrics?.is_camera_recording ||
            visualData.forensic_metrics?.is_camera_recording
        );

        if (cameraBadge) {
            if (isCameraRec) {
                cameraBadge.style.display = 'inline-flex';
                cameraBadge.textContent = '📹 CAMERA RECORDING';
            } else {
                cameraBadge.style.display = 'none';
            }
        }

        if (verdictBadge) {
            if (isFake) {
                verdictBadge.textContent = "MANIPULATED";
                verdictBadge.className = "verdict-badge verdict-fake";
                if (riskBadge) {
                    riskBadge.textContent = data.risk_level || visualData.risk_level || "CRITICAL RISK";
                    riskBadge.className = "risk-badge risk-high";
                }
            } else if (data.prediction === "suspicious_visual" || visualData.prediction === "suspicious_visual") {
                verdictBadge.textContent = "INCONCLUSIVE";
                verdictBadge.className = "verdict-badge verdict-warning";
                if (riskBadge) {
                    riskBadge.textContent = data.risk_level || visualData.risk_level || "MODERATE RISK";
                    riskBadge.className = "risk-badge risk-med";
                }
            } else {
                verdictBadge.textContent = "AUTHENTIC";
                verdictBadge.className = "verdict-badge verdict-real";
                if (riskBadge) {
                    riskBadge.textContent = data.risk_level || visualData.risk_level || "LOW RISK";
                    riskBadge.className = "risk-badge risk-low";
                }
            }
        }

        const overallConf = data.overall_confidence ?? visualData.confidence ?? 0;
        const confidence = (overallConf * 100).toFixed(1);

        if (confidenceLabel) {
            confidenceLabel.textContent = isFake ? "Manipulation Confidence" : "Authenticity Confidence";
        }
        if (confidenceValue) confidenceValue.textContent = `${confidence}%`;
        if (confidenceBar) {
            confidenceBar.style.width = `${confidence}%`;
            if (isFake) {
                confidenceBar.style.backgroundColor = 'var(--danger)';
            } else if (data.prediction === "suspicious_visual" || visualData.prediction === "suspicious_visual") {
                confidenceBar.style.backgroundColor = 'var(--warning)';
            } else {
                confidenceBar.style.backgroundColor = 'var(--success)';
            }
        }

        const probs = data.probabilities || visualData.probabilities || (isFake ? { real: 0.1, fake: 0.9 } : { real: 0.9, fake: 0.1 });
        const probFake = ((probs.fake ?? (isFake ? 0.9 : 0.1)) * 100).toFixed(1);
        const probReal = ((probs.real ?? (isFake ? 0.1 : 0.9)) * 100).toFixed(1);
        if (probRealValue) probRealValue.textContent = `${probReal}%`;
        if (probFakeValue) probFakeValue.textContent = `${probFake}%`;

        // 5 Diagnostic Vector Meters
        const diag = data.diagnostic_breakdown || visualData.diagnostic_breakdown || {};
        renderDiagnosticVector('FFT', diag.spectral_lattice || diag.frequency_domain_fft || {});
        renderDiagnosticVector('ELA', diag.boundary_seams || diag.error_level_analysis || {});
        renderDiagnosticVector('Phase', diag.temporal_stability || diag.temporal_phase_correlation || {});
        renderDiagnosticVector('Deep', diag.identity_coherence || diag.deep_spatial_artifacts || {});
        renderDiagnosticVector('Diffusion', diag.generative_diffusion || diag.generative_diffusion_ocular || {});

        // Observations & Findings Log
        const findingsList = document.getElementById('findingsList');
        const findings = data.findings_log || visualData.findings_log || [];
        if (findingsList) {
            findingsList.innerHTML = '';
            if (findings.length === 0) {
                findingsList.innerHTML = `<div class="finding-item finding-authentic"><span class="finding-icon">✓</span><span>All biometric & neural diagnostic vectors within verified natural variance bounds.</span></div>`;
            } else {
                findings.forEach(finding => {
                    const findingStr = typeof finding === 'string' ? finding : (finding.message || '');
                    const isAnomaly = findingStr.startsWith('[ANOMALY]') || findingStr.includes('🚨') || (finding.severity === 'danger');
                    const cleanText = findingStr.replace(/^\[(AUTHENTIC|ANOMALY|INFO|WARN)\]\s*/, '').replace(/^[✅🚨⚠️✓ℹ️]\s*/, '');
                    const icon = isAnomaly ? '⚠️' : '✓';
                    const item = document.createElement('div');
                    item.className = `finding-item ${isAnomaly ? 'finding-anomaly' : 'finding-authentic'}`;
                    item.innerHTML = `<span class="finding-icon">${icon}</span><span>${cleanText}</span>`;
                    findingsList.appendChild(item);
                });
            }
        }

        // Frame-by-Frame Timeline Gallery
        const frameGallery = document.getElementById('frameGallery');
        const timelineHelp = document.getElementById('timelineHelpText');
        const frames = data.frame_analysis || visualData.frame_analysis || [];
        const meta = data.video_metadata || visualData.video_metadata || {};
        const duration = data.duration_seconds || meta.duration_seconds || (frames.length > 0 ? frames[frames.length - 1].timestamp_sec : 0);

        if (timelineHelp) {
            timelineHelp.textContent = `${frames.length} frames evaluated across ${Number(duration).toFixed(1)}s timeline`;
        }

        if (frameGallery) {
            frameGallery.innerHTML = '';
            frames.forEach(f => {
                const card = document.createElement('div');
                const status = (f.status || 'authentic').toLowerCase();
                const statusClass = status === 'tampered' ? 'frame-tampered' : (status === 'suspicious' ? 'frame-suspicious' : 'frame-authentic');
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
                        <span class="frame-status-tag ${statusClass}-tag">${status.toUpperCase()}</span>
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
        }

        // Video Forensic Metadata
        const framesAnalyzed = data.frames_analyzed ?? visualData.frames_analyzed ?? frames.length;
        const facesDetected = data.faces_detected ?? visualData.faces_detected ?? (frames.filter(f => f.face_detected).length);
        const facePresenceRatio = data.face_presence_ratio ?? visualData.face_presence_ratio ?? (framesAnalyzed > 0 ? (facesDetected / framesAnalyzed) : 0.0);
        const latency = data.inference_time_seconds ?? visualData.inference_time_seconds ?? 0;

        const resElem = document.getElementById('metaResolution');
        const fpsElem = document.getElementById('metaFps');
        const durElem = document.getElementById('metaDuration');
        const faElem = document.getElementById('metaFramesAnalyzed');
        const fpElem = document.getElementById('metaFacePresence');
        const latElem = document.getElementById('metaLatency');

        if (resElem) resElem.textContent = meta.resolution || (meta.width && meta.height ? `${meta.width}x${meta.height}` : 'N/A');
        if (fpsElem) fpsElem.textContent = meta.fps ? `${meta.fps} FPS` : 'N/A';
        if (durElem) durElem.textContent = duration ? `${Number(duration).toFixed(2)}s` : 'N/A';
        if (faElem) faElem.textContent = `${framesAnalyzed} (${meta.total_frames || framesAnalyzed} total)`;
        if (fpElem) fpElem.textContent = `${facesDetected} detected (${(facePresenceRatio * 100).toFixed(0)}%)`;
        if (latElem) latElem.textContent = `${latency}s`;

        // Tampering Segments for Video
        const tamperingSection = document.getElementById('tamperingSection');
        const tamperingList = document.getElementById('tamperingList');
        const fakeSegments = data.visual_fake_segments || visualData.visual_fake_segments || data.fake_segments || [];

        if (tamperingSection && tamperingList) {
            if (fakeSegments.length > 0) {
                tamperingList.innerHTML = fakeSegments.map(seg => 
                    `<span class="tamper-tag">Interval [${seg[0].toFixed(2)}s ➔ ${seg[1].toFixed(2)}s]</span>`
                ).join('');
                tamperingSection.style.display = 'block';
            } else {
                tamperingSection.style.display = 'none';
            }
        }
    }

    // Common: Raw JSON and display
    const jsonOutput = document.getElementById('jsonOutput');
    if (jsonOutput) {
        jsonOutput.textContent = JSON.stringify(data, null, 2);
    }

    if (resultsCard) {
        resultsCard.style.display = 'block';
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }
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

// Camera Recording Modal & MediaRecorder Handler (Video Only)
function stopCameraStream() {
    if (activeRecorder && activeRecorder.state !== 'inactive') {
        try { activeRecorder.stop(); } catch (e) {}
    }
    if (activeStream) {
        activeStream.getTracks().forEach(track => track.stop());
        activeStream = null;
    }
    if (recTimerInterval) {
        clearInterval(recTimerInterval);
        recTimerInterval = null;
    }
    if (webcamLiveFeed) {
        webcamLiveFeed.srcObject = null;
    }
    if (recordingTimerBadge) recordingTimerBadge.style.display = 'none';
    if (btnStartRec) btnStartRec.style.display = 'inline-block';
    if (btnStopRec) btnStopRec.style.display = 'none';
}

if (btnRecordCamera) {
    btnRecordCamera.addEventListener('click', async () => {
        try {
            activeStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } },
                audio: true,
            });
            webcamLiveFeed.srcObject = activeStream;
            cameraModal.style.display = 'flex';
        } catch (err) {
            showError(`Camera access denied or unavailable: ${err.message}`);
        }
    });
}

if (btnCloseCameraModal) {
    btnCloseCameraModal.addEventListener('click', () => {
        stopCameraStream();
        cameraModal.style.display = 'none';
    });
}

if (btnStartRec) {
    btnStartRec.addEventListener('click', () => {
        if (!activeStream) return;
        recordedBlobs = [];
        
        let options = { mimeType: 'video/webm;codecs=vp8,opus' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options = { mimeType: 'video/webm' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = { mimeType: '' };
            }
        }

        try {
            activeRecorder = new MediaRecorder(activeStream, options);
        } catch (e) {
            activeRecorder = new MediaRecorder(activeStream);
        }

        activeRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                recordedBlobs.push(event.data);
            }
        };

        activeRecorder.onstop = () => {
            const superBuffer = new Blob(recordedBlobs, { type: 'video/webm' });
            const recDurationSec = Math.max(1, (Date.now() - recStartTime) / 1000);
            const recFile = new File([superBuffer], `camera_recording_${Date.now()}_dur_${Math.round(recDurationSec)}s.webm`, { type: 'video/webm' });
            recFile._mediaDuration = recDurationSec;
            if (chkCameraRecording) chkCameraRecording.checked = true;
            updateSelectedFile(recFile);
            stopCameraStream();
            cameraModal.style.display = 'none';
        };

        activeRecorder.start(100);
        btnStartRec.style.display = 'none';
        btnStopRec.style.display = 'inline-block';
        recordingTimerBadge.style.display = 'flex';
        recStartTime = Date.now();
        recTimerInterval = setInterval(() => {
            const elapsedSec = Math.floor((Date.now() - recStartTime) / 1000);
            const m = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
            const s = String(elapsedSec % 60).padStart(2, '0');
            recTimerText.textContent = `${m}:${s}`;
        }, 500);
    });
}

if (btnStopRec) {
    btnStopRec.addEventListener('click', () => {
        if (activeRecorder && activeRecorder.state === 'recording') {
            activeRecorder.stop();
        }
    });
}

// --- Live Talk Feature (Audio / Voice Only) ---
const btnLiveStart = document.getElementById('btnLiveStart');
const btnLiveStop = document.getElementById('btnLiveStop');
const liveTimer = document.getElementById('liveTimer');
const canvas = document.getElementById('liveGraph');
let canvasCtx = null;
if (canvas) {
    canvasCtx = canvas.getContext('2d');
}

let micMediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let dataArray;
let animationId;
let liveStartTime;
let liveTimerInterval;
let liveDownloadUrl = null;

if (btnLiveStart) {
    btnLiveStart.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Setup Audio Context for visualization
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            const bufferLength = analyser.frequencyBinCount;
            dataArray = new Uint8Array(bufferLength);
            
            // Start drawing the graph
            drawGraph();
            
            // Setup Media Recorder
            micMediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            micMediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            micMediaRecorder.onstop = async () => {
                cancelAnimationFrame(animationId);
                clearInterval(liveTimerInterval);
                if (liveTimer) liveTimer.style.display = 'none';
                if (btnLiveStart) btnLiveStart.style.display = 'inline-block';
                if (btnLiveStop) btnLiveStop.style.display = 'none';
                
                // Clear canvas
                if (canvasCtx && canvas) {
                    canvasCtx.fillStyle = '#0f172a';
                    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
                }
                
                // Upload the blob
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                
                // Configure Download Button
                const btnLiveDownload = document.getElementById('btnLiveDownload');
                if (btnLiveDownload) {
                    if (liveDownloadUrl) {
                        URL.revokeObjectURL(liveDownloadUrl);
                    }
                    liveDownloadUrl = URL.createObjectURL(audioBlob);
                    btnLiveDownload.href = liveDownloadUrl;
                    btnLiveDownload.style.display = 'inline-block';
                }
                
                // Send to backend simulating a file upload
                const file = new File([audioBlob], "live_audio_mic.wav", { type: 'audio/wav' });
                updateSelectedFile(file);
                
                // Enable analyze button
                if (btnAnalyze) btnAnalyze.disabled = false;
            };
            
            // Start recording
            micMediaRecorder.start();
            
            // UI updates
            btnLiveStart.style.display = 'none';
            if (btnLiveStop) btnLiveStop.style.display = 'inline-block';
            if (liveTimer) liveTimer.style.display = 'inline-block';
            const btnLiveDownload = document.getElementById('btnLiveDownload');
            if (btnLiveDownload) btnLiveDownload.style.display = 'none';
            
            liveStartTime = Date.now();
            liveTimerInterval = setInterval(() => {
                const elapsed = Math.floor((Date.now() - liveStartTime) / 1000);
                const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
                const secs = String(elapsed % 60).padStart(2, '0');
                if (liveTimer) liveTimer.textContent = `${mins}:${secs}`;
            }, 1000);
            
        } catch (err) {
            showError('Could not access microphone: ' + err.message);
        }
    });
}

if (btnLiveStop) {
    btnLiveStop.addEventListener('click', () => {
        if (micMediaRecorder && micMediaRecorder.state !== 'inactive') {
            micMediaRecorder.stop();
            micMediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    });
}

function drawGraph() {
    animationId = requestAnimationFrame(drawGraph);
    if (!analyser || !canvasCtx || !canvas) return;
    
    analyser.getByteFrequencyData(dataArray);
    
    // Smooth clear with trail effect for dynamic fluid motion
    canvasCtx.fillStyle = 'rgba(15, 23, 42, 0.25)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    const centerY = canvas.height / 2;
    const barWidth = (canvas.width / dataArray.length) * 2;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
        let barHeight = dataArray[i] * 0.75;
        if (barHeight < 2) barHeight = 2; // Minimum center line
        
        // Dynamic Sci-Fi Color mapping (Cyan to Purple)
        const hue = 190 + (i / dataArray.length) * 90;
        
        canvasCtx.shadowBlur = 15;
        canvasCtx.shadowColor = `hsl(${hue}, 100%, 50%)`;
        canvasCtx.fillStyle = `hsl(${hue}, 100%, 65%)`;
        
        // Draw symmetrical bars starting from center Y
        canvasCtx.fillRect(x, centerY - barHeight / 2, barWidth - 1, barHeight);
        
        x += barWidth;
    }
    
    canvasCtx.shadowBlur = 0;
}

// Initialize on page load: default to video mode
setMode('video');
checkSystemHealth();
