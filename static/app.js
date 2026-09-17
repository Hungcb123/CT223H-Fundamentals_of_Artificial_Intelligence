/* ========================================================
   PneumoScan AI — Frontend Application Logic
   ======================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const sampleNormalBtn = document.getElementById('sampleNormalBtn');
    const samplePneumoniaBtn = document.getElementById('samplePneumoniaBtn');

    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const resultContent = document.getElementById('resultContent');
    const latencyBadge = document.getElementById('latencyBadge');
    const latencyValue = document.getElementById('latencyValue');

    const diagnosisBanner = document.getElementById('diagnosisBanner');
    const bannerIcon = document.getElementById('bannerIcon');
    const diagnosisTitle = document.getElementById('diagnosisTitle');
    const diagnosisSeverity = document.getElementById('diagnosisSeverity');
    const confidenceValue = document.getElementById('confidenceValue');

    const normalPercent = document.getElementById('normalPercent');
    const normalBar = document.getElementById('normalBar');
    const pneumoniaPercent = document.getElementById('pneumoniaPercent');
    const pneumoniaBar = document.getElementById('pneumoniaBar');

    const imageViewer = document.getElementById('imageViewer');
    const cardOriginal = document.getElementById('cardOriginal');
    const cardGradCAM = document.getElementById('cardGradCAM');
    const cardHeatmap = document.getElementById('cardHeatmap');
    const imgOriginal = document.getElementById('imgOriginal');
    const imgGradCAM = document.getElementById('imgGradCAM');
    const imgHeatmap = document.getElementById('imgHeatmap');
    const toggleTabs = document.querySelectorAll('.toggle-tab');

    const recommendationText = document.getElementById('recommendationText');
    const printReportBtn = document.getElementById('printReportBtn');
    const resetBtn = document.getElementById('resetBtn');

    const patientNameInput = document.getElementById('patientNameInput');
    const patientAgeInput = document.getElementById('patientAgeInput');
    const themeToggleBtn = document.getElementById('themeToggleBtn');

    let currentDiagnosisData = null;

    // ========================================================
    // THEME MANAGEMENT
    // ========================================================
    const savedTheme = localStorage.getItem('pneumoscan_theme') || 'dark';
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
    }

    themeToggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('light-theme');
        const isLight = document.body.classList.contains('light-theme');
        localStorage.setItem('pneumoscan_theme', isLight ? 'light' : 'dark');
    });

    // ========================================================
    // FILE INPUT & DRAG AND DROP
    // ========================================================
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // Paste from clipboard support
    window.addEventListener('paste', (e) => {
        const items = e.clipboardData?.items;
        if (items) {
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf('image') !== -1) {
                    const blob = items[i].getAsFile();
                    handleFileUpload(blob);
                    break;
                }
            }
        }
    });

    // ========================================================
    // SAMPLE PRESETS
    // ========================================================
    sampleNormalBtn.addEventListener('click', () => {
        fetchDiagnosis({ sample_id: 'normal' });
    });

    samplePneumoniaBtn.addEventListener('click', () => {
        fetchDiagnosis({ sample_id: 'pneumonia' });
    });

    // ========================================================
    // INFERENCE API CALL
    // ========================================================
    function handleFileUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn tệp hình ảnh hợp lệ (PNG, JPG, JPEG)');
            return;
        }
        fetchDiagnosis({ file: file });
    }

    async function fetchDiagnosis(payload) {
        // UI State: Loading
        emptyState.classList.add('hidden');
        resultContent.classList.add('hidden');
        loadingState.classList.remove('hidden');

        const formData = new FormData();
        if (payload.file) {
            formData.append('file', payload.file);
        } else if (payload.sample_id) {
            formData.append('sample_id', payload.sample_id);
        }

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || 'Lỗi xử lý chẩn đoán từ máy chủ');
            }

            const data = await response.json();
            currentDiagnosisData = data;
            renderDiagnosis(data);

        } catch (error) {
            console.error('Inference error:', error);
            alert(`Lỗi: ${error.message}`);
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
        }
    }

    // ========================================================
    // RENDER DIAGNOSIS
    // ========================================================
    function renderDiagnosis(data) {
        loadingState.classList.add('hidden');
        resultContent.classList.remove('hidden');

        // Latency
        latencyBadge.classList.remove('hidden');
        latencyValue.textContent = `${data.inference_time_ms} ms`;

        // Classification status
        const isPneumonia = (data.predicted_label === 'PNEUMONIA');

        if (isPneumonia) {
            diagnosisBanner.className = 'diagnosis-banner pneumonia';
            bannerIcon.innerHTML = `
                <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="2.2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
            `;
            diagnosisTitle.textContent = 'PNEUMONIA (VIÊM PHỔI)';
            diagnosisSeverity.textContent = data.severity;
        } else {
            diagnosisBanner.className = 'diagnosis-banner normal';
            bannerIcon.innerHTML = `
                <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="2.2">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            `;
            diagnosisTitle.textContent = 'NORMAL (PHỔI BÌNH THƯỜNG)';
            diagnosisSeverity.textContent = data.severity;
        }

        confidenceValue.textContent = `${data.confidence}%`;

        // Progress bars
        normalPercent.textContent = `${data.normal_prob}%`;
        normalBar.style.width = `${data.normal_prob}%`;

        pneumoniaPercent.textContent = `${data.pneumonia_prob}%`;
        pneumoniaBar.style.width = `${data.pneumonia_prob}%`;

        // Images
        imgOriginal.src = data.images.original;
        imgGradCAM.src = data.images.gradcam;
        imgHeatmap.src = data.images.heatmap;

        // Recommendation
        recommendationText.textContent = data.recommendation;

        // Scroll into view on mobile
        if (window.innerWidth <= 1080) {
            resultContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    // ========================================================
    // VIEW MODE TOGGLES (SIDE-BY-SIDE / GRADCAM ONLY)
    // ========================================================
    toggleTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            toggleTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const view = tab.getAttribute('data-view');
            if (view === 'side-by-side') {
                imageViewer.className = 'image-viewer-grid';
                cardOriginal.classList.remove('hidden');
                cardGradCAM.classList.remove('hidden');
                cardHeatmap.classList.add('hidden');
            } else if (view === 'gradcam-only') {
                imageViewer.className = 'image-viewer-grid single-view';
                cardOriginal.classList.add('hidden');
                cardGradCAM.classList.remove('hidden');
                cardHeatmap.classList.add('hidden');
            } else if (view === 'heatmap-only') {
                imageViewer.className = 'image-viewer-grid single-view';
                cardOriginal.classList.add('hidden');
                cardGradCAM.classList.add('hidden');
                cardHeatmap.classList.remove('hidden');
            }
        });
    });

    // ========================================================
    // RESET / NEW EXAM
    // ========================================================
    resetBtn.addEventListener('click', () => {
        currentDiagnosisData = null;
        fileInput.value = '';
        resultContent.classList.add('hidden');
        latencyBadge.classList.add('hidden');
        emptyState.classList.remove('hidden');
    });

    // ========================================================
    // PRINT MEDICAL REPORT
    // ========================================================
    printReportBtn.addEventListener('click', () => {
        if (!currentDiagnosisData) return;

        const patientName = patientNameInput.value.trim() || 'Nguyễn Văn A';
        const patientAge = patientAgeInput.value.trim() || '45 tuổi / Nam';
        const now = new Date();
        const dateStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')} - ${now.getDate().toString().padStart(2, '0')}/${(now.getMonth() + 1).toString().padStart(2, '0')}/${now.getFullYear()}`;

        document.getElementById('reportDate').textContent = dateStr;
        document.getElementById('repPatientName').textContent = patientName;
        document.getElementById('repPatientAge').textContent = patientAge;

        document.getElementById('repImgOriginal').src = currentDiagnosisData.images.original;
        document.getElementById('repImgGradCAM').src = currentDiagnosisData.images.gradcam;

        document.getElementById('repDiagnosis').textContent = (currentDiagnosisData.predicted_label === 'PNEUMONIA') 
            ? 'PHÁT HIỆN TỔN THƯƠNG VIÊM PHỔI (PNEUMONIA)' 
            : 'PHẾ TRƯỜNG TRONG GIỚI HẠN BÌNH THƯỜNG (NORMAL)';

        document.getElementById('repProbNormal').textContent = `${currentDiagnosisData.normal_prob}%`;
        document.getElementById('repProbPneumonia').textContent = `${currentDiagnosisData.pneumonia_prob}%`;
        document.getElementById('repConfidence').textContent = `${currentDiagnosisData.confidence}%`;
        document.getElementById('repNotes').textContent = currentDiagnosisData.recommendation;

        // Trigger native print dialog
        window.print();
    });
});
