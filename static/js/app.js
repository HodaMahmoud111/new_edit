let currentLang = 'en';
let isGenerating = false;
let abortController = null;

const STORAGE_KEYS = {
    MESSAGES: 'patellia_chat_messages',
    EVIDENCE: 'patellia_last_evidence',
    INSPECTOR: 'patellia_last_inspector',
    LANG: 'patellia_lang',
    IS_AR_QUESTION: 'patellia_is_ar_q'
};

document.addEventListener('DOMContentLoaded', () => {
    restoreSessionState();
});

function saveSessionState(lastEvidence = null, lastInspectorData = null, isArabicQ = null) {
    const list = document.getElementById('messagesList');
    if (list) {
        sessionStorage.setItem(STORAGE_KEYS.MESSAGES, list.innerHTML);
    }
    sessionStorage.setItem(STORAGE_KEYS.LANG, currentLang);

    if (lastEvidence !== null) {
        sessionStorage.setItem(STORAGE_KEYS.EVIDENCE, JSON.stringify(lastEvidence));
    }
    if (lastInspectorData !== null) {
        sessionStorage.setItem(STORAGE_KEYS.INSPECTOR, JSON.stringify(lastInspectorData));
    }
    if (isArabicQ !== null) {
        sessionStorage.setItem(STORAGE_KEYS.IS_AR_QUESTION, JSON.stringify(isArabicQ));
    }
}

function restoreSessionState() {
    const savedLang = sessionStorage.getItem(STORAGE_KEYS.LANG);
    if (savedLang && savedLang !== currentLang) {
        currentLang = savedLang;
        applyLanguageUI();
    }

    const savedMessages = sessionStorage.getItem(STORAGE_KEYS.MESSAGES);
    const list = document.getElementById('messagesList');
    if (savedMessages && list) {
        list.innerHTML = savedMessages;
        list.scrollTop = list.scrollHeight;
    }

    const savedEvidence = sessionStorage.getItem(STORAGE_KEYS.EVIDENCE);
    const savedInspector = sessionStorage.getItem(STORAGE_KEYS.INSPECTOR);
    const isArabicQ = JSON.parse(sessionStorage.getItem(STORAGE_KEYS.IS_AR_QUESTION) || 'false');

    if (savedEvidence) {
        try {
            renderEvidence(JSON.parse(savedEvidence), isArabicQ, false);
        } catch (e) {
            console.error('Error parsing evidence cache', e);
        }
    }

    if (savedInspector) {
        try {
            renderInspector(JSON.parse(savedInspector), isArabicQ, false);
        } catch (e) {
            console.error('Error parsing inspector cache', e);
        }
    }
}

function applyLanguageUI() {
    const html = document.documentElement;
    const langBtn = document.getElementById('langBtn');
    const input = document.getElementById('questionInput');
    const headerTitle = document.getElementById('headerTitle');
    const evidenceTitle = document.getElementById('evidenceTitle');

    if (currentLang === 'ar') {
        html.setAttribute('dir', 'rtl');
        html.setAttribute('lang', 'ar');
        if (langBtn) langBtn.textContent = 'English';
        if (input && !isGenerating) input.placeholder = 'اطرح سؤالك السريري حول المرجع...';
        if (headerTitle) headerTitle.textContent = 'مساحة الحوار مع الدليل الطبي';
        if (evidenceTitle) evidenceTitle.textContent = 'مقاطع داعمة من المرجع النشط';
    } else {
        html.setAttribute('dir', 'ltr');
        html.setAttribute('lang', 'en');
        if (langBtn) langBtn.textContent = 'العربية';
        if (input && !isGenerating) input.placeholder = 'Ask a clinical question based on the reference...';
        if (headerTitle) headerTitle.textContent = 'Evidence Conversation Workspace';
        if (evidenceTitle) evidenceTitle.textContent = 'Supporting Reference Passages';
    }
}

function toggleLang() {
    currentLang = (currentLang === 'en') ? 'ar' : 'en';
    applyLanguageUI();
    saveSessionState();
}

function applySuggestion(text) {
    if (isGenerating) return;
    const input = document.getElementById('questionInput');
    input.value = text;
    handleSendAction();
}

function formatMarkdown(text) {
    if (!text) return "";

    let formatted = text
        .replace(/^### (.*$)/gim, '<h4 class="ai-h4">$1</h4>')
        .replace(/^## (.*$)/gim, '<h3 class="ai-h3">$1</h3>')
        .replace(/^# (.*$)/gim, '<h2 class="ai-h2">$1</h2>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/^\s*[\*\-]\s+(.*$)/gim, '<li class="ai-li">$1</li>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');

    formatted = formatted.replace(/(<li class="ai-li">.*<\/li>)/gs, '<ul>$1</ul>');
    return formatted;
}

function handleSendAction() {
    if (isGenerating) {
        if (abortController) {
            abortController.abort();
        }
    } else {
        handleSend();
    }
}

async function handleSend() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    if (!question || question.length < 4 || isGenerating) return;

    const isArabicQuestion = /[\u0600-\u06FF]/.test(question);

    isGenerating = true;
    updateButtonUI(true, isArabicQuestion);

    appendUserMessage(question);
    input.value = '';

    const loadingId = 'loading-' + Date.now();
    appendLoading(loadingId, isArabicQuestion);

    abortController = new AbortController();

    try {
        const res = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
            signal: abortController.signal
        });

        if (!res.ok) throw new Error('Server response error');

        const data = await res.json();
        removeElement(loadingId);

        appendAiMessage(data, isArabicQuestion);
        renderEvidence(data.evidence_excerpts, isArabicQuestion, true);
        renderInspector(data, isArabicQuestion, true);

        saveSessionState(data.evidence_excerpts, data, isArabicQuestion);
    } catch (err) {
        removeElement(loadingId);
        if (err.name === 'AbortError') {
            const list = document.getElementById('messagesList');
            const div = document.createElement('div');
            div.className = 'msg-bubble msg-ai';
            div.style.opacity = '0.7';
            div.textContent = (isArabicQuestion || currentLang === 'ar') ? 'تم إيقاف التوليد بواسطة المستخدم.' : 'Generation stopped by user.';
            list.appendChild(div);
            list.scrollTop = list.scrollHeight;
            saveSessionState();
        } else {
            appendErrorMessage(isArabicQuestion);
            saveSessionState();
        }
    } finally {
        isGenerating = false;
        abortController = null;
        updateButtonUI(false, isArabicQuestion);
    }
}

function updateButtonUI(generating, isAr) {
    const btn = document.getElementById('sendBtn');
    const input = document.getElementById('questionInput');
    if (!btn || !input) return;

    if (generating) {
        btn.classList.add('stop-mode');
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>`;
        btn.setAttribute('aria-label', 'Stop generation');
        input.disabled = true;
        input.placeholder = (isAr || currentLang === 'ar') ? "جاري التوليد... اضغط على المربع للإيقاف" : "Generating... Click stop icon to abort.";
    } else {
        btn.classList.remove('stop-mode');
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>`;
        btn.setAttribute('aria-label', 'Send query');
        input.disabled = false;
        input.placeholder = (currentLang === 'ar') ? "اطرح سؤالك السريري حول المرجع..." : "Ask a clinical question based on the reference...";
        input.focus();
    }
}

function appendUserMessage(text) {
    const list = document.getElementById('messagesList');
    if (!list) return;
    const div = document.createElement('div');
    div.className = 'msg-bubble msg-user';
    div.textContent = text;
    list.appendChild(div);
    list.scrollTop = list.scrollHeight;
    saveSessionState();
}

function appendAiMessage(data, isArabicQuestion) {
    const list = document.getElementById('messagesList');
    if (!list) return;
    const div = document.createElement('div');
    div.className = 'msg-bubble msg-ai';

    let tagText = "High Confidence";
    if (isArabicQuestion || currentLang === 'ar') {
        tagText = data.confidence === 'high' ? 'مستوى الثقة: مرتفع' : (data.confidence === 'medium' ? 'مستوى الثقة: متوسط' : 'دليل غير كافٍ');
    } else {
        tagText = data.confidence === 'high' ? 'High Confidence' : (data.confidence === 'medium' ? 'Moderate Confidence' : 'Insufficient Evidence');
    }

    const cleanHtml = formatMarkdown(data.recommendation);

    div.innerHTML = `
        <span class="confidence-tag">${tagText}</span>
        <div class="ai-rendered-content">${cleanHtml}</div>
    `;
    list.appendChild(div);
    list.scrollTop = list.scrollHeight;
}

function renderEvidence(evidenceList, isArabicQuestion, shouldSave = true) {
    const container = document.getElementById('evidenceListContainer');
    if (!container) return;
    const isAr = isArabicQuestion || currentLang === 'ar';

    if (!evidenceList || evidenceList.length === 0) {
        container.innerHTML = `<div style="font-size:12px; color:var(--text-muted); text-align:center; margin-top:30px;">
            ${isAr ? 'الإجابة لم تعتمد على اقتباس قابل للعرض.' : 'No direct quotes available for this query.'}
        </div>`;
        return;
    }

    let html = '';
    evidenceList.forEach((item) => {
        html += `
            <div class="evidence-card">
                <div class="evidence-card-header">
                    <span>${item.document || 'Patellofemoral Pain CPG'}</span>
                    <span>${isAr ? 'صفحة ' + item.page : 'Page ' + item.page}</span>
                </div>
                <div style="font-size:11px; color:#cbd5e1; font-weight:600;">${item.section || 'General'}</div>
                <div class="evidence-quote">"${item.quote}"</div>
            </div>
        `;
    });
    container.innerHTML = html;
    if (shouldSave) {
        saveSessionState(evidenceList, null, isArabicQuestion);
    }
}

function renderInspector(data, isArabicQuestion, shouldSave = true) {
    const container = document.getElementById('inspectorBody');
    if (!container || !data) return;
    const isAr = isArabicQuestion || currentLang === 'ar';

    let diagnosticHtml = '';
    if (data.diagnostic) {
        diagnosticHtml = `
            <div class="inspector-item">
                <span class="inspector-key">Pipeline Layer:</span>
                <span class="inspector-val highlight">${data.diagnostic.layer || 'complete'}</span>
            </div>
            <div class="inspector-item">
                <span class="inspector-key">Status Code:</span>
                <span class="inspector-val">${data.diagnostic.code || 'ready'}</span>
            </div>
        `;
    }

    let retrievedHtml = '';
    if (Array.isArray(data._retrieved) && data._retrieved.length > 0) {
        retrievedHtml = '<div class="inspector-retrieved-list">';
        data._retrieved.forEach((doc, idx) => {
            const rank = idx + 1;
            retrievedHtml += `
                <div class="inspector-chunk">
                    <div class="inspector-chunk-head">
                        <span>#${rank} • Page ${doc.page}</span>
                        <span class="inspector-hash">${doc.content_hash || ''}</span>
                    </div>
                    <div class="inspector-chunk-body">${doc.original_text ? doc.original_text.substring(0, 180) + '...' : ''}</div>
                </div>
            `;
        });
        retrievedHtml += '</div>';
    } else {
        retrievedHtml = `<div class="inspector-empty">${isAr ? 'لا توجد مصفوفة استرجاع مرفقة.' : 'No raw retrieval array returned.'}</div>`;
    }

    container.innerHTML = `
        <div class="inspector-meta-grid">
            <div class="inspector-item">
                <span class="inspector-key">Retriever:</span>
                <span class="inspector-val">${data.retriever || 'hybrid_rrf'}</span>
            </div>
            <div class="inspector-item">
                <span class="inspector-key">Response Mode:</span>
                <span class="inspector-val">${data.response_mode || 'standard'}</span>
            </div>
            ${diagnosticHtml}
        </div>
        <div class="inspector-subhead">${isAr ? 'المقاطع المسترجعة مرتبة (Derived Rank)' : 'Retrieved Evidence Order'}</div>
        ${retrievedHtml}
    `;
    if (shouldSave) {
        saveSessionState(null, data, isArabicQuestion);
    }
}

function toggleInspector() {
    const details = document.getElementById('answerInspectorDetails');
    if (details) {
        details.open = !details.open;
    }
}

function appendLoading(id, isArabicQuestion) {
    const list = document.getElementById('messagesList');
    if (!list) return;
    const div = document.createElement('div');
    div.id = id;
    div.className = 'msg-bubble msg-ai';
    div.style.opacity = '0.6';
    div.textContent = (isArabicQuestion || currentLang === 'ar') ? 'جاري استخراج الأدلة من المرجع الطبي...' : 'Retrieving clinical evidence from guidelines...';
    list.appendChild(div);
    list.scrollTop = list.scrollHeight;
}

function appendErrorMessage(isArabicQuestion) {
    const list = document.getElementById('messagesList');
    if (!list) return;
    const div = document.createElement('div');
    div.className = 'msg-bubble msg-ai';
    div.textContent = (isArabicQuestion || currentLang === 'ar') ? 'نواجه صعوبة في الاتصال بالخدمة حالياً.' : 'Unable to connect to the clinical service.';
    list.appendChild(div);
}

function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function openUploadModal() {
    const modal = document.getElementById('uploadModal');
    if (modal) modal.style.display = 'flex';
}

function closeUploadModal() {
    const modal = document.getElementById('uploadModal');
    const status = document.getElementById('uploadStatus');
    if (modal) modal.style.display = 'none';
    if (status) status.textContent = '';
}

async function submitPdf() {
    const fileInput = document.getElementById('pdfFileInput');
    const statusDiv = document.getElementById('uploadStatus');
    if (!fileInput || !fileInput.files[0]) return;

    const file = fileInput.files[0];
    const reader = new FileReader();

    statusDiv.style.color = '#38bdf8';
    statusDiv.textContent = 'Validating and embedding reference...';

    reader.onload = async function() {
        const base64Content = reader.result.split(',')[1];
        try {
            const res = await fetch('/api/source-pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: file.name,
                    contentBase64: base64Content
                })
            });
            const result = await res.json();

            if (result.accepted) {
                const activeDoc = document.getElementById('activeDocName');
                if (activeDoc) activeDoc.textContent = result.document;
                statusDiv.style.color = '#34d399';
                statusDiv.textContent = 'Reference ready and active!';
                setTimeout(closeUploadModal, 1200);
            } else if (result.code === 'not_pfp_topic') {
                statusDiv.style.color = '#f87171';
                statusDiv.textContent = 'File rejected. Only Patellofemoral Pain (PFP) references are accepted.';
            } else {
                statusDiv.style.color = '#f87171';
                statusDiv.textContent = 'Processing error. Please provide a valid PDF.';
            }
        } catch (e) {
            statusDiv.style.color = '#f87171';
            statusDiv.textContent = 'Upload failed.';
        }
    };
    reader.readAsDataURL(file);
}

document.getElementById('questionInput')?.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendAction();
    }
});