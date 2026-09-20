// ============================================================================
// D2INV Agent — Phase 4 Conversational Web Client
// ============================================================================

const API_BASE = "http://localhost:8000/api";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const uploadBtn = document.getElementById("uploadBtn");
const datasetFile = document.getElementById("datasetFile");
const datasetSelect = document.getElementById("datasetSelect");
const refreshDatasetsBtn = document.getElementById("refreshDatasetsBtn");
const sessionInfo = document.getElementById("sessionInfo");

const progressSteps = document.querySelectorAll(".progress-step");
const tabButtons = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

const summaryPre = document.getElementById("summaryPre");
const summaryPlaceholder = document.getElementById("summaryPlaceholder");

const storyPre = document.getElementById("storyPre");

const templateFrame = document.getElementById("templateFrame");
const templatePlaceholder = document.getElementById("templatePlaceholder");

const invFrame = document.getElementById("invFrame");
const invPlaceholder = document.getElementById("invPlaceholder");

const evaluateFrame = document.getElementById("evaluateFrame");
const evaluatePlaceholder = document.getElementById("evaluatePlaceholder");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let sessionId = localStorage.getItem("d2inv_session_id") || "";
let isProcessing = false;
let currentThinkingId = null; // module-level holder for loading indicator

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    loadDatasetList();
    if (sessionId) {
        sessionInfo.textContent = sessionId.slice(0, 8) + "...";
        sessionInfo.title = sessionId;
    }
});

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------

sendBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

datasetSelect.addEventListener("change", () => {
    const val = datasetSelect.value;
    if (val) {
        chatInput.value = `summarize dataset ${val}`;
        chatInput.focus();
        // If this dataset already has generated results on disk, show them
        // right away without waiting for the agent round-trip.
        refreshPanelsFromResults(val);
    }
});

refreshDatasetsBtn.addEventListener("click", loadDatasetList);

datasetFile.addEventListener("change", async () => {
    const file = datasetFile.files[0];
    if (!file) return;

    addMessage("user", `📎 Uploaded: ${file.name}`);
    chatInput.value = `summarize dataset ${file.name}`;
});

// Tab switching
tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
        switchTab(btn.dataset.tab);
    });
});

// ---------------------------------------------------------------------------
// Send message to agent
// ---------------------------------------------------------------------------

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || isProcessing) return;

    addMessage("user", text);
    chatInput.value = "";
    isProcessing = true;
    sendBtn.disabled = true;

    // Show a temporary "thinking" indicator
    currentThinkingId = addThinkingIndicator();

    // Build the SSE URL
    const params = new URLSearchParams({ message: text });
    if (sessionId) params.set("session_id", sessionId);

    const url = `${API_BASE}/agent/chat_stream?${params.toString()}`;

    try {
        const eventSource = new EventSource(url);

        eventSource.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleAgentEvent(msg, currentThinkingId);
        };

        eventSource.onerror = (err) => {
            console.error("SSE error", err);
            eventSource.close();
            removeThinkingIndicator(currentThinkingId);
            addMessage("assistant", "⚠️ Connection error. Please try again.");
            finishProcessing();
        };

        // Store cleanup hook
        chatMessages._eventSource = eventSource;

    } catch (err) {
        removeThinkingIndicator(currentThinkingId);
        addMessage("assistant", `❌ Error: ${err.message}`);
        finishProcessing();
    }
}

// ---------------------------------------------------------------------------
// Handle streaming agent events
// ---------------------------------------------------------------------------

function handleAgentEvent(msg, thinkingId) {
    const eventType = msg.event;
    const data = msg.data || {};

    switch (eventType) {
        case "step":
            removeThinkingIndicator(currentThinkingId);
            addMessage("assistant-thinking", `💭 ${data.thought || "Thinking..."}`);
            if (data.action) {
                const stage = mapToolToStage(data.action);
                if (stage >= 0) updateProgress(stage, "active");
            }
            break;

        case "tool_result":
            if (data.success) {
                const stage = mapToolToStage(data.tool);
                if (stage >= 0) updateProgress(stage, "done");

                // Route HTML/content to the correct preview panel & auto-switch
                if (data.data) {
                    handleToolData(data.tool, data.data);
                }
                addMessage("tool-ok", `✓ ${data.tool} succeeded`);
            } else {
                addMessage("tool-fail", `✗ ${data.tool} failed: ${data.error || ""}`);
            }

            // Show a thinking indicator while waiting for the planner's response
            currentThinkingId = addThinkingIndicator();
            break;

        case "done":
            removeThinkingIndicator(currentThinkingId);
            addMessage("assistant", data.answer || "Done!");
            if (data.answer && data.answer.match(/^<(!doctype|<html)/i)) {
                renderINV(data.answer);
            }
            finalizeStream();
            finishProcessing();
            break;

        case "error":
            removeThinkingIndicator(currentThinkingId);
            addMessage("assistant-error", `❌ ${data.message}`);
            finalizeStream();
            finishProcessing();
            break;

        default:
            console.log("Unknown event:", eventType, data);
    }
}

// ---------------------------------------------------------------------------
// Route tool-generated data into preview panels (and auto-switch tabs)
// ---------------------------------------------------------------------------

function handleToolData(toolName, data) {
    if (typeof data !== "object") return;

    // Refresh whatever panels we can from the on-disk results/<dataset> folder.
    // This is robust: results are written by each pipeline stage, so for the
    // current dataset we can always read them directly instead of relying on
    // in-memory session state (which may be empty on the first message).
    if (data.dataset_name) {
        refreshPanelsFromResults(data.dataset_name);
    }

    switch (toolName) {
        case "summarize_dataset":
            // Show column-level statistics in the Data Summary panel
            summaryPre.textContent = JSON.stringify(data, null, 2);
            summaryPlaceholder.classList.add("hidden");
            summaryPre.classList.remove("hidden");
            switchTab("summary");
            break;

        case "generate_data_story":
        case "generate_story":
            switchTab("story");
            break;

        case "generate_template":
            switchTab("template");
            break;

        case "generate_charts":
        case "assemble_inv":
            switchTab("inv");
            break;

        case "evaluate_inv":
        case "evaluate":
            switchTab("evaluate");
            break;

        default:
            // Fallback: if data has html, render to INV panel
            if (data.html) renderINV(data.html);
    }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function addMessage(role, content) {
    const div = document.createElement("div");
    div.className = `chat-message chat-${role}`;
    div.innerHTML = `<p>${escapeHTML(content)}</p>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function addThinkingIndicator() {
    const id = `thinking-${Date.now()}`;
    const div = document.createElement("div");
    div.id = id;
    div.className = "chat-message chat-thinking";
    div.innerHTML = `<p class="thinking-dots">Thinking<span>.</span><span>.</span><span>.</span></p>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function removeThinkingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function finishProcessing() {
    isProcessing = false;
    sendBtn.disabled = false;
    chatInput.focus();
}

function finalizeStream() {
    if (chatMessages._eventSource) {
        chatMessages._eventSource.close();
        delete chatMessages._eventSource;
    }
}

function renderINV(html) {
    invFrame.srcdoc = html;
    invPlaceholder.classList.add("hidden");
    invFrame.classList.remove("hidden");
    switchTab("inv");
    notifyINVVisible();
}

// Ask the INV iframe to re-layout its charts now that it is visible. Charts
// initialized while the preview was still hidden measure 0x0 and stay blank
// until resized, so this must be sent after the panel is actually shown.
function notifyINVVisible() {
    setTimeout(() => {
        try {
            invFrame.contentWindow.postMessage({ type: "D2INV_RESIZE" }, "*");
        } catch (e) {
            /* iframe not loaded yet; the bootstrap's own resize listener handles it */
        }
    }, 50);
}

// Fetch the latest generated pipeline artifacts for a dataset straight from
// the on-disk results/<dataset>/ folder and refresh the preview panels.
// Streamed tool_result events only carry truncated summaries, so this reads
// the full outputs (data story JSON, HTML template, INV, evaluation) directly.
function refreshPanelsFromResults(datasetName) {
    if (!datasetName) return;
    const url = `${API_BASE}/results/artifacts?dataset=${encodeURIComponent(datasetName)}`;
    fetch(url)
        .then((res) => res.json())
        .then((artifacts) => {
            if (artifacts.error) {
                console.warn("refreshPanelsFromResults:", artifacts.error);
                return;
            }
            if (artifacts.data_summary) {
                summaryPre.textContent = JSON.stringify(artifacts.data_summary, null, 2);
                summaryPlaceholder.classList.add("hidden");
                summaryPre.classList.remove("hidden");
            }
            if (artifacts.data_story) {
                storyPre.textContent = JSON.stringify(artifacts.data_story, null, 2);
            }
            if (artifacts.html_template) {
                templateFrame.srcdoc = artifacts.html_template;
                templatePlaceholder.classList.add("hidden");
                templateFrame.classList.remove("hidden");
            }
            if (artifacts.inv) {
                invFrame.srcdoc = artifacts.inv;
                invPlaceholder.classList.add("hidden");
                invFrame.classList.remove("hidden");
            }
            if (artifacts.evaluation) {
                evaluateFrame.srcdoc = artifacts.evaluation;
                evaluatePlaceholder.classList.add("hidden");
                evaluateFrame.classList.remove("hidden");
            }
        })
        .catch((err) => console.warn("refreshPanelsFromResults failed:", err));
}

function switchTab(tabName) {
    tabButtons.forEach(b => {
        b.classList.toggle("active", b.dataset.tab === tabName);
    });
    tabContents.forEach(c => {
        c.classList.toggle("active", c.id === `tab-${tabName}`);
    });
    if (tabName === "inv" && !invFrame.classList.contains("hidden")) {
        notifyINVVisible();
    }
}

function updateProgress(stage, status) {
    const stepEl = document.querySelector(`.progress-step[data-step="${stage + 1}"]`);
    if (!stepEl) return;
    stepEl.classList.remove("active", "done");
    if (status === "active") stepEl.classList.add("active");
    if (status === "done") stepEl.classList.add("done");
}

function mapToolToStage(toolName) {
    const map = {
        "summarize_dataset": 0,
        "generate_data_story": 1,
        "generate_story": 1,
        "generate_template": 2,
        "generate_charts": 3,
        "assemble_inv": 3,
        "evaluate_inv": 4,
        "evaluate": 4,
    };
    return map[toolName] ?? -1;
}

// ---------------------------------------------------------------------------
// Dataset list loader
// ---------------------------------------------------------------------------

async function loadDatasetList() {
    try {
        const res = await fetch(`${API_BASE}/list_datasets`);
        const files = await res.json();
        datasetSelect.innerHTML = '<option value="">📁 Pick a dataset...</option>';
        files.sort().forEach(f => {
            const opt = document.createElement("option");
            opt.value = f;
            opt.textContent = f;
            datasetSelect.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to load datasets", err);
    }
}

// Initial load
loadDatasetList();

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

async function createNewSession() {
    sessionId = "";
    localStorage.removeItem("d2inv_session_id");
    sessionInfo.textContent = "";
    sessionInfo.title = "";
    chatMessages.innerHTML = "";
    resetUI();
}

function resetUI() {
    document.querySelectorAll(".progress-step").forEach(s => s.classList.remove("active", "done"));
    invFrame.srcdoc = "";
    invFrame.classList.add("hidden");
    invPlaceholder.classList.remove("hidden");
    evaluateFrame.srcdoc = "";
    evaluateFrame.classList.add("hidden");
    evaluatePlaceholder.classList.remove("hidden");
    templateFrame.srcdoc = "";
    templateFrame.classList.add("hidden");
    templatePlaceholder.classList.remove("hidden");
    summaryPre.classList.add("hidden");
    summaryPlaceholder.classList.remove("hidden");
    storyPre.textContent = "";
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}