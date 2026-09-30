/**
 * LAGOM HUMANIZER — MINIMAL FRONTEND LOGIC
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const apiUrlInput = document.getElementById("apiUrlInput");
  const apiStatusBadge = document.getElementById("apiStatusBadge");
  const apiStatusText = document.getElementById("apiStatusText");
  const settingsToggleBtn = document.getElementById("settingsToggleBtn");
  const settingsDrawer = document.getElementById("settingsDrawer");

  const tempSlider = document.getElementById("tempSlider");
  const tempVal = document.getElementById("tempVal");
  const maxTokensSlider = document.getElementById("maxTokensSlider");
  const maxTokensVal = document.getElementById("maxTokensVal");
  const streamToggle = document.getElementById("streamToggle");

  const stylePillsContainer = document.getElementById("stylePillsContainer");
  const styleDescription = document.getElementById("styleDescription");

  const inputText = document.getElementById("inputText");
  const inputWordCount = document.getElementById("inputWordCount");
  const inputCharCount = document.getElementById("inputCharCount");
  const clearInputBtn = document.getElementById("clearInputBtn");
  const pasteInputBtn = document.getElementById("pasteInputBtn");

  const humanizeBtn = document.getElementById("humanizeBtn");
  const btnText = document.getElementById("btnText");

  const outputPlaceholder = document.getElementById("outputPlaceholder");
  const outputText = document.getElementById("outputText");
  const outputDiff = document.getElementById("outputDiff");
  const outputWordCount = document.getElementById("outputWordCount");
  const outputReadingTime = document.getElementById("outputReadingTime");
  const statModelMode = document.getElementById("statModelMode");
  const copyOutputBtn = document.getElementById("copyOutputBtn");
  const copyText = document.getElementById("copyText");
  const compareDiffBtn = document.getElementById("compareDiffBtn");
  const toast = document.getElementById("toast");
  const modelBanner = document.getElementById("modelBanner");

  // State
  let currentStyle = "general";
  let isGenerating = false;
  let lastRawOutput = "";
  let isDiffView = false;
  let modelReady = false;

  const STYLE_DESCRIPTIONS = {
    general: "Rewrite this text into natural, organic human writing. Remove artificial AI cadence, repetitive transitions, and generic over-explanations while keeping the meaning intact.",
    essay: "Rewrite this essay passage in the voice of a skilled human writer. Vary the sentence rhythm naturally, remove formulaic signposts (e.g. 'furthermore', 'moreover', 'in conclusion'), and preserve genuine thesis flow without repetitive summary statements.",
    academic: "Rewrite this academic passage in genuine, human-authored scholarly prose. Use precise, substantive vocabulary without artificial buzzwords or robotic hedging patterns.",
    email: "Rewrite this email to sound like an authentic human professional. Use warm, natural greetings and sign-offs, realistic phrasing, and eliminate stiff corporate AI cliches.",
    document: "Rewrite this document/report in clean, natural human report style. Ensure sharp readability, logical structural hierarchy, and authentic business prose.",
  };

  const SAMPLES = {
    essay: "Furthermore, it is widely acknowledged that technological evolution plays a pivotal role in modern sociological development. Moreover, one must consider that artificial algorithms inevitably influence human daily routines. In conclusion, it is evident that continuous assessment of computational advancements remains crucial for future prosperity.",
    academic: "It is important to note that the empirical investigation delved deeply into the multi-faceted dynamics of algorithmic learning. The paradigm shift underscores a profound testament to cognitive mechanisms, thereby substantiating the overarching theoretical framework with notable precision.",
    email: "I hope this email finds you well. I am reaching out to proactively touch base regarding our forthcoming deliverables. Delving into the project roadmap, it is imperative that we align on key milestones to facilitate seamless synergy across departments. Looking forward to your prompt response.",
  };

  // Restore API URL from localStorage
  const savedUrl = localStorage.getItem("lagom_api_url");
  if (savedUrl) apiUrlInput.value = savedUrl;

  function getBaseUrl() {
    return (apiUrlInput.value || "http://127.0.0.1:8000").trim().replace(/\/+$/, "");
  }

  // ── Model readiness ─────────────────────────────────────────────────────────
  // Disables the button and shows a banner until training is done.
  // The health-check loop re-enables everything automatically once the model
  // appears in outputs/adapter or outputs/merged.

  function setModelReady(ready) {
    modelReady = ready;
    if (ready) {
      if (modelBanner) modelBanner.classList.add("hidden");
      if (!isGenerating) humanizeBtn.disabled = false;
      humanizeBtn.title = "";
    } else {
      if (modelBanner) modelBanner.classList.remove("hidden");
      if (!isGenerating) humanizeBtn.disabled = true;
      humanizeBtn.title = "Waiting for model — complete training first";
    }
  }

  // ── Health check ─────────────────────────────────────────────────────────
  async function checkHealth() {
    apiStatusBadge.className = "status-indicator checking";
    apiStatusText.textContent = "connecting...";

    try {
      const res = await fetch(`${getBaseUrl()}/api/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error("Status " + res.status);
      const data = await res.json();

      if (data.model_loaded) {
        apiStatusBadge.className = "status-indicator online";
        apiStatusText.textContent = `online · ${data.model_type}`;
        if (!isGenerating) statModelMode.textContent = `local_${data.model_type}`;
        setModelReady(true);
      } else {
        apiStatusBadge.className = "status-indicator pending";
        apiStatusText.textContent = "api up · awaiting model";
        setModelReady(false);
      }
    } catch {
      apiStatusBadge.className = "status-indicator offline";
      apiStatusText.textContent = "offline";
      setModelReady(false);
    }
  }

  // Poll every 15 s — UI unlocks automatically the moment training finishes
  checkHealth();
  setInterval(checkHealth, 15000);

  // ── Settings ─────────────────────────────────────────────────────────────
  settingsToggleBtn.addEventListener("click", () => {
    settingsDrawer.classList.toggle("hidden");
  });
  apiUrlInput.addEventListener("change", () => {
    localStorage.setItem("lagom_api_url", apiUrlInput.value);
    checkHealth();
  });
  tempSlider.addEventListener("input", (e) => {
    tempVal.textContent = parseFloat(e.target.value).toFixed(2);
  });
  maxTokensSlider.addEventListener("input", (e) => {
    maxTokensVal.textContent = e.target.value;
  });

  // ── Style selector ─────────────────────────────────────────────────────────
  stylePillsContainer.addEventListener("click", (e) => {
    const tab = e.target.closest(".style-tab, .style-pill");
    if (!tab) return;
    document.querySelectorAll(".style-tab, .style-pill").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    currentStyle = tab.dataset.style;
    styleDescription.textContent = STYLE_DESCRIPTIONS[currentStyle] || STYLE_DESCRIPTIONS.general;
  });

  // ── Word counters ──────────────────────────────────────────────────────────
  function countWords(str) {
    const trimmed = str.trim();
    return trimmed ? trimmed.split(/\s+/).length : 0;
  }

  function updateInputStats() {
    const text = inputText.value;
    const words = countWords(text);
    inputWordCount.textContent = `${words} word${words === 1 ? "" : "s"}`;
    inputCharCount.textContent = `${text.length} character${text.length === 1 ? "" : "s"}`;
  }

  function updateOutputStats(text) {
    const words = countWords(text);
    outputWordCount.textContent = `${words} word${words === 1 ? "" : "s"}`;
    const minutes = Math.max(1, Math.round(words / 200));
    outputReadingTime.textContent = words ? `${minutes} min read` : "0 min read";
  }

  inputText.addEventListener("input", updateInputStats);

  // ── Clear & paste ──────────────────────────────────────────────────────────
  clearInputBtn.addEventListener("click", () => {
    inputText.value = "";
    updateInputStats();
    inputText.focus();
  });

  pasteInputBtn.addEventListener("click", async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        inputText.value = text;
        updateInputStats();
      }
    } catch {
      showToast("Clipboard access denied. Press Ctrl+V instead.");
    }
  });

  // ── Sample chips ───────────────────────────────────────────────────────────
  document.querySelectorAll(".sample-link, .sample-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const sampleKey = chip.dataset.sample;
      if (SAMPLES[sampleKey]) {
        inputText.value = SAMPLES[sampleKey];
        updateInputStats();
        const matchingTab = document.querySelector(`[data-style="${sampleKey}"]`);
        if (matchingTab) matchingTab.click();
      }
    });
  });

  // ── Toast ──────────────────────────────────────────────────────────────────
  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.remove("hidden");
    setTimeout(() => toast.classList.add("hidden"), 2800);
  }

  // ── Copy output ────────────────────────────────────────────────────────────
  copyOutputBtn.addEventListener("click", async () => {
    if (!lastRawOutput) return;
    try {
      await navigator.clipboard.writeText(lastRawOutput);
      copyText.textContent = "Copied ✓";
      showToast("Copied humanized text to clipboard");
      setTimeout(() => { copyText.textContent = "copy"; }, 2000);
    } catch {
      showToast("Failed to copy");
    }
  });

  // ── Diff view ──────────────────────────────────────────────────────────────
  function renderDiff(original, humanized) {
    const origSet = new Set(
      original.trim().split(/\s+/).map((w) => w.toLowerCase().replace(/[^\w]/g, ""))
    );
    return humanized
      .trim()
      .split(/\s+/)
      .map((word) => {
        const clean = word.toLowerCase().replace(/[^\w]/g, "");
        return !origSet.has(clean) && clean.length > 2
          ? `<span class="diff-tag-human">${word}</span>`
          : word;
      })
      .join(" ");
  }

  compareDiffBtn.addEventListener("click", () => {
    if (!lastRawOutput) return;
    isDiffView = !isDiffView;
    if (isDiffView) {
      outputText.classList.add("hidden");
      outputDiff.classList.remove("hidden");
      outputDiff.innerHTML = renderDiff(inputText.value, lastRawOutput);
      compareDiffBtn.textContent = "Raw Text";
    } else {
      outputDiff.classList.add("hidden");
      outputText.classList.remove("hidden");
      compareDiffBtn.textContent = "Diff";
    }
  });

  // ── Core humanize action ───────────────────────────────────────────────────
  async function runHumanize() {
    const text = inputText.value.trim();
    if (!text) {
      showToast("Please enter or paste AI text first.");
      inputText.focus();
      return;
    }
    if (!modelReady) {
      showToast("Model not ready — finish training first.");
      return;
    }
    if (isGenerating) return;

    isGenerating = true;
    humanizeBtn.disabled = true;
    btnText.textContent = "humanizing...";

    outputPlaceholder.classList.add("hidden");
    outputDiff.classList.add("hidden");
    outputText.classList.remove("hidden");
    outputText.textContent = "";
    outputText.classList.add("streaming");
    lastRawOutput = "";
    compareDiffBtn.textContent = "Diff";
    isDiffView = false;
    statModelMode.textContent = "generating...";

    const payload = {
      text,
      style: currentStyle,
      temperature: parseFloat(tempSlider.value),
      max_new_tokens: parseInt(maxTokensSlider.value, 10),
    };

    const isStreaming = streamToggle.checked;
    const startTime = performance.now();

    try {
      if (isStreaming) {
        const response = await fetch(`${getBaseUrl()}/api/humanize/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (response.status === 503) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.detail || "model_not_ready");
        }
        if (!response.ok) throw new Error("HTTP Error " + response.status);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.token) {
                  lastRawOutput += data.token;
                  outputText.textContent = lastRawOutput;
                  updateOutputStats(lastRawOutput);
                }
                if (data.mode) statModelMode.textContent = data.mode;
              } catch {}
            }
          }
        }
      } else {
        const response = await fetch(`${getBaseUrl()}/api/humanize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (response.status === 503) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.detail || "model_not_ready");
        }
        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.detail || "Request failed");
        }

        const data = await response.json();
        lastRawOutput = data.humanized;
        outputText.textContent = lastRawOutput;
        statModelMode.textContent = `${data.model_mode} (${data.elapsed_seconds}s)`;
        updateOutputStats(lastRawOutput);
      }

      const totalSeconds = ((performance.now() - startTime) / 1000).toFixed(2);
      statModelMode.textContent += ` • ${totalSeconds}s`;

    } catch (err) {
      const isNotReady =
        err.message.includes("model_not_ready") ||
        err.message.toLowerCase().includes("not ready") ||
        err.message.toLowerCase().includes("training");

      if (isNotReady) {
        // Reset output pane to waiting state — no garbage text
        outputText.textContent = "";
        outputText.classList.add("hidden");
        outputPlaceholder.classList.remove("hidden");
        statModelMode.textContent = "awaiting training";
        setModelReady(false);
        showToast("Model training not complete — run the training script first.");
      } else {
        outputText.textContent =
          `[Connection Error]: ${err.message}\n` +
          `Make sure the API server is running on ${getBaseUrl()} — run 'start.bat' in the api/ folder.`;
        statModelMode.textContent = "error";
      }
    } finally {
      outputText.classList.remove("streaming");
      isGenerating = false;
      if (modelReady) humanizeBtn.disabled = false;
      btnText.textContent = "Humanize";
    }
  }

  humanizeBtn.addEventListener("click", runHumanize);

  // Keyboard shortcut: Ctrl/Cmd + Enter
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      runHumanize();
    }
  });
});
