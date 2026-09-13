/* =========================================================================
   PulmoScan — front-end logic
   Everything in the "PREDICTION API" block below is the only part that
   needs to change to connect a real PyTorch backend. Everything else
   (upload handling, preview, result rendering) already expects the same
   { label, confidence } shape a real API would return.
   ========================================================================= */

(function () {
  "use strict";

  // ---- DOM refs ----------------------------------------------------------
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzoneEmpty = document.getElementById("dropzoneEmpty");
  const dropzonePreview = document.getElementById("dropzonePreview");
  const previewImg = document.getElementById("previewImg");
  const scanOverlay = document.getElementById("scanOverlay");
  const fileInfo = document.getElementById("fileInfo");
  const fileNameEl = document.getElementById("fileName");
  const removeFileBtn = document.getElementById("removeFile");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const statusText = document.getElementById("statusText");

  const resultEmpty = document.getElementById("resultEmpty");
  const resultCard = document.getElementById("resultCard");
  const resultBadge = document.getElementById("resultBadge");
  const resultTime = document.getElementById("resultTime");
  const confidenceNum = document.getElementById("confidenceNum");
  const confidenceFill = document.getElementById("confidenceFill");
  const splitNormal = document.getElementById("splitNormal");
  const splitPneumonia = document.getElementById("splitPneumonia");

  let currentFile = null;
  let objectUrl = null;

  // ---- Upload handling ----------------------------------------------------
  function openFileDialog() { fileInput.click(); }

  dropzone.addEventListener("click", openFileDialog);
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openFileDialog(); }
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) handleFile(file);
  });

  removeFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      statusText.textContent = "Please choose an image file (JPG or PNG).";
      return;
    }
    currentFile = file;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);

    previewImg.src = objectUrl;
    dropzoneEmpty.hidden = true;
    dropzonePreview.hidden = false;

    fileInfo.hidden = false;
    fileNameEl.textContent = file.name;

    analyzeBtn.disabled = false;
    statusText.textContent = "Ready to analyze.";
    resetResult();
  }

  function resetUpload() {
    currentFile = null;
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    fileInput.value = "";
    dropzoneEmpty.hidden = false;
    dropzonePreview.hidden = true;
    fileInfo.hidden = true;
    analyzeBtn.disabled = true;
    statusText.textContent = "Upload an image to begin.";
    resetResult();
  }

  function resetResult() {
    resultCard.hidden = true;
    resultEmpty.hidden = false;
    confidenceFill.style.width = "0%";
  }

  // ---- Analyze action -------------------------------------------------------
  analyzeBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    analyzeBtn.disabled = true;
    statusText.textContent = "Analyzing…";
    scanOverlay.hidden = false;
    resetResult();

    try {
      const result = await predictXray(currentFile);
      renderResult(result);
      statusText.textContent = "Analysis complete.";
    } catch (err) {
      statusText.textContent = "Something went wrong analyzing this image. Please try again.";
      console.error(err);
    } finally {
      scanOverlay.hidden = true;
      analyzeBtn.disabled = false;
    }
  });

  function renderResult({ label, confidence }) {
    const isPneumonia = label === "PNEUMONIA";

    resultEmpty.hidden = true;
    resultCard.hidden = false;
    resultCard.classList.toggle("is-pneumonia", isPneumonia);
    resultCard.classList.toggle("is-normal", !isPneumonia);

    resultBadge.textContent = isPneumonia ? "Pneumonia" : "Normal";
    resultTime.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    const pct = (confidence * 100).toFixed(1);
    confidenceNum.textContent = pct + "%";

    const normalPct = isPneumonia ? 100 - confidence * 100 : confidence * 100;
    const pneumoniaPct = 100 - normalPct;
    splitNormal.textContent = normalPct.toFixed(1) + "%";
    splitPneumonia.textContent = pneumoniaPct.toFixed(1) + "%";

    // animate the bar on the next frame so the CSS transition actually runs
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { confidenceFill.style.width = pct + "%"; });
    });
  }

  // =========================================================================
  // PREDICTION API — swap this block out for a real backend call.
  //
  // Expected contract: predictXray(file) returns a Promise resolving to
  //   { label: "NORMAL" | "PNEUMONIA", confidence: number between 0 and 1 }
  //
  // Real implementation would look roughly like:
  //
  //   async function predictXray(file) {
  //     const formData = new FormData();
  //     formData.append("file", file);
  //     const res = await fetch("/api/predict", { method: "POST", body: formData });
  //     if (!res.ok) throw new Error("Prediction request failed");
  //     return res.json(); // { label, confidence }
  //   }
  //
  // The PyTorch side (see pneumonia_cnn.py) would load the saved checkpoint,
  // run the same preprocessing transform used at training time, and return
  // torch.sigmoid(logits) as `confidence` with the thresholded class as `label`.
  // =========================================================================
  async function predictXray(file) {
    await wait(1600 + Math.random() * 500);

    const isPneumonia = Math.random() > 0.45;
    const confidence = isPneumonia
      ? 0.78 + Math.random() * 0.19
      : 0.82 + Math.random() * 0.16;

    return { label: isPneumonia ? "PNEUMONIA" : "NORMAL", confidence };
  }

  function wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
})();
