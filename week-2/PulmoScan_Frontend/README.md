# PulmoScan — Pneumonia Detection Dashboard (Frontend)

Week 2 Task: Designing an AI Model Architecture
Project: AI-Based Pneumonia Detection from Chest X-Ray Images

## What this is

A static, self-contained frontend (no build step, no framework) for demonstrating
the pneumonia-detection model architecture:

- `index.html` — page structure and content
- `styles.css` — visual design (dark radiology-lightbox theme, responsive layout)
- `script.js` — upload handling, mock prediction, result rendering

Open `index.html` directly in any modern browser — nothing to install or build.

## Sections

1. **Hero** — project title and objective.
2. **Run a scan** (`#scan`) — drag-and-drop / click-to-browse X-ray upload,
   "Analyze X-Ray" button, and a result card with label + confidence score.
3. **Model architecture** (`#architecture`) — the 7-stage pipeline: X-Ray Input →
   Preprocessing → Data Augmentation → ResNet-18 → Global Average Pooling →
   Fully Connected Layer → Normal/Pneumonia.
4. **Technology** (`#technology`) — Python, PyTorch, ResNet-18, Transfer Learning.
5. **Design details** (`#details`) — Data Preprocessing, Algorithms, Benefits,
   Challenges, Future Scalability.

## Connecting a real backend

All prediction logic lives in one place: the `predictXray(file)` function near
the bottom of `script.js`. It currently returns a randomized mock result after
a short delay so the UI can be demoed end-to-end without a model running.

To connect a real PyTorch backend (e.g. an API built around `pneumonia_cnn.py`
from the Week 2 architecture report), replace only that function:

```js
async function predictXray(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/predict", { method: "POST", body: formData });
  if (!res.ok) throw new Error("Prediction request failed");
  return res.json(); // { label: "NORMAL" | "PNEUMONIA", confidence: 0-1 }
}
```

Nothing else in the file needs to change — the upload handling and result
rendering already expect exactly this `{ label, confidence }` shape.

## Notes

- No image is uploaded anywhere in this demo build; the preview is rendered
  entirely in the browser via `URL.createObjectURL`.
- Respects `prefers-reduced-motion`, has visible keyboard focus states, and is
  responsive from mobile widths up.
