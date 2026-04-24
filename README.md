# VisionWords – Image Captioning
### Xception + Attention LSTM · Flickr8k · TensorFlow/Keras

---

## Two-Repo Workflow

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  Google Colab  (training)   │        │  VS Code / Laptop  (app)     │
│                             │        │                              │
│  ImageCaptioning_Colab.ipynb│──3 ──▶│  captioning_model.keras      │
│                             │  files │  tokenizer.pkl               │
│  trains on T4 GPU (~15 min) │        │  config.pkl                  │
│  saves to Google Drive      │        │                              │
│  downloads to your laptop   │        │  streamlit run app.py  🚀    │
└─────────────────────────────┘        └──────────────────────────────┘
```

---

## Step-by-Step Setup

### Part A — Train in Google Colab

1. Open `ImageCaptioning_Colab.ipynb` in Google Colab
2. Set runtime: Runtime → Change runtime type → T4 GPU
3. Run all cells — last cell auto-downloads 3 files

### Part B — VS Code project structure

```
your-repo/
├── app.py
├── inference.py
├── requirements.txt
├── captioning_model.keras   ← from Colab
├── tokenizer.pkl            ← from Colab
└── config.pkl               ← from Colab
```

### Part C — Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
