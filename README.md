---
title: Concrete Anomaly Segregation Detection
emoji: 📚
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.52.2
app_file: app.py
python_version: '3.10'
pinned: false
license: mit
short_description: An AI-powered computer vision platform designed for structur
---

# 🏗️ Concrete Anomaly Inspector

A Gradio web app that runs a full 5-stage concrete defect detection pipeline on a single uploaded image.

**Pipeline stages:**

1. **Roboflow** → detect concrete area → crop ROI (fallback: full image)
2. **ResNet-18** → classify: `crack | crack\_segregation | segregation | normal`
3. **U-Net** → pixel-level binary anomaly mask *(segregation classes only)*
4. **Feature extraction** → GAI (Global Anomaly Index) + SI\_ia (Segregation Index)
5. **Stage assignment** → KMeans (primary) + rule-based (reference)
6. **Inspection card** → 4-panel or 2-panel summary figure

\---

## 📁 Project structure

```
concrete\_inspector/
├── app.py               ← Gradio application (main entry point)
├── requirements.txt     ← Python dependencies
├── .env.example         ← Template for environment variables
├── .env                 ← Your secrets (NOT committed, see .gitignore)
├── .gitignore
├── README.md
└── saved\_models/        ← Put your model files here
    ├── best\_unet.pth
    ├── best\_classifier.pth
    ├── kmeans\_stages.pkl
    ├── kmeans\_scaler.pkl
    └── kmeans\_rank\_map.json
```

\---

## 🖥️ Run locally

### 1 — Clone / download the project

```bash
git clone <your-repo-url>
cd concrete\_inspector
```

### 2 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\\Scripts\\activate         # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

If you want Roboflow ROI detection, also install:

```bash
pip install inference supervision
```

### 4 — Add your model files

Copy your trained weights into the `saved\_models/` folder:

```
saved\_models/
├── best\_unet.pth
├── best\_classifier.pth
├── kmeans\_stages.pkl
├── kmeans\_scaler.pkl
└── kmeans\_rank\_map.json
```

### 5 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set:

|Variable|Description|Default|
|-|-|-|
|`MODELS\_DIR`|Path to the folder with your model files|`./saved\_models/`|
|`ROBOFLOW\_API\_KEY`|Your Roboflow API key (leave blank to skip)|*(blank)*|
|`ROBOFLOW\_MODEL\_ID`|Roboflow model identifier|`concrete\_detection/6`|
|`SEG\_THRESHOLD`|U-Net binarisation threshold|`0.90`|
|`CONF\_THRESHOLD`|Roboflow detection confidence threshold|`0.90`|
|`SHOW\_MASK`|Overlay anomaly mask on image|`true`|

### 6 — Launch the app

```bash
python app.py
```

Open your browser at **http://127.0.0.1:7860**.

\---

## 🚀 Deploy to Hugging Face Spaces

### Step 1 — Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Gradio** as the SDK.
3. Select a hardware tier (CPU Basic is free; use GPU for faster inference).
4. Click **Create Space**.

### Step 2 — Push your code

```bash
# Install git-lfs (needed for large model files)
git lfs install

# Clone your new Space repo
git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
cd <your-space-name>

# Copy project files (everything EXCEPT .env and saved\_models/)
cp /path/to/concrete\_inspector/app.py .
cp /path/to/concrete\_inspector/requirements.txt .
cp /path/to/concrete\_inspector/README.md .

# Track model files with Git LFS before adding
git lfs track "\*.pth" "\*.pkl"
git add .gitattributes

# Copy and add model files
mkdir -p saved\_models
cp /path/to/saved\_models/\* saved\_models/
git add saved\_models/

# Add everything else
git add app.py requirements.txt README.md
git commit -m "Initial deployment"
git push
```

> \*\*Tip:\*\* Alternatively, upload model files directly through the Hugging Face Space \*\*Files\*\* tab in your browser — no LFS setup needed for files under 5 GB.

### Step 3 — Set secret environment variables

On Hugging Face, **do not** use a `.env` file. Instead:

1. Open your Space → **Settings** → **Variables and secrets**.
2. Click **New secret** and add each sensitive variable:

|Secret name|Value|
|-|-|
|`ROBOFLOW\_API\_KEY`|Your Roboflow API key|

3. Click **New variable** (non-secret) for the rest:

|Variable name|Value|
|-|-|
|`MODELS\_DIR`|`./saved\_models/`|
|`ROBOFLOW\_MODEL\_ID`|`concrete\_detection/6`|
|`SEG\_THRESHOLD`|`0.90`|
|`CONF\_THRESHOLD`|`0.90`|
|`SHOW\_MASK`|`true`|

The app reads all of these via `os.getenv()`, so it works on both local (`.env` file) and HF Spaces (Space secrets/variables) without any code change.

### Step 4 — Wait for the build

Hugging Face automatically installs `requirements.txt` and starts the app. Watch the **Logs** tab for progress. Once the build finishes your Space is live at:

```
https://huggingface.co/spaces/<your-username>/<your-space-name>
```

\---

## 🧩 Adding Roboflow

If you skip `ROBOFLOW\_API\_KEY`, the pipeline uses the full uploaded image as the ROI — the app still works correctly. To enable Roboflow:

1. Uncomment the Roboflow lines in `requirements.txt`.
2. Set `ROBOFLOW\_API\_KEY` in your `.env` / Space secrets.
3. Make sure `ROBOFLOW\_MODEL\_ID` matches your published model version.

\---

## 📄 License

MIT