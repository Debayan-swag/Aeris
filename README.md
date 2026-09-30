# Aeris

Aeris is a local satellite-intelligence application for searching a Sentinel-2 imagery archive, analyzing an uploaded image, and detecting change between two observations. It is built with Streamlit, RemoteCLIP, FAISS, OpenCV, and DistilGPT-2.

1. It has ```satellite imagery change detection```  with advance inage processing algorithms. It does pixel to pixel analysis for better change detection in every area of the image being referenced.

2. It has ```image to image semantic search``` fearure which is implemented with HyDE for better analysis and retrieval.

3. It has ```text to image semantic search``` fearure that retrieves the relevant images according to the user's query.


The usual user journey is:

1. Open the local Aeris landing page.
2. Select **Search satellite imagery**.
3. Choose Semantic Text Search, Image & Text Analysis, or Temporal Change Detection in the Streamlit hub.

## What you need

- Windows PowerShell (the included paths and launchers are currently Windows-oriented).
- Python 3.10 or 3.11.
- Internet access on the first model load, unless the model caches are already populated.
- Enough free disk space and RAM for the 3+ GB Sentinel-2 archive and model/index files.

## One-time setup

Open PowerShell in the project directory and create a virtual environment:

```powershell
cd {yourpath}\Aeris
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal only, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

`sentinel-2-processed.parquet` is the image archive. The FAISS file and Parquet mapping must describe the same archive, and the RemoteCLIP directory must contain a `.pt` checkpoint. These assets are project data, not Python packages; obtain them from the project owner or the location where your team stores the Aeris assets.

DistilGPT-2 is downloaded from Hugging Face when it is not already cached. The application stores that cache under `models/DistilGPT2` by default.

## Start Aeris

### Recommended: landing page plus application

```powershell
python server.py
```

This opens `http://localhost:8000/index.html`. Press **Search satellite imagery** to launch the Streamlit application. It will appear at `http://localhost:8501` after its first startup.

Keep the PowerShell window running while you use Aeris. Press `Ctrl+C` in that window to stop the landing server. The Streamlit process is started separately, so stop it from its own terminal if it remains running.

### Start Streamlit directly

Use this when you do not need the landing page:

```powershell
streamlit run GUI.py
```

Then open `http://localhost:8501`.

### Alternative launchers

```powershell
python fast_start.py
```

Starts Streamlit with the performance-oriented settings used by the launcher.

```powershell
python run_aeris.py
```

Runs a preflight check for the primary archive/index files before starting Streamlit. It can continue when optional configuration is missing.

## Use the application

### Semantic Text Search

1. Select **Semantic Text Search** from the hub.
2. Enter a target such as `large airport with runways` or choose a suggested query.
3. Select **Search Archive**.

Aeris uses DistilGPT-2 to analyze and expand the query, then searches the RemoteCLIP FAISS index. The page progressively shows the query-analysis stages, generated search variations, retrieval activity, and the best matching Sentinel-2 scenes.

### Image & Text Analysis

1. Select **Image & Text Analysis**.
2. Upload a satellite image in JPG, PNG, BMP, TIFF, or TIF format.
3. Add an analytical instruction and select **Analyze Multimodal Features**.

The uploaded image is encoded with RemoteCLIP and compared with the indexed archive. This mode needs the RemoteCLIP checkpoint and the FAISS/index files.

### Temporal Change Detection

1. Select **Temporal Change Detection**.
2. Upload a baseline image as **T-0** and a later image as **T-1**.
3. Select **Compute Temporal Discrepancies**.

For the most meaningful result, use captures of the same location with similar coverage, viewpoint, and resolution. Aeris aligns the images, normalizes their radiometry, computes multi-scale differences, identifies changed regions, and displays an overlay with metrics and region coordinates.

The numerical detection and visual outputs work locally. A longer natural-language intelligence report is optional and uses NVIDIA NIM when an API key is available; without it, Aeris shows the built-in summary instead.

Do not share API keys or commit a real `.env` file to a remote repository.

## Optional warmup

After the data assets are in place, you can check the expensive resources before opening the UI:

```powershell
python warmup_cache.py
```

This can make the first interactive use feel faster, but it loads models and archive metadata, so it may take time and use significant memory.
