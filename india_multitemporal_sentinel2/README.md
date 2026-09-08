# India Multi-Temporal Sentinel-2 Dataset (Earth Engine)

## 1. Dataset purpose

This side-car dataset supports TerraWatch AI capabilities beyond semantic search:

- Multi-temporal satellite analysis and historical timelines
- Before/after comparison and change detection
- Construction / road / vegetation / water / urban monitoring preparation
- NDVI, NDWI, NDBI computation
- Future VLM-based change explanation and automated monitoring

It does **not** replace the existing OpenCLIP + FAISS semantic search corpus.

## 2. Data source

| Item | Value |
|------|--------|
| Sensor | Sentinel-2 |
| Collection | `COPERNICUS/S2_SR_HARMONIZED` |
| Bands | B2, B3, B4, B8, B11, B12 |
| Resolution | 10 m |
| Format | Multispectral GeoTIFF (not RGB-only JPEG) |
| AOI | Fixed ~256×256 @ 10 m (2.56 km × 2.56 km) per location |
| Coverage | 20 Indian cities × 5 fixed locations = 100 AOIs |

All acquisition dates, cloud percentages, product IDs, granule IDs, and MGRS tiles come from Earth Engine metadata. Nothing is fabricated.

## 3. Authentication (Google Earth Engine)

1. Create / select a Google Cloud project with Earth Engine enabled.
2. Install dependencies (see below).
3. Authenticate once:

```bash
python -c "import ee; ee.Authenticate()"
```

4. Set your project (recommended):

```powershell
# PowerShell
$env:EE_PROJECT = "your-gcp-project-id"
```

```bash
# bash
export EE_PROJECT=your-gcp-project-id
```

Or pass `--project your-gcp-project-id` to the downloader.

Credentials are never hardcoded in this repository.

## 4. Installation

From the TerraWatch repository root:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r india_multitemporal_sentinel2/requirements.txt
```

**Linux / macOS:**

```bash
source .venv/bin/activate
pip install -r india_multitemporal_sentinel2/requirements.txt
```

## 5. Dependencies

See `requirements.txt` in this folder:

- `earthengine-api`, `rasterio`, `pyproj`, `requests`, `numpy`, `pandas`, `matplotlib`, `tqdm`

These are isolated from the main TerraWatch search stack (OpenCLIP / FAISS).

## 6. Download

From the **TerraWatch repo root** (so imports resolve):

```bash
# Full dataset (~8 scenes × 100 locations — actual count depends on EE availability)
python india_multitemporal_sentinel2/scripts/download_dataset.py --project YOUR_PROJECT

# Smoke test: one location, 3 real scenes
python india_multitemporal_sentinel2/scripts/download_dataset.py --project YOUR_PROJECT --locations DEL_001 --max-images 3
```

Environment overrides:

| Variable | Meaning |
|----------|---------|
| `EE_PROJECT` | Earth Engine / GCP project |
| `TW_S2_START_DATE` | Default `2018-01-01` |
| `TW_S2_END_DATE` | Default today |
| `TW_S2_PRIMARY_CLOUD` | Default `10` |
| `TW_S2_FALLBACK_CLOUD` | Default `20` |
| `TW_S2_TARGET_IMAGES` | Default `8` |
| `TW_S2_LOCATION_FILTER` | Comma-separated location IDs |
| `TW_S2_MAX_IMAGES` | Cap images per location |
| `TW_S2_PIXEL_CLOUD_MASK` | `1` to enable S2 cloud probability mask |

## 7. Validate

```bash
python india_multitemporal_sentinel2/scripts/validate_dataset.py
```

Report: `metadata/validation_report.txt`

## 8. Visualization

```bash
python india_multitemporal_sentinel2/scripts/visualize_dataset.py --location-id DEL_001 --mode truecolor
python india_multitemporal_sentinel2/scripts/visualize_dataset.py --location-id DEL_001 --mode NDVI
python india_multitemporal_sentinel2/scripts/visualize_dataset.py --location-id DEL_001 --mode before_after
```

## 9. Spectral indices

```text
NDVI = (B8 - B4) / (B8 + B4)
NDWI = (B3 - B8) / (B3 + B8)
NDBI = (B11 - B8) / (B11 + B8)
```

```bash
python india_multitemporal_sentinel2/scripts/spectral_indices.py --image-id DEL_001_YYYYMMDD --write
```

Source GeoTIFFs are never modified.

## 10. Resume behavior

The downloader is **resumable**:

- If `DEL_001_2021-05-12.tif` already exists and verifies as a 6-band GeoTIFF, it is **not** re-downloaded (`already_exists` in the acquisition log).
- Failed individual scenes are logged; processing continues for remaining locations.
- Re-run the same command after a crash to continue.

Also:

```bash
python india_multitemporal_sentinel2/scripts/create_metadata.py
python india_multitemporal_sentinel2/scripts/change_candidates.py
python india_multitemporal_sentinel2/scripts/sync_temporal_db.py   # optional SQLite table
```

## Directory layout

```text
india_multitemporal_sentinel2/
├── images/<City>/<LOCATION_ID>/<LOCATION_ID>_YYYY-MM-DD.tif
├── metadata/
│   ├── locations.csv
│   ├── temporal_metadata.csv
│   ├── dataset_statistics.csv
│   ├── acquisition_log.csv
│   ├── change_candidates.csv
│   └── ...
├── scripts/
├── config.py
├── requirements.txt
└── README.md
```

## Integration with TerraWatch

| Component | Action |
|-----------|--------|
| OpenCLIP / FAISS / `indexes/` | **Untouched** |
| Existing API / Streamlit | **Untouched** |
| `scripts/download_temporal_dataset.py` (Planetary Computer RGB) | **Untouched** |
| Optional `temporal_images` SQLite table | Additive only via `sync_temporal_db.py` |

Do **not** run `build_embeddings.py` / `build_index.py` on these GeoTIFFs unless you intentionally create a separate RGB preview index.
