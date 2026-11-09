# Land cover from Sentinel-1 radar: 12 monthly maps (v4 prototype)

Input: latitude, longitude and a start month. Output: 12 monthly land cover maps (6 classes) of a 7.7 km x 7.7 km area,
class shares per month (% of pixels with data), month-to-month change and first-to-last change in percentage points,
a majority map, and CSV/PNG exports. Sentinel-1 only; labels (ESA WorldCover 2021) are used for training, never at run time.

This README is in the project root (`LULC_project/`). The prototype itself is in `prototype_v4/`; all file names below without a folder (`app.py`, `weights/`, `visuals.py`, ...) refer to that folder.

## What is in this submission
| Item | Where |
| --- | --- |
| Report | `Final_Report_Shruti_Bhandari.pdf` and `.docx` |
| Final notebook: data, training, evaluation, training curves (saved outputs; a record, **do not re-run**) | `Deep_Learning_final_submission.ipynb` |
| Earlier attempt (version 3: 500 locations, single-pass images): notebook, split and statistics files, histories and figures | `Earlier_Attempt/` |
| Prototype: source code and requirements | `prototype_v4/` |
| Trained model weights and normalisation statistics | `prototype_v4/weights/` (`best_v4b_fpn_resnet50_imagenet_ctx.pt`, about 100 MB, and `stats_v4.json`) |
| Saved results the final notebook reads, dataset lists, figures | `evidence/` (`dataset_files/`, `results_final_model/`, `figures/`) |

The notebooks need the Google Drive dataset, a local cache and Earth Engine access, so they are records of finished runs. Cells that would delete or overwrite data are disabled with a guard.

## Setup
1. Open a terminal in `prototype_v4/` (`cd prototype_v4`) and create a Python environment with the packages in `requirements.txt` (the project used Python 3.12, torch with MPS on an Apple-silicon Mac): `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`. A Google account with Earth Engine access and a Google Cloud project with the Earth Engine API enabled is required; no other API keys are needed.
2. Authorise Earth Engine once: `earthengine authenticate`. The default project id is in the sidebar (change it to your own).
3. The weights are already in `weights/` in this submission (the app builds the network without downloading anything; the ImageNet initialisation is already inside the trained file). The commands below only show where they came from and are not needed:
   ```bash
   mkdir -p weights
   cp ~/lulc_local/v4_weights/best_v4b_fpn_resnet50_imagenet_ctx.pt weights/                       # the final model
   cp "$HOME/Google Drive/My Drive/LULC_project/native_v4_project/data/native_v4/stats_v4.json" weights/
   # only if you want the earlier pipelines selectable in the sidebar:
   cp ~/lulc_local/v4_weights/best_v4_fpn_ctx.pt ~/lulc_local/v4_weights/best_v4_fpn_ctx2.pt ~/lulc_local/v4_weights/postproc_val.json weights/
   ```
4. The bright theme is already set in `.streamlit/config.toml` (hidden folder, included). The optional training-place dots on the banner map (`scene_locations.csv`) are not included; the banner then shows the pin and the two held-out regions only.
5. `python -m streamlit run app.py` (run it inside `prototype_v4/`). The first run of a new place downloads up to 12 months from Earth Engine and takes several minutes; downloaded months are cached afterwards, so a repeated run takes seconds.

## Dataset
Dataset (Google Drive, view only): https://drive.google.com/drive/folders/1nsfC78u5OnmtzRJkRxCLyvQTpYt7_Fvp?usp=sharing
- The folder `native_v4` holds 372 places (the first 300 and the 72 extra training places) as `s1/<place>/<year>-<month>.npz`, `labels/<place>.npz` and `meta/<place>.json`, with `scenes_usable_v4.csv` and `stats_v4.json`. It is about 80 GB, so download single places, not the whole folder.
- `scenes_usable_v4b_370_places.csv` lists the **370 places used by the final model** with their split (train 257, validation 41, test 39, Australia 10, South Asia 23). `scenes_usable_v4.csv` lists only the first 298 usable places. The selection lists are `scenes_v4.csv`, `scenes_v4_extra.csv` and `pool_v4.csv`; copies of these small files are in `evidence/dataset_files/`.
- The training and evaluation code is in the cells of `Deep_Learning_final_submission.ipynb`.

Earlier attempt (version 3) data and model weights (`best_unet_e40.pt`, `best_fpn_e40.pt`, `best_fpn_ctx_e40.pt`), Google Drive, view only: https://drive.google.com/drive/folders/1AeAi391eNKHXWkX_Q2BGXYxtyTTb3FAC?usp=sharing


## Look and feel (files that only change the appearance)
- `visuals.py` and `visuals_data.py`: colours, the banner with a dotted world map (a pin follows the latitude and longitude you type), the step cards and the summary cards. They contain no analysis code; every number is passed in from `app.py` and `lulc_core_v4.py`. The map outline is Natural Earth (public domain).
- `.streamlit/config.toml` (copy of `streamlit_config.toml`): the bright theme. Without it the page still works, but in a dark browser the tables and inputs stay dark. Create it with `mkdir -p .streamlit && cp streamlit_config.toml .streamlit/config.toml` inside this folder.
- `scene_locations.csv` (optional, made by `export_scene_locations.py`): one dot per place the model was trained, validated or tested on, drawn on the banner map together with the two held-out regions. Without the file the banner shows only the pin and the held-out regions.

## What the app does (same recipe as the training data)
- 10 km box on an exact UTM grid at 10 m; the 9 central 256 x 256 tiles (7.68 km square).
- Sentinel-1 GRD, IW, VV+VH, plain dB from Earth Engine (no offset), images whose footprint covers the area.
- Per month at most 5 evenly spaced passes; an image with under 90% valid pixels is replaced by its nearest pass in time (at most 9 tries);
  the monthly composite is the per-pixel median. A month without an image stays empty and is left out of all shares and changes.
- Inputs are z-scored with the training statistics (`stats_v4.json`). The network gets 6 channels: the monthly composite (VV, VH) plus
  the mean and standard deviation (divided by 3) of VV and VH over the 12 months.
- Model: FPN with a ResNet-50 encoder. **Default (final model): one FPN whose encoder was initialised with ImageNet weights and then trained on 257 scenes**
  (chosen on validation by rules fixed in advance; no ensemble, no offsets). Selectable for comparison: the earlier pipeline (two models averaged plus fixed
  per-class offsets, `postproc_val.json`) and the earlier single model `v4_fpn_ctx`.
- Shares: pixels per class / pixels with data in that month x 100. Month-to-month change: share(month n+1) - share(month n) in percentage points
  (empty if either month has no data). Total change: share(last month with data) - share(first month with data). "Pixels that changed class"
  counts every pixel whose class differs between those two months, which includes classification errors and seasonal effects (upper bound).

## Additional analyses in the app (all computed from the same 12 monthly maps)
- **Persistent change.** The most frequent class of the first k months with data (k = 1 to 4, default 3) is compared with that of the last k months. A pixel counts as
  changed only if both majorities are stable (at least 2 of the k months agree) and differ, so one noisy month cannot create change. The app shows the three maps, the
  persistent change next to the raw first-to-last change, the share of pixels without a stable majority, and a from-to table with areas (1 pixel = 0.01 ha).
  The vote removes one-month flicker; errors that last several months stay, so the result is still an upper bound for real change.
  The from-to table also gives the pixels that move in the opposite direction between the same two classes and the net difference; "two-way swaps" (matched pixels)
  mostly cancel and are typical of jitter along class boundaries, so the net change is the more careful estimate of one-way conversion.
  The measurement of how much the vote lowers the change estimate on the validation scenes is below (upper bounds: validation scenes contain real change too).
- **Radar signature.** Mean VV and VH backscatter (dB) of the pixels the model assigns to each class, by month (class-months need at least 500 pixels). It describes
  what the model calls each class; it is not independent ground truth.
- **CORINE comparison (optional, Europe only).** Tick the box in the sidebar before running. The 12-month majority map is compared with CORINE Land Cover 2018 on the same
  10 m grid, at 100 m (the resolution of CORINE, the headline number) and at 10 m. The 44 CORINE classes are mapped to the six classes by an approximate rule written in
  `lulc_core_v4.py` (`CLC_TO6`). Indicative only: CORINE has a 25 ha minimum mapping unit and dates from 2018.
  Check the Earth Engine asset once before using it: `python -c "import ee; ee.Initialize(project='<your project>'); print(ee.Image('COPERNICUS/CORINE/V20/100m/2018').bandNames().getInfo())"` must print `['landcover']`.

## Prediction method and tile seams
The models were trained and evaluated on separate 256 x 256 tiles. Predicting each tile on its own ("tile by tile") leaves visible straight seams along the borders of the
3 x 3 grid, because near a tile border the network sees padding instead of the neighbouring tile. Measured on the 41 validation scenes (983 scene-months, final model), the
predicted class changes across an interior tile border for 18.7% of adjacent pixel pairs against 1.3% elsewhere (seam excess 17.4 percentage points, 95% interval 13.4 to 21.2).
Two alternatives remove it (excess 0.14 and 0.16 pp) without lowering accuracy against the WorldCover labels:

| method | OA | mIoU | mIoU difference to tile by tile (paired scene bootstrap) |
| --- | --- | --- | --- |
| tile by tile | 0.772 | 0.634 | - |
| whole area in one pass (768 x 768) | 0.775 | 0.639 | +0.0045 [-0.0073, +0.0184] |
| overlapping windows (256 x 256 every 128 px, triangular blending) | 0.779 | 0.646 | +0.0119 [+0.0057, +0.0194] |

The default is **whole area**, chosen by a rule fixed before the test (adopt a method only if its mIoU is at most 0.005 below tile by tile and its seam excess at least 50% lower;
if both qualify, the lower seam excess wins unless they are within 10%: then the cheaper one). Whole area won by a hair (0.140 against 0.157 pp, a 10.8% difference). Caveats:
whole area changes about 7% of the pixels even far from any tile border, probably because the network never saw 768 x 768 inputs, while overlapping windows change mainly
the pixels near the borders (1.4% far away); overlapping windows are the alternative to select in the sidebar (or set `DEFAULT_METHOD` in `lulc_core_v4.py`).
All test-set and held-out-region metrics in the project were computed tile by tile; the validation scenes were also used to choose the model, so absolute validation values are
somewhat optimistic, but the comparison between the methods uses the same model on the same scenes.

## How much of the change is noise (measured)
The false-change measurement (code in the final notebook, Part 7) runs the final model on the 41 validation scenes for 2020 and 2021 (82 scene-years) with each prediction method and measures the numbers the app reports.
The scenes contain real change too, so these are **upper bounds for false change**. Medians over the 82 scene-years (whole area, the default; tile by tile / overlapping windows in brackets):

| measure | median | 90th percentile | maximum |
| --- | --- | --- | --- |
| largest monthly move of any class (pp) | 2.55 (2.93 / 2.63) | 10.86 (13.92 / 12.59) | 65.07 (52.43 / 55.17) |
| raw first-to-last change (% of pixels) | 2.86 (3.99 / 3.95) | 8.53 (10.65 / 9.56) | 36.56 (27.89 / 28.16) |
| persistent change, k = 3 (% of stable pixels) | 2.58 (3.14 / 3.02) | 7.26 (10.53 / 9.88) | 15.91 (18.12 / 18.02) |

Paired against tile by tile, the methods differ little (persistent change: whole -0.071 pp [-0.355, +0.002], overlap -0.117 pp [-0.196, +0.000]; intervals over scenes), so the prediction
method does not add false change. A result of about 3% of pixels is within this noise; the worst scene-year moved by 65 pp in a month.

## Known differences from training and limitations
- **Context window.** In training the annual statistics came from a calendar year. The app computes them over the 12 months of the chosen
  window. For a window starting in January this is identical to training; for other start months it is an approximation that has not been
  tested on held-out scenes.
- **Years.** Training used 2020 and 2021 imagery. Other years are an untested transfer; since 2022 fewer passes per month are available in many places.
- **Labels.** WorldCover 2021 has a reported global overall accuracy of 76.7%; Shrub/Grass is the weakest class (WorldCover and Dynamic World disagree strongly on it).
- **Regions unlike the training data** (held-out tests): Australia (overall accuracy below a constant guess) and South Asia (Bare/Arid almost lost).
- **False change.** Month-to-month differences include seasonal backscatter effects, different pass counts and classification errors. For the final model it was measured on 82 scene-years (section above): a median of 2.9% of the pixels changes class between the first and the last month, so changes near 3% are within the noise. A different number of passes per month can also mimic change (Singapore: months with only 2 or 3 images showed about 2.7 points more Urban); the app shows a notice for months with fewer than four images. An earlier pilot with a version-3 model on Heidelberg showed swings up to 28 percentage points.
- No land/sea check: over open water the maps are Water/Wetland or the app reports that no image was usable.
- Run time is dominated by the Earth Engine downloads (up to 12 months x up to 5 passes, 1000 x 1000 pixels each). Downloaded months are cached in `cache/`.

## Tests
Offline (synthetic data, fake Earth Engine): the copied functions equal the notebook's; composites and context equal the training procedure
for a January window; the network input equals the training `get_sample`; predictions equal the final-test script; stitching, statistics,
window checks and the full app flow including error paths. The persistent-change vote equals an independent pixel-by-pixel implementation on random maps, and cuts
simulated independent flicker as theory predicts (25.5% raw to 11.3% persistent at 15% flicker); radar signatures equal a direct computation; the CORINE class mapping covers
all 44 classes, the 768 x 768 crop is aligned with the map (a one-pixel shift is detected), and a failed or outside-Europe CORINE request leaves the rest of the analysis intact.
**Run on real data:** Heidelberg 2020 (with the CORINE comparison) and 2025, Auckland 2025, and Singapore April 2025 to March 2026 (a non-January start). **Not tested:** a place with a known real change, other Earth Engine accounts or a fresh installation, other start months, run time. The new look (banner, cards, theme) was previewed in a headless Chromium with synthetic results, not in every browser.
