"""
lulc_core_v4.py: input stage, inference and statistics of the v4 prototype.

The input stage reproduces the v4 training data exactly: a 10 km box on a UTM grid at 10 m, the 9 central 256-px tiles,
at most 5 evenly spaced Sentinel-1 passes per month (image needs >= 90% valid pixels, nearest pass replaces a bad one),
monthly median, no incidence-angle offset, z-score with the training statistics, and the annual context channels
(mean/std of VV and VH, std divided by 3). Functions in the PURE blocks are copied from the project notebook
(download cell and cache-build cell) and are tested against it.
"""
import os, io, json, time
from datetime import date
import numpy as np

BOX_PX, TILE = 1000, 256
PER_MONTH, VALID_MIN, MAX_TRIES = 5, 0.9, 9
STD_SCALE, NC = 3.0, 6
TILES = list(range(9))
CLASS_NAMES = ['Forest', 'Shrub/Grass', 'Cropland', 'Urban', 'Bare/Arid', 'Water/Wetland']
CLASS_COLORS = ['#1b7837', '#a6d96a', '#f4d35e', '#d73027', '#bf9b6f', '#2b83ba']
NODATA_COLOR = '#d9d9d9'
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
FINAL_TAG = 'v4b_fpn_resnet50_imagenet_ctx'            # the final model: one FPN-ResNet50, ImageNet-pretrained encoder, trained on 257 scenes
ENSEMBLE_TAGS = ['v4_fpn_ctx', 'v4_fpn_ctx2']          # the earlier pipeline (two models averaged + fixed offsets), kept for comparison
SINGLE_TAG = 'v4_fpn_ctx'                              # the earlier single model
DEFAULT_METHOD = 'whole'          # chosen by the rule fixed before the test in v4b_seam_test.py (validation: seam excess 17.4 pp -> 0.14 pp, mIoU +0.0045). 'overlap' is the alternative:
                                  # mIoU +0.0119 [+0.0057, +0.0194], and it leaves pixels far from tile borders much closer to the tile-by-tile maps; edit this line to switch the default
PREDICT_METHODS = {'tiles': 'Tile by tile (as trained and evaluated)', 'whole': 'Whole area in one pass', 'overlap': 'Overlapping windows, blended (smooth)'}
MODES = {'final': 'Final model: ImageNet-pretrained FPN (single model)', 'ensemble': 'Earlier pipeline: two models + class offsets', 'single': 'Earlier single model (v4_fpn_ctx)'}

# ==== PURE FUNCTIONS (copied from the project notebook, download and cache cells) ====
def tile_origins():
    """3 x 3 grid of 256-px tiles centred in the 1000-px box (margin of 116 px on every side)."""
    m = (BOX_PX - 3 * TILE) // 2
    return [(m + r * TILE, m + c * TILE) for r in range(3) for c in range(3)]

def evenly_spaced(n, k):
    """The min(n, k) indices spread evenly over n time-ordered candidates."""
    if n == 0:
        return []
    return sorted(set(np.unique(np.linspace(0, n - 1, min(n, k)).round().astype(int)).tolist()))

def pick_month(cands, k, fetch, max_tries):
    """Take up to k usable images, evenly spread in time. If an image is unusable, its nearest unused neighbour in time replaces it."""
    n, used, got, tried = len(cands), set(), [], 0
    for tgt in evenly_spaced(n, k):
        for i in [tgt] + sorted((j for j in range(n) if j != tgt), key=lambda j: (abs(j - tgt), j)):
            if i in used:
                continue
            if tried >= max_tries:
                break
            used.add(i); tried += 1
            r = fetch(cands[i])
            if r is not None:
                got.append((cands[i], r)); break
        if tried >= max_tries:
            break
    got.sort(key=lambda g: g[0]['t'])
    return got, tried

def crop_tiles(arr, tiles):
    """arr: (2, 1000, 1000) with -9999 where there is no data -> (n_tiles, 2, 256, 256) float16 with NaN, valid share."""
    o = tile_origins()
    out = np.stack([arr[:, o[t][0]:o[t][0] + TILE, o[t][1]:o[t][1] + TILE] for t in tiles]).astype(np.float32)
    out[(out <= -9998) | (out == 0)] = np.nan          # no-data and zero-filled areas -> NaN
    valid = float(1 - np.isnan(out).any(axis=1).mean())
    return out.astype(np.float16), valid

def nanmedian_small(x):
    """Median over axis 0 ignoring NaN, for a small first axis (here at most 5 passes). NaN where every value is NaN."""
    x = x.astype(np.float32)
    s = np.sort(x, axis=0)                                  # NaN sorts last
    n = (~np.isnan(x)).sum(axis=0)
    lo = np.clip((n - 1) // 2, 0, x.shape[0] - 1); hi = np.clip(n // 2, 0, x.shape[0] - 1)
    out = (np.take_along_axis(s, lo[None], 0)[0] + np.take_along_axis(s, hi[None], 0)[0]) / 2
    out[n == 0] = np.nan
    return out

def month_composite(x, tiles_shape):
    """x: (n_images, tiles, 2, H, W) or empty -> (tiles, 2, H, W) float32 monthly median, all NaN if there is no image."""
    if len(x) == 0:
        return np.full(tiles_shape, np.nan, np.float32)
    return nanmedian_small(x)

def year_context(comp12):
    """comp12: (12, tiles, 2, H, W) monthly composites -> (tiles, 4, H, W): mean VV, mean VH, std VV, std VH."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mean, std = np.nanmean(comp12, axis=0), np.nanstd(comp12, axis=0)
    return np.concatenate([mean, std], axis=1)
# ==== END PURE ====

# ------------------------------------------------------------------ window logic
def window_months(year, month):
    """The 12 (year, month) pairs from the start month."""
    return [(year + (month - 1 + i) // 12, (month - 1 + i) % 12 + 1) for i in range(12)]

def check_window(year, month, today=None):
    """-> (ok, error message or '', list of warnings)."""
    today = today or date.today(); warns = []
    ym = window_months(year, month)
    if not (1 <= month <= 12): return False, 'The start month must be 1 to 12.', warns
    if year < 2015: return False, 'Sentinel-1 data in Earth Engine start in late 2014; choose 2015 or later.', warns
    if date(year, month, 1) > today: return False, 'The start month is in the future.', warns
    if date(*ym[-1], 1) > today:
        return False, 'The 12-month window ends in the future. Choose an earlier start month so that all 12 months have passed.', warns
    if (ym[-1][0], ym[-1][1]) == (today.year, today.month):
        warns.append('The last month of the window is the current month and may be incomplete.')
    if not all(y in (2020, 2021) for y, _ in ym):
        warns.append('The models were trained on 2020 and 2021 imagery. Other years are a transfer that was not tested; '
                     'since 2022 fewer Sentinel-1 passes per month are available in many places (hypothesis for the effect on accuracy).')
    return True, '', warns

# ------------------------------------------------------------------ Earth Engine (same calls as the download cell)
def init_ee(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(); ee.Initialize(project=project)

def make_region(lat, lon):
    import ee
    return ee.Geometry.Point([lon, lat]).buffer(BOX_PX * 5 + 300).bounds()

def grid(lat, lon):
    import ee
    zone = min(int((lon + 180) // 6) + 1, 60)
    epsg = (32600 if lat >= 0 else 32700) + zone
    cx, cy = ee.Geometry.Point([lon, lat]).transform(f'EPSG:{epsg}', 1).coordinates().getInfo()
    return epsg, [10, 0, round(cx / 10) * 10 - BOX_PX * 5, 0, -10, round(cy / 10) * 10 + BOX_PX * 5]

def s1_image(image_id):
    import ee
    return ee.Image(f'COPERNICUS/S1_GRD/{image_id}').select(['VV', 'VH']).unmask(-9999)        # no offset: plain GRD dB

def list_month(region, y, m):
    import ee
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    core = region.centroid(1).buffer(4200).bounds()
    col = (ee.ImageCollection('COPERNICUS/S1_GRD').filterDate(f'{y}-{m:02d}-01', f'{ny}-{nm:02d}-01').filterBounds(region)
           .filter(ee.Filter.contains('.geo', core))
           .filter(ee.Filter.eq('instrumentMode', 'IW'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).sort('system:time_start'))
    info = ee.Dictionary({'id': col.aggregate_array('system:index'), 't': col.aggregate_array('system:time_start'),
                          'o': col.aggregate_array('orbitProperties_pass')}).getInfo()
    return [{'id': i, 't': t, 'o': 'A' if o == 'ASCENDING' else 'D'} for i, t, o in zip(info['id'], info['t'], info['o'])]

def fetch_array(img, bands, epsg, tf, tries=4):
    import requests
    for a in range(tries):
        try:
            url = img.getDownloadURL({'bands': bands, 'crs': f'EPSG:{epsg}', 'crs_transform': tf,
                                      'dimensions': f'{BOX_PX}x{BOX_PX}', 'format': 'NPY'})
            r = requests.get(url, timeout=600); r.raise_for_status()
            arr = np.load(io.BytesIO(r.content))
            return np.stack([arr[b] for b in bands])
        except Exception:
            time.sleep(3 * (a + 1) ** 2)
    return None

class EEBackend:
    """The two Earth Engine operations the prototype needs. Replaced by a fake in the offline tests."""
    def __init__(self, lat, lon, project):
        init_ee(project)
        self.epsg, self.tf = grid(lat, lon); self.region = make_region(lat, lon)
    def list_month(self, y, m):
        return list_month(self.region, y, m)
    def fetch(self, image_id):
        return fetch_array(s1_image(image_id), ['VV', 'VH'], self.epsg, self.tf)
    def corine(self, year=2018):
        return corine_array(self.epsg, self.tf, year)

def get_month(backend, y, m, cache_dir=None):
    """One month, as the training data: (n_images, 9, 2, 256, 256) float16, dates, n_candidates. Cached on disk."""
    path = f'{cache_dir}/{y}-{m:02d}.npz' if cache_dir else None
    if path and os.path.exists(path):
        with np.load(path) as z:
            return z['x'], [str(d) for d in z['dates']], int(z['n_candidates'])
    cands = backend.list_month(y, m)
    def fetch(c):
        arr = backend.fetch(c['id'])
        if arr is None:
            return None
        x, valid = crop_tiles(arr, TILES)
        return (x, valid) if valid >= VALID_MIN else None
    got, tried = pick_month(cands, PER_MONTH, fetch, MAX_TRIES)
    x = np.stack([g[1][0] for g in got]) if got else np.zeros((0, 9, 2, TILE, TILE), np.float16)
    dates = [time.strftime('%Y-%m-%d', time.gmtime(g[0]['t'] / 1000)) for g in got]
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + '.tmp.npz'; np.savez(tmp, x=x, dates=np.array(dates, dtype='U10'), n_candidates=len(cands)); os.replace(tmp, path)
    return x, dates, len(cands)

def download_window(backend, year, month, cache_dir=None, workers=3, progress=None):
    """12 months downloaded in parallel -> list of (x, dates, n_candidates) in time order.
    Progress is reported from the calling thread (Streamlit elements must not be touched from worker threads)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ym = window_months(year, month); out = [None] * 12
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(get_month, backend, y, m, cache_dir): i for i, (y, m) in enumerate(ym)}
        for n, f in enumerate(as_completed(futs), 1):
            i = futs[f]; out[i] = f.result()                       # re-raises a failure of that month here
            if progress: progress(n / 12, f'{MONTH_NAMES[ym[i][1] - 1]} {ym[i][0]}: {len(out[i][1])} image(s)')
    return out

def assemble(months):
    """list of 12 (x, dates, n_cand) -> composites (12, 9, 2, 256, 256) float32 (NaN = no image) and context (9, 4, 256, 256)."""
    shape = next((x.shape[1:] for x, _, _ in months if len(x)), None)
    if shape is None:
        raise RuntimeError('No usable Sentinel-1 image in any of the 12 months (open sea, or outside coverage?).')
    comp = np.stack([month_composite(x, shape) for x, _, _ in months])
    return comp, year_context(comp)        # the context is computed over the same 12 months (see README)

# ------------------------------------------------------------------ models and inference
def load_models(weights_dir, mode='final', device='cpu'):
    """-> (models, offsets, mean, std). mode 'final' = the ImageNet-pretrained FPN alone, no offsets (the model chosen on validation);
    'ensemble' = the earlier two-model pipeline with its fixed offsets from postproc_val.json; 'single' = v4_fpn_ctx alone."""
    import torch, segmentation_models_pytorch as smp
    stats = json.load(open(f'{weights_dir}/stats_v4.json'))
    mean = np.array(stats['mean'], dtype=np.float32); std = np.array(stats['std'], dtype=np.float32)
    tags = {'final': [FINAL_TAG], 'ensemble': ENSEMBLE_TAGS, 'single': [SINGLE_TAG]}[mode]
    models = []
    for t in tags:
        m = smp.FPN('resnet50', encoder_weights=None, in_channels=6, classes=NC)          # all three models are FPN with a ResNet-50 encoder
        m.load_state_dict(torch.load(f'{weights_dir}/best_{t}.pt', map_location='cpu'))
        models.append(m.to(device).eval())
    offsets = np.zeros(NC, np.float32)
    if mode == 'ensemble':
        offsets = np.array(json.load(open(f'{weights_dir}/postproc_val.json'))['biases_all_scenes']['ens'], dtype=np.float32)
    return models, offsets, mean, std

def make_input(comp_m, ctx, mean, std):
    """One month -> network input (9, 6, H, W) and the no-data mask (9, H, W). Same formulas as the training code; float16 rounding as in the cache."""
    x = comp_m.astype(np.float16).astype(np.float32)
    bad = np.isnan(x).any(axis=1)
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    x = np.where(bad[:, None], 0.0, x)
    c = ctx.astype(np.float16).astype(np.float32)
    c = np.nan_to_num(np.concatenate([(c[:, :2] - mean[None, :, None, None]) / std[None, :, None, None], c[:, 2:] / STD_SCALE], axis=1), nan=0.0)
    return np.concatenate([x, c], axis=1).astype(np.float32), bad

def _probs(models, xt):
    """Average softmax probabilities of the models for an input batch (B, 6, H, W)."""
    return sum(m(xt).softmax(1) for m in models) / len(models)

def window_weights(size=TILE):
    """2-D triangular weights for blending overlapping windows: small at the window edge, 1 in the middle, never zero."""
    w1 = np.minimum(np.arange(size) + 1, size - np.arange(size)).astype(np.float32)
    w1 /= w1.max()
    return np.outer(w1, w1)

def predict_month(comp_m, ctx, models, offsets, mean, std, device='cpu', method='tiles', stride=TILE // 2, batch=9):
    """One month -> (768, 768) uint8 class map, 255 where there is no data.
    method 'tiles'   : every 256 x 256 tile on its own (how the models were trained and evaluated)
           'whole'   : the stitched 768 x 768 area in one pass, so no tile border is an image border
           'overlap' : 256 x 256 windows every `stride` pixels over the stitched area, probabilities blended with triangular weights"""
    import torch
    x9, bad9 = make_input(comp_m, ctx, mean, std)
    off = torch.tensor(offsets, dtype=torch.float32, device=device).view(1, NC, 1, 1)
    with torch.no_grad():
        if method == 'tiles':
            p = _probs(models, torch.from_numpy(x9).to(device))
            out = stitch((p.clamp_min(1e-6).log() + off).argmax(1).cpu().numpy().astype(np.uint8))
        else:
            xm = stitch(np.moveaxis(x9, 1, 0))                                          # (6, 768, 768)
            if method == 'whole':
                p = _probs(models, torch.from_numpy(xm[None]).to(device))
                out = (p.clamp_min(1e-6).log() + off).argmax(1)[0].cpu().numpy().astype(np.uint8)
            elif method == 'overlap':
                S = 3 * TILE; pos = [(r, c) for r in range(0, S - TILE + 1, stride) for c in range(0, S - TILE + 1, stride)]
                w = torch.from_numpy(window_weights()).to(device)
                acc = torch.zeros(NC, S, S, device=device); wsum = torch.zeros(S, S, device=device)
                for i in range(0, len(pos), batch):
                    chunk = pos[i:i + batch]
                    pb = _probs(models, torch.from_numpy(np.stack([xm[:, r:r + TILE, c:c + TILE] for r, c in chunk])).to(device))
                    for (r, c), pi in zip(chunk, pb):
                        acc[:, r:r + TILE, c:c + TILE] += pi * w; wsum[r:r + TILE, c:c + TILE] += w
                p = (acc / wsum)[None]
                out = (p.clamp_min(1e-6).log() + off).argmax(1)[0].cpu().numpy().astype(np.uint8)
            else:
                raise ValueError(f'unknown prediction method {method!r}')
    out[stitch(bad9)] = 255
    return out

def predict_maps(comp, ctx, models, offsets, mean, std, device='cpu', method='tiles'):
    """-> (12, 9, 256, 256) uint8 class map per month and tile; 255 where there is no data. method: see predict_month."""
    import torch
    out = np.full(comp.shape[:2] + comp.shape[3:], 255, np.uint8)
    off = torch.tensor(offsets, dtype=torch.float32, device=device).view(1, NC, 1, 1)
    with torch.no_grad():
        for k in range(comp.shape[0]):
            if np.isnan(comp[k]).all():
                continue
            if method != 'tiles':
                out[k] = unstitch(predict_month(comp[k], ctx, models, offsets, mean, std, device, method)); continue
            x, bad = make_input(comp[k], ctx, mean, std)                                # the original tile-by-tile path, unchanged
            xt = torch.from_numpy(x).to(device)
            p = sum(m(xt).softmax(1) for m in models) / len(models)
            pred = (p.clamp_min(1e-6).log() + off).argmax(1).cpu().numpy().astype(np.uint8)
            pred[bad] = 255; out[k] = pred
    return out

def stitch(maps9):
    """(..., 9, 256, 256) row-major 3 x 3 tiles -> (..., 768, 768)."""
    s = maps9.shape[:-3]
    a = maps9.reshape(*s, 3, 3, TILE, TILE)
    a = np.moveaxis(a, -3, -2)                  # (..., 3, 256, 3, 256)
    return a.reshape(*s, 3 * TILE, 3 * TILE)

def unstitch(m):
    """(..., 768, 768) -> (..., 9, 256, 256), row-major tiles: the inverse of stitch."""
    sh = m.shape[:-2]
    a = np.moveaxis(m.reshape(*sh, 3, TILE, 3, TILE), -2, -3)
    return a.reshape(*sh, 9, TILE, TILE)

# ------------------------------------------------------------------ statistics
def class_shares(maps):
    """maps (12, H, W) uint8 -> (12, 6) share of valid pixels in %, NaN row where a month has no valid pixel."""
    out = np.full((maps.shape[0], NC), np.nan)
    for k in range(maps.shape[0]):
        v = maps[k][maps[k] != 255]
        if v.size: out[k] = np.bincount(v, minlength=NC)[:NC] / v.size * 100
    return out

def month_to_month(shares):
    """(12, 6) -> (11, 6) change in percentage points between consecutive months; NaN if either month has no data."""
    return shares[1:] - shares[:-1]

def first_to_last(shares):
    """Change in percentage points from the first to the last month with data. -> (delta (6,), index first, index last) or None."""
    ok = np.flatnonzero(~np.isnan(shares).any(axis=1))
    if len(ok) < 2: return None
    return shares[ok[-1]] - shares[ok[0]], int(ok[0]), int(ok[-1])

def pixel_change_pct(maps, i, j):
    """Share (%) of pixels valid in both months i and j whose class differs. Includes classification noise: an upper bound for real change."""
    a, b = maps[i], maps[j]; v = (a != 255) & (b != 255)
    return float((a[v] != b[v]).mean() * 100) if v.any() else float('nan')

def majority_map(maps):
    """Most frequent class over the months with data, per pixel; 255 where no month has data."""
    counts = np.stack([(maps == c).sum(0) for c in range(NC)])
    out = counts.argmax(0).astype(np.uint8); out[counts.sum(0) == 0] = 255
    return out

# ------------------------------------------------------------------ persistent change (majority vote at both ends of the window)
def persistent_change(maps, k=3):
    """maps (12, H, W) uint8, 255 = no data. Compares the MAJORITY class of the first k months with data with that of the last k months with data.
    A pixel counts as changed only if it has a stable majority at both ends (at least 2 of the k months and more than half of its valid months)
    and the two majorities differ. One-month flicker is voted out; pixels without a stable majority are reported as 'unstable', not as change.
    Returns None if fewer than 2 months have data."""
    has = (maps != 255).any(axis=(1, 2)); idx = np.flatnonzero(has)
    if len(idx) < 2:
        return None
    k_eff = int(min(k, len(idx) // 2)); e_idx, l_idx = idx[:k_eff], idx[-k_eff:]
    need = 2 if k_eff >= 2 else 1
    def vote(sel):
        counts = np.stack([(maps[sel] == c).sum(0) for c in range(NC)])
        n, best, bc = counts.sum(0), counts.argmax(0), counts.max(0)
        return np.where((bc >= need) & (bc * 2 > n), best, 255).astype(np.uint8)
    early, late = vote(e_idx), vote(l_idx)
    stable = (early != 255) & (late != 255); changed = stable & (early != late)
    trans = np.bincount(early[stable].astype(np.int64) * NC + late[stable], minlength=NC * NC).reshape(NC, NC)
    valid_px = int((maps != 255).any(axis=0).sum()); sp, cp = int(stable.sum()), int(changed.sum())
    off = trans.copy(); np.fill_diagonal(off, 0); balanced = int(np.minimum(off, off.T).sum()); net = cp - balanced      # balanced = pixels that are part of a two-way swap
    return {'net_px': net, 'balanced_px': balanced, 'balanced_pct': (balanced / cp * 100) if cp else float('nan'), 'net_pct_of_stable': (net / sp * 100) if sp else float('nan'),
            'early': early, 'late': late, 'stable': stable, 'changed': changed, 'transitions': trans, 'k': k_eff,
            'early_months': e_idx.tolist(), 'late_months': l_idx.tolist(), 'stable_px': sp, 'changed_px': cp,
            'changed_pct': (cp / sp * 100) if sp else float('nan'), 'unstable_pct': ((valid_px - sp) / valid_px * 100) if valid_px else float('nan'),
            'raw_changed_pct': pixel_change_pct(maps, int(idx[0]), int(idx[-1])), 'valid_px': valid_px}

def transition_table(res, px_area_ha=0.01):
    """Off-diagonal transitions of a persistent_change result, largest first: dicts with from, to, pixels, area_ha, pct_of_stable."""
    t, sp = res['transitions'], max(res['stable_px'], 1); rows = []
    for a in range(NC):
        for b in range(NC):
            if a != b and t[a, b] > 0:
                rows.append({'from': CLASS_NAMES[a], 'to': CLASS_NAMES[b], 'pixels': int(t[a, b]), 'area_ha': round(float(t[a, b]) * px_area_ha, 1),
                             'pct_of_stable': round(float(t[a, b]) / sp * 100, 2), 'opposite_px': int(t[b, a]), 'net_px': int(t[a, b] - t[b, a]),
                             'net_area_ha': round(float(t[a, b] - t[b, a]) * px_area_ha, 1)})
    return sorted(rows, key=lambda r: -r['pixels'])

# ------------------------------------------------------------------ radar signature of the predicted classes
def class_signatures(comp, maps9, min_px=500):
    """comp (12, 9, 2, H, W) composites in dB (NaN = none), maps9 (12, 9, H, W) predicted classes -> mean and std of VV and VH per month and class
    (NaN where fewer than min_px valid pixels) and the pixel counts. Describes what the model calls each class, not independent ground truth."""
    mean = np.full((12, NC, 2), np.nan); std = np.full((12, NC, 2), np.nan); n = np.zeros((12, NC), np.int64)
    for k in range(comp.shape[0]):
        vv, vh = comp[k][:, 0], comp[k][:, 1]; fin = np.isfinite(vv) & np.isfinite(vh)
        for c in range(NC):
            sel = (maps9[k] == c) & fin; n[k, c] = int(sel.sum())
            if n[k, c] >= min_px:
                for b, band in enumerate((vv, vh)):
                    v = band[sel].astype(np.float64); mean[k, c, b], std[k, c, b] = v.mean(), v.std()
    return {'mean': mean, 'std': std, 'n': n}

# ------------------------------------------------------------------ comparison with CORINE Land Cover (Europe only, 100 m, 2018)
# Approximate mapping of the 44 CORINE classes to the six project classes, consistent with the WorldCover mapping used for the labels
# (snow and ice and moss/lichen count as Bare/Arid there). Mixed classes are a judgement call and a source of disagreement.
CLC_TO6 = {**{c: 3 for c in (111, 112, 121, 122, 123, 124)},                                           # artificial surfaces -> Urban
           **{c: 4 for c in (131, 132, 133, 331, 332, 333, 334, 335)},                                  # mines, dumps, construction, beaches, rock, sparse, burnt, ice -> Bare/Arid
           **{c: 1 for c in (141, 142, 231, 321, 322, 323, 324)},                                       # green urban, sport, pasture, grass, heath, scrub -> Shrub/Grass
           **{c: 2 for c in (211, 212, 213, 221, 222, 223, 241, 242, 243, 244)},                        # arable, permanent crops, mosaics -> Cropland
           **{c: 0 for c in (311, 312, 313)},                                                           # forests
           **{c: 5 for c in (411, 412, 421, 422, 423, 511, 512, 521, 522, 523)}}                        # wetlands and water
CORINE_ASSET = 'COPERNICUS/CORINE/V20/100m/{year}'

def clc_to_six(arr):
    """CORINE codes (e.g. 112) -> project classes 0..5; 255 for codes without a mapping (0 = outside the CORINE area)."""
    out = np.full(arr.shape, 255, np.uint8)
    for code, c in CLC_TO6.items():
        out[arr == code] = c
    return out

def corine_array(epsg, tf, year=2018):
    """CORINE codes on the same 1000 x 1000 grid (10 m, same CRS and transform) as the Sentinel-1 downloads."""
    import ee
    img = ee.Image(CORINE_ASSET.format(year=year)).select('landcover').unmask(0)
    arr = fetch_array(img, ['landcover'], epsg, tf)
    if arr is None:
        raise RuntimeError('the CORINE download failed')
    return arr[0].astype(np.int32)

def block_mode(m, block=10, min_valid=50):
    """Most frequent class in every block x block window (255 = ignored); blocks with fewer than min_valid valid pixels stay 255."""
    h, w = (m.shape[0] // block) * block, (m.shape[1] // block) * block
    a = m[:h, :w].reshape(h // block, block, w // block, block)
    counts = np.stack([(a == c).sum(axis=(1, 3)) for c in range(NC)], -1)
    out = counts.argmax(-1).astype(np.uint8); out[counts.sum(-1) < min_valid] = 255
    return out

def compare_maps(ref, pred):
    """Agreement of two class maps (255 ignored). Rows of the confusion matrix = reference, columns = prediction."""
    v = (ref != 255) & (pred != 255)
    cm = np.bincount(ref[v].astype(np.int64) * NC + pred[v], minlength=NC * NC).reshape(NC, NC).astype(np.float64)
    tp = np.diag(cm)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = tp / (cm.sum(0) + cm.sum(1) - tp); prec = tp / cm.sum(0); rec = tp / cm.sum(1)
        oa = float(tp.sum() / cm.sum()) if cm.sum() else float('nan')
    agree = np.full(ref.shape, 255, np.uint8); agree[v] = (ref[v] == pred[v]).astype(np.uint8)
    present = [c for c in range(NC) if cm[c].sum() + cm[:, c].sum() > 0]
    return {'cm': cm, 'oa': oa, 'iou': iou, 'precision': prec, 'recall': rec, 'miou': float(np.nanmean(iou[present])) if present else float('nan'),
            'n': int(cm.sum()), 'agree': agree}

def compare_with_corine(majority768, corine1000):
    """Majority map of the 12 months (768 x 768, 10 m) against CORINE on the same grid: at 100 m (the resolution of CORINE, the headline) and at 10 m."""
    m = (BOX_PX - 3 * TILE) // 2
    ref = clc_to_six(corine1000[m:m + 3 * TILE, m:m + 3 * TILE])
    ref_b, pred_b = block_mode(ref), block_mode(majority768)
    return {'block': compare_maps(ref_b, pred_b), 'pixel': compare_maps(ref, majority768), 'ref_blocks': ref_b, 'pred_blocks': pred_b,
            'ref_share': np.bincount(ref_b[ref_b != 255], minlength=NC) / max(int((ref_b != 255).sum()), 1) * 100,
            'pred_share': np.bincount(pred_b[pred_b != 255], minlength=NC) / max(int((pred_b != 255).sum()), 1) * 100}

# ------------------------------------------------------------------ the whole analysis
def analyse(backend, year, month, models, offsets, mean, std, device='cpu', cache_dir=None, workers=3, progress=None, with_corine=False, corine_year=2018, method='tiles'):
    """Download 12 months, build composites and context, predict, and compute all statistics (optionally also the CORINE comparison)."""
    months = download_window(backend, year, month, cache_dir, workers, progress)
    comp, ctx = assemble(months)
    if progress: progress(1.0, 'running the model')
    maps9 = predict_maps(comp, ctx, models, offsets, mean, std, device, method)
    maps = stitch(maps9)                                                                  # (12, 768, 768)
    shares = class_shares(maps); has = ~np.isnan(shares).any(axis=1)
    ftl = first_to_last(shares)
    res = {'window': window_months(year, month), 'n_images': [len(x) for x, _, _ in months], 'dates': [d for _, d, _ in months],
           'n_candidates': [c for _, _, c in months], 'maps': maps, 'shares': shares, 'has_data': has,
           'month_to_month': month_to_month(shares), 'first_to_last': ftl,
           'pixel_change_pct': pixel_change_pct(maps, ftl[1], ftl[2]) if ftl else None,
           'majority': majority_map(maps), 'valid_share': [float((m != 255).mean() * 100) for m in maps],
           'signatures': class_signatures(comp, maps9)}
    if with_corine:
        try:
            res['corine'] = compare_with_corine(res['majority'], backend.corine(corine_year)); res['corine_year'] = corine_year
        except Exception as e:                                   # the rest of the analysis must survive a failed comparison
            res['corine_error'] = f'{type(e).__name__}: {e}'
    return res
