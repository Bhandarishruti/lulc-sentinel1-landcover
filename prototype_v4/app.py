"""
Streamlit prototype: 12 monthly land cover maps from Sentinel-1 only (v4 models).
Run:  streamlit run app.py
"""
import os, io
import numpy as np, pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import lulc_core_v4 as core
import visuals

st.set_page_config(page_title='LULC from Sentinel-1 (v4)', page_icon='🛰️', layout='wide')
HERE = os.path.dirname(os.path.abspath(__file__))
CMAP = ListedColormap(core.CLASS_COLORS + [core.NODATA_COLOR])

def to_rgb_index(m):
    """class map with 255 = no data -> index into CMAP."""
    return np.where(m == 255, core.NC, m)

@st.cache_data(show_spinner=False)
def load_scenes(path):
    """Optional scene_locations.csv next to app.py (made by export_scene_locations.py): the places the model was trained on, drawn on the hero map."""
    return visuals.load_scenes(path)

def kpi_cards(r):
    """Summary cards shown above the maps; every value comes from the result dictionary."""
    cards = [('Sentinel-1 images', str(sum(r['n_images'])), 'used for the 12 months', '#0b6fa4')]
    try:
        mean_sh = np.nanmean(r['shares'], axis=0); k = int(np.nanargmax(mean_sh))
        cards.append(('Most common class', core.CLASS_NAMES[k], f'{mean_sh[k]:.1f}% of pixels on average', core.CLASS_COLORS[k]))
    except Exception:
        pass
    if r.get('pixel_change_pct') is not None:
        cards.append(('First-to-last change', f'{r["pixel_change_pct"]:.1f}%', 'of pixels (upper bound for real change)', '#ff9f1c'))
    flat = np.abs(r['month_to_month'])
    if np.isfinite(flat).any():
        i, c = np.unravel_index(np.argmax(np.where(np.isnan(flat), -1, flat)), flat.shape)
        cards.append(('Largest monthly move', f'{float(flat[i, c]):.1f} pp', f'{core.CLASS_NAMES[c]}, {month_label(*r["window"][i])} → {month_label(*r["window"][i + 1])}', '#ff3d81'))
    return cards

@st.cache_resource(show_spinner='Loading the models ...')
def get_models(weights_dir, mode):
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    return core.load_models(weights_dir, mode, device) + (device,)

def month_label(y, m):
    return f'{core.MONTH_NAMES[m - 1]} {y}'

def shares_table(r):
    rows = []
    for (y, m), s, n in zip(r['window'], r['shares'], r['n_images']):
        rows.append({'Month': month_label(y, m), 'Images used': n, **{c: (None if np.isnan(v) else round(float(v), 1)) for c, v in zip(core.CLASS_NAMES, s)}})
    return pd.DataFrame(rows)

def change_table(r):
    rows = []
    for i in range(11):
        d = r['month_to_month'][i]
        a, b = r['window'][i], r['window'][i + 1]
        rows.append({'From → to': f'{month_label(*a)} → {month_label(*b)}', **{c: (None if np.isnan(v) else round(float(v), 1)) for c, v in zip(core.CLASS_NAMES, d)}})
    return pd.DataFrame(rows)

def maps_figure(r):
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.6))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(to_rgb_index(r['maps'][k]), cmap=CMAP, vmin=0, vmax=core.NC, interpolation='nearest')
        ax.set_title(month_label(*r['window'][k]) + ('' if r['has_data'][k] else ' (no data)'), fontsize=10); ax.axis('off')
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(core.CLASS_COLORS + [core.NODATA_COLOR], core.CLASS_NAMES + ['No data'])],
               loc='lower center', ncol=7, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig

def majority_figure(r):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(to_rgb_index(r['majority']), cmap=CMAP, vmin=0, vmax=core.NC, interpolation='nearest'); ax.axis('off')
    ax.set_title('Most frequent class over the 12 months'); fig.tight_layout()
    return fig

CMAP_CHG = ListedColormap(core.CLASS_COLORS + ['#cfcfcf', '#ffffff'])          # 0-5 classes, 6 = no persistent change, 7 = unstable or no data
CMAP_AGREE = ListedColormap(['#c62828', '#2e7d32', '#d9d9d9'])                  # 0 disagree, 1 agree, 2 not compared

def span_label(window, idx):
    """The months actually used, e.g. 'Jan, Feb, Apr 2020' (a month without data is not listed)."""
    yrs = {window[i][0] for i in idx}
    return (', '.join(core.MONTH_NAMES[window[i][1] - 1] for i in idx) + f' {yrs.pop()}') if len(yrs) == 1 else ', '.join(month_label(*window[i]) for i in idx)

def persistent_figure(p, window):
    chg = np.full(p['early'].shape, 7, np.uint8); chg[p['stable']] = 6; chg[p['changed']] = p['late'][p['changed']]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    axes[0].imshow(to_rgb_index(p['early']), cmap=CMAP, vmin=0, vmax=core.NC, interpolation='nearest'); axes[0].set_title('Majority class, ' + span_label(window, p['early_months']), fontsize=10)
    axes[1].imshow(to_rgb_index(p['late']), cmap=CMAP, vmin=0, vmax=core.NC, interpolation='nearest'); axes[1].set_title('Majority class, ' + span_label(window, p['late_months']), fontsize=10)
    axes[2].imshow(chg, cmap=CMAP_CHG, vmin=0, vmax=7, interpolation='nearest'); axes[2].set_title('Persistent change (colour = the new class)', fontsize=10)
    for ax in axes: ax.axis('off')
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(core.CLASS_COLORS, core.CLASS_NAMES)] +
                       [Patch(facecolor='#cfcfcf', label='No persistent change'), Patch(facecolor='#ffffff', edgecolor='#999999', label='No stable majority / no data')],
               loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig

def signature_figure(r):
    sg = r['signatures']; x = np.arange(12); labels = [core.MONTH_NAMES[m - 1] for _, m in r['window']]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharex=True)
    for b, (ax, name) in enumerate(zip(axes, ('VV', 'VH'))):
        for c in range(core.NC):
            m, s = sg['mean'][:, c, b], sg['std'][:, c, b]
            if np.isfinite(m).any():
                ax.plot(x, m, marker='o', ms=3, color=core.CLASS_COLORS[c], label=core.CLASS_NAMES[c])
                ax.fill_between(x, m - s, m + s, color=core.CLASS_COLORS[c], alpha=0.10)
        ax.set_title(f'{name} backscatter of the predicted classes (dB)', fontsize=10); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45); ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, frameon=False, fontsize=9); fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig

def corine_figure(c):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    axes[0].imshow(to_rgb_index(c['pred_blocks']), cmap=CMAP, vmin=0, vmax=core.NC, interpolation='nearest'); axes[0].set_title('This app: most frequent class, 100 m blocks', fontsize=10)
    axes[1].imshow(to_rgb_index(c['ref_blocks']), cmap=CMAP, vmin=0, vmax=core.NC, interpolation='nearest'); axes[1].set_title('CORINE (mapped to the six classes)', fontsize=10)
    ag = c['block']['agree']; axes[2].imshow(np.where(ag == 255, 2, ag), cmap=CMAP_AGREE, vmin=0, vmax=2, interpolation='nearest'); axes[2].set_title('Agreement (green) and disagreement (red)', fontsize=10)
    for ax in axes: ax.axis('off')
    fig.legend(handles=[Patch(color=c_, label=n) for c_, n in zip(core.CLASS_COLORS + [core.NODATA_COLOR], core.CLASS_NAMES + ['No data'])], loc='lower center', ncol=7, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig

def csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header('Settings')
    project = st.text_input('Earth Engine project', 'friendly-aurora-426714-v0')
    weights = st.text_input('Weights folder', os.path.join(HERE, 'weights'))
    mode = st.radio('Model', list(core.MODES), format_func=lambda v: core.MODES[v])
    cache = st.text_input('Download cache folder', os.path.join(HERE, 'cache'))
    workers = st.slider('Parallel downloads', 1, 4, 3)
    method = st.selectbox('Prediction method', list(core.PREDICT_METHODS), index=list(core.PREDICT_METHODS).index(core.DEFAULT_METHOD),
                          format_func=lambda v: core.PREDICT_METHODS[v],
                          help='Tile by tile is how the models were trained and evaluated, and it leaves visible straight seams at the tile borders (on the validation scenes the class changes across '
                               'a border for 18.7% of pixel pairs against 1.3% elsewhere). Whole area and overlapping windows predict across the borders and remove the seams without lowering '
                               'the validation mIoU (v4b_seam_test.py).')
    use_corine = st.checkbox('Compare with CORINE 2018 (Europe only)', value=False, help='Adds an indicative comparison with the CORINE Land Cover 2018 map (100 m). Needs one more Earth Engine request.')
    st.caption('Earth Engine must be authorised once on this computer (earthengine authenticate).')

st.markdown(visuals.css(), unsafe_allow_html=True)
hero_slot = st.empty()          # filled below, once the latitude and longitude are known (the pin on the world map follows them)
st.write('Enter a location and a start month. The app downloads Sentinel-1 for the 12 months, builds monthly median composites '
         '(at most 5 passes per month, 10 m, as in the training data), classifies a 7.7 km × 7.7 km area around the point into six classes, '
         'and reports class shares and changes.')

c1, c2, c3, c4 = st.columns(4)
lat = c1.number_input('Latitude', -80.0, 80.0, 49.41, 0.01, format='%.4f')
lon = c2.number_input('Longitude', -180.0, 180.0, 8.68, 0.01, format='%.4f')
year = c3.number_input('Start year', 2015, 2030, 2020, 1)
month = c4.selectbox('Start month', list(range(1, 13)), format_func=lambda m: core.MONTH_NAMES[m - 1])

hero_slot.markdown(visuals.hero_html(float(lat), float(lon), load_scenes(os.path.join(HERE, 'scene_locations.csv'))), unsafe_allow_html=True)
st.markdown(visuals.legend_html(core.CLASS_NAMES, core.CLASS_COLORS), unsafe_allow_html=True)
ok, err, warns = core.check_window(int(year), int(month))
if not ok:
    st.error(err)
for w in warns:
    st.warning(w)

if st.button('Run analysis', type='primary', disabled=not ok):
    try:
        models, offsets, mean, std, device = get_models(weights, mode)
        bar = st.progress(0.0, text='Connecting to Earth Engine ...')
        backend = core.EEBackend(float(lat), float(lon), project)
        key = f'{lat:.4f}_{lon:.4f}'.replace('-', 'm')
        res = core.analyse(backend, int(year), int(month), models, offsets, mean, std, device,
                           cache_dir=os.path.join(cache, key), workers=workers, progress=lambda f, t: bar.progress(min(f, 1.0), text=t),
                           with_corine=use_corine, method=method)
        bar.empty()
        st.session_state['result'] = res; st.session_state['meta'] = (float(lat), float(lon), mode, method)
    except Exception as e:
        st.session_state.pop('result', None)
        st.error(f'The analysis failed ({type(e).__name__}): {e}')

r = st.session_state.get('result')
if not r:
    st.markdown(visuals.steps_html(), unsafe_allow_html=True)
if r:
    la, lo, md, mt = st.session_state['meta']
    st.success(f'Done: {la:.4f}, {lo:.4f} | {month_label(*r["window"][0])} to {month_label(*r["window"][-1])} | '
               f'{core.MODES[md]} | {core.PREDICT_METHODS[mt]} | {sum(r["n_images"])} Sentinel-1 images used')
    missing = [month_label(*r['window'][k]) for k in range(12) if not r['has_data'][k]]
    if missing:
        st.warning('No usable image in: ' + ', '.join(missing) + '. These months are left out of all shares and changes.')
    few = [f'{month_label(*r["window"][k])}: {r["n_images"][k]}' for k in range(12) if 0 < r['n_images'][k] < 4]
    if few:
        st.info('Months with fewer than 4 images (' + ', '.join(few) + ') are noisier: the monthly median keeps more speckle. In one measured run (Singapore) months with 2 to 3 images showed about 2.7 points more Urban '
                'and 2.6 points less Forest than months with 4 or 5 images, which can look like change. Read the persistent-change section below with this in mind.')
    st.markdown(visuals.kpi_html(kpi_cards(r)), unsafe_allow_html=True)
    st.subheader('🗺️ Monthly maps')
    st.pyplot(maps_figure(r))
    st.subheader('📊 Class shares per month (% of pixels with data)')
    st1 = shares_table(r); st.dataframe(st1, hide_index=True, use_container_width=True)
    st.subheader('📈 Month-to-month change (percentage points)')
    ch = change_table(r); st.dataframe(ch, hide_index=True, use_container_width=True)
    flat = np.abs(r['month_to_month']); maxchg = None
    if np.isfinite(flat).any():
        i, c = np.unravel_index(np.argmax(np.where(np.isnan(flat), -1, flat)), flat.shape); maxchg = float(flat[i, c])
        st.caption(f'Largest change between two consecutive months in this run: {maxchg:.1f} percentage points '
                   f'({core.CLASS_NAMES[c]}, {month_label(*r["window"][i])} → {month_label(*r["window"][i + 1])}).')
    st.subheader('🧭 Total change, first to last month with data')
    ftl = r['first_to_last']
    if ftl is None:
        st.info('Fewer than two months with data: no total change can be computed.')
    else:
        d, i, j = ftl
        tot = pd.DataFrame([{'Class': c, f'{month_label(*r["window"][i])} (%)': round(float(r['shares'][i][k]), 1),
                             f'{month_label(*r["window"][j])} (%)': round(float(r['shares'][j][k]), 1), 'Change (pp)': round(float(d[k]), 1)}
                            for k, c in enumerate(core.CLASS_NAMES)])
        st.dataframe(tot, hide_index=True, use_container_width=True)
        st.write(f'Pixels whose class differs between the two months: **{r["pixel_change_pct"]:.1f}%**. '
                 'This counts classification errors and seasonal effects as change, so it is an upper bound for real change.')
    left, right = st.columns([1, 1])
    with left:
        st.pyplot(majority_figure(r))
    with right:
        st.download_button('Download class shares (CSV)', csv_bytes(st1), 'class_shares.csv', 'text/csv')
        st.download_button('Download month-to-month changes (CSV)', csv_bytes(ch), 'month_to_month_change.csv', 'text/csv')
        buf = io.BytesIO(); maps_figure(r).savefig(buf, format='png', dpi=150, bbox_inches='tight')
        st.download_button('Download the 12 maps (PNG)', buf.getvalue(), 'monthly_maps.png', 'image/png')
    # ---------------------------------------------------------------- persistent change
    st.subheader('🔁 Persistent change: only what stays changed')
    st.write('The most frequent class of the first months with data is compared with the most frequent class of the last months with data. '
             'A pixel counts as changed only if both are stable (at least 2 of the months agree) and different, so a single noisy month cannot create change.')
    kv = st.slider('Months at each end', 1, 4, 3, key='k_vote')
    p = core.persistent_change(r['maps'], kv)
    if p is None:
        st.info('Fewer than two months with data: no persistent change can be computed.')
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric('Persistent change', f'{p["changed_pct"]:.1f}% of stable pixels', help='Pixels whose stable majority class differs between the two ends, as a share of pixels that are stable at both ends.')
        k2.metric('Raw first-to-last change', f'{p["raw_changed_pct"]:.1f}% of pixels', help='Pixels whose class differs between the single first and last month with data (the number shown above).')
        k3.metric('No stable majority', f'{p["unstable_pct"]:.1f}% of pixels', help='Pixels that flip between classes within an end window, so they are not counted either way.')
        n1, n2 = st.columns(2)
        n1.metric('Changed pixels that are two-way swaps', f'{p["balanced_pct"]:.0f}%' if np.isfinite(p['balanced_pct']) else '-',
                  help='Share of the changed pixels that is matched by pixels moving the opposite way between the same two classes (for example Urban to Forest and Forest to Urban).')
        n2.metric('Net one-way change', f'{p["net_pct_of_stable"]:.2f}% of stable pixels' if np.isfinite(p['net_pct_of_stable']) else '-',
                  help='Persistent change left after cancelling every two-way swap: the more careful estimate of one-way conversion.')
        st.pyplot(persistent_figure(p, r['window']))
        trows = core.transition_table(p)
        if trows:
            tdf = pd.DataFrame(trows).rename(columns={'from': 'From', 'to': 'To', 'pixels': 'Pixels', 'area_ha': 'Area (ha)', 'pct_of_stable': '% of stable pixels',
                                                      'opposite_px': 'Opposite direction (px)', 'net_px': 'Net (px)', 'net_area_ha': 'Net area (ha)'})
            tdf = tdf[['From', 'To', 'Pixels', 'Area (ha)', 'Opposite direction (px)', 'Net (px)', 'Net area (ha)', '% of stable pixels']]
            st.write(f'What turned into what (largest {min(10, len(tdf))} of {len(tdf)} transitions; one pixel = 10 m x 10 m = 0.01 ha):')
            st.dataframe(tdf.head(10), hide_index=True, use_container_width=True)
            st.download_button('Download all transitions (CSV)', csv_bytes(tdf), 'persistent_transitions.csv', 'text/csv')
        else:
            st.info('No persistent change between the two ends of the window.')
        st.caption(f'Uses {p["k"]} month(s) at each end ({span_label(r["window"], p["early_months"])} and {span_label(r["window"], p["late_months"])}). '
                   'The vote removes one-month flicker, not errors that persist for several months, so this is still an upper bound for real change. '
                   'Two-way swaps between the same two classes mostly cancel and are typical of jitter along class boundaries; they can also be real, so the net columns are the more careful estimate of one-way conversion.')

    # ---------------------------------------------------------------- radar signature
    st.subheader('📡 Radar signature of the predicted classes')
    st.pyplot(signature_figure(r))
    st.caption('Mean VV and VH backscatter (dB) of the pixels the model assigns to each class, month by month; the band is one standard deviation; a class needs at least 500 pixels '
               'in a month to be drawn. This shows what the model calls each class. It is not independent ground truth, because the same radar values drove the prediction.')

    # ---------------------------------------------------------------- CORINE
    if 'corine' in r:
        cc = r['corine']; bb = cc['block']
        st.subheader(f'🇪🇺 Comparison with CORINE Land Cover {r["corine_year"]} (Europe only)')
        if bb['n'] == 0:
            st.warning('No CORINE data cover this area (CORINE exists for Europe only), so nothing could be compared.')
        else:
            q1, q2, q3 = st.columns(3)
            q1.metric('Agreement at 100 m', f'{bb["oa"] * 100:.1f}%'); q2.metric('mIoU at 100 m', f'{bb["miou"]:.3f}'); q3.metric('Agreement at 10 m', f'{cc["pixel"]["oa"] * 100:.1f}%')
            pc = pd.DataFrame({'Class': core.CLASS_NAMES, 'CORINE share (%)': cc['ref_share'], 'This app (%)': cc['pred_share'], 'IoU': bb['iou'],
                               'Precision': bb['precision'], 'Recall': bb['recall']}).round(3)
            st.dataframe(pc, hide_index=True, use_container_width=True)
            st.pyplot(corine_figure(cc))
            with st.expander('Confusion matrix at 100 m (% of each CORINE class that the app assigns to ...)'):
                rows = [i for i in range(core.NC) if bb['cm'][i].sum() > 0]
                st.dataframe(pd.DataFrame(bb['cm'][rows] / bb['cm'][rows].sum(1, keepdims=True) * 100, index=[core.CLASS_NAMES[i] for i in rows], columns=core.CLASS_NAMES).round(1))
        st.caption('Indicative only, not an accuracy assessment: CORINE is a 100 m map with a 25 ha minimum mapping unit from 2018, the app maps 10 m pixels of the chosen year, '
                   'and the 44 CORINE classes were mapped to the six project classes by an approximate rule (mixed classes such as green urban areas and mosaics are a judgement call).')
    elif 'corine_error' in r:
        st.warning('The CORINE comparison could not be made: ' + r['corine_error'])
    elif use_corine:
        st.info('Run the analysis again to add the CORINE comparison.')
    st.subheader('💡 How to read the changes')
    st.markdown(
        '- The models classify each month from radar backscatter alone. Differences between months mix **real change, seasonal backscatter effects '
        '(crop cycle, soil moisture, snow, wet leaves), different numbers of passes per month, and classification errors**.\n'
        '- Shares are percentages of the pixels with data in that month. A month without an image is skipped, not treated as zero.\n'
        '- **How much change is noise?** Measured on 41 validation places and 2 years each (82 place-years, whole-area method; the other methods are similar): the median place-year shows '
        '**2.9% of pixels changing** class between its first and last month and **2.6%** persistent change; 90% of place-years stay below 8.5% and 7.3%; the largest monthly move of any class '
        'has a median of 2.6 pp (90% below 10.9 pp). These places contain real change as well, so the figures are **upper bounds for false change**: a result near 3% is within the noise. '
        'The worst place-year moved by 65 pp in one month, so large swings do occur (real events or artifacts, not distinguished). '
        + (f'In this run the largest monthly move was **{maxchg:.1f} pp**. ' if maxchg is not None else '') +
        'The app has **not been tested on a place with known real change**, so a quiet result does not prove that real change would show up.\n'
        '- The training labels are ESA WorldCover 2021; the models were trained on 2020 and 2021 imagery. Other years are an untested transfer.')

st.markdown(visuals.footer_html(), unsafe_allow_html=True)
