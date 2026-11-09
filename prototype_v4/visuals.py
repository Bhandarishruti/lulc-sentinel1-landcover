"""
visuals.py: the look of the prototype. Colours, a hero banner with a dotted world map and a live pin, step cards, summary cards.
It contains no analysis code: every number shown here is passed in from app.py / lulc_core_v4.py.
"""
import os
import html as _h
from visuals_data import LAND_PATH, VB_W, VB_H, LAT_TOP, LAT_BOT

K = VB_W / 360.0
# held-out regions of the project (latitude range, longitude range), drawn as dashed boxes
REGIONS = {'Australia': (-45, -10, 112, 155), 'South Asia': (5, 35, 68, 98)}
SPLIT_STYLE = {'train': ('#FFE066', 'trained on'), 'val': ('#FF9ECD', 'validation'), 'test': ('#7DF9FF', 'test'),
               'region_australia': ('#8CFFB4', 'held-out places'), 'region_south_asia': ('#8CFFB4', 'held-out places')}


def _xy(lat, lon):
    x = (lon + 180.0) * K
    y = (LAT_TOP - max(min(lat, LAT_TOP), LAT_BOT)) * K
    return x, y


def load_scenes(path):
    """Optional scene_locations.csv (lat, lon, split) -> list of (lat, lon, split). Empty list if the file is missing or unreadable."""
    try:
        import pandas as pd
        if not os.path.exists(path):
            return []
        d = pd.read_csv(path)
        return [(float(a), float(b), str(c)) for a, b, c in zip(d['lat'], d['lon'], d['split'])]
    except Exception:
        return []


def css():
    return """<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Poppins', 'Segoe UI', system-ui, sans-serif; }
.stApp { background: linear-gradient(180deg,#eaf7ff 0%,#fbfdff 30%,#f4fff6 100%); color:#14213D; }
[data-testid="stHeader"] { background: rgba(244,250,255,.90); backdrop-filter: blur(8px); }
.block-container { padding-top: 1.3rem; max-width: 1280px; }
[data-testid="stSidebar"] h2 { color:#0b2e59; font-weight:700; }
[data-testid="stSidebar"] > div:first-child { background: linear-gradient(180deg,#d9f1ff 0%,#e4fff1 100%); border-right: 1px solid #bfe0f3; }
.stApp h2, .stApp h3 { color:#0b2e59; font-weight:700; }
[data-testid="stMain"] h2, [data-testid="stMain"] h3, section.main h2, section.main h3 { margin-top:1.4rem; padding-left:.7rem;
  border-left:8px solid #ff5e5b; border-image: linear-gradient(180deg,#ff5e5b,#ffb703 50%,#2ec4b6) 1; }
/* hero */
.hero { position:relative; overflow:hidden; border-radius:26px; min-height:330px; margin:0 0 18px 0; color:#fff;
  background: radial-gradient(800px 360px at 90% 0%, rgba(255,200,60,.50), transparent 60%),
              radial-gradient(700px 380px at 0% 110%, rgba(255,94,91,.40), transparent 60%),
              linear-gradient(115deg,#0a3d91 0%,#0b6fa4 38%,#12a594 72%,#6fd16b 100%);
  box-shadow: 0 18px 40px rgba(10,61,145,.28); }
.hero-map { position:absolute; right:0; top:0; width:100%; height:100%;
  -webkit-mask-image: linear-gradient(90deg, rgba(0,0,0,.30) 0%, rgba(0,0,0,.30) 26%, #000 54%); mask-image: linear-gradient(90deg, rgba(0,0,0,.30) 0%, rgba(0,0,0,.30) 26%, #000 54%); }
.hero-text { position:relative; z-index:2; padding:34px 38px; max-width:52%; }
.hero-badge { display:inline-block; font-size:.72rem; letter-spacing:.14em; font-weight:700; padding:6px 14px; border-radius:999px;
  background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.45); backdrop-filter: blur(4px); }
.hero-title { font-size:2.75rem; line-height:1.08; font-weight:800; margin:14px 0 10px 0; text-shadow:0 3px 18px rgba(0,0,0,.28); }
.hero-title span:first-of-type { background:linear-gradient(90deg,#ffe066,#ff9ecd); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-sub { font-size:1.02rem; line-height:1.5; opacity:.96; max-width:560px; text-shadow:0 2px 10px rgba(0,0,0,.25); }
.hero-chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
.hero-chip { font-size:.8rem; font-weight:600; padding:5px 13px; border-radius:999px; background:rgba(255,255,255,.92); color:#0b2e59; box-shadow:0 4px 12px rgba(0,0,0,.14); }
.hero-key { position:absolute; z-index:3; right:20px; bottom:14px; display:flex; flex-wrap:wrap; gap:12px; justify-content:flex-end; font-size:.72rem; font-weight:600;
  padding:6px 12px; border-radius:12px; background:rgba(8,36,84,.45); backdrop-filter: blur(5px); }
.hero-key i { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:middle; }
.ring { transform-box: fill-box; transform-origin: center; animation: ping 2.4s ease-out infinite; }
.ring.r2 { animation-delay:1.2s; }
@keyframes ping { 0% { transform: scale(1); opacity:.95; } 100% { transform: scale(7); opacity:0; } }
.sat { position:absolute; z-index:1; font-size:1.9rem; filter: drop-shadow(0 4px 8px rgba(0,0,0,.35)); animation: orbit 16s linear infinite; }
@keyframes orbit { 0% { left:-6%; top:56%; transform:rotate(-20deg);} 25% { left:20%; top:24%; } 50% { left:50%; top:6%; transform:rotate(0deg);} 75% { left:78%; top:20%; } 100% { left:106%; top:50%; transform:rotate(25deg);} }
/* legend, steps, kpis */
.legend { display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 18px 0; }
.chip { display:inline-flex; align-items:center; gap:8px; padding:6px 14px; border-radius:999px; background:#fff; font-weight:600; font-size:.85rem; color:#14213D; box-shadow:0 4px 14px rgba(20,33,61,.10); }
.chip i { width:14px; height:14px; border-radius:50%; display:inline-block; box-shadow: inset 0 0 0 2px rgba(255,255,255,.7); }
.steps, .kpis { display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; margin:8px 0 20px 0; }
.step, .kpi { position:relative; background:#fff; border-radius:18px; padding:16px 18px 16px 18px; box-shadow:0 8px 24px rgba(20,33,61,.09); border-top:6px solid var(--c); color:#14213D; }
.step b { display:block; font-size:1.02rem; margin:4px 0 4px 0; color:var(--c); }
.step p { margin:0; font-size:.86rem; line-height:1.45; color:#44546a; }
.step .ico { font-size:1.7rem; }
.step .num { position:absolute; right:14px; top:8px; font-size:1.9rem; font-weight:800; color:var(--c); opacity:.18; }
.kpi .l { font-size:.72rem; letter-spacing:.09em; text-transform:uppercase; font-weight:700; color:#5b6b82; }
.kpi .v { font-size:2rem; font-weight:800; line-height:1.15; color:#0b2e59; }
.kpi .s { font-size:.82rem; color:#5b6b82; }
.foot { margin:34px 0 8px 0; padding:16px 20px; border-radius:16px; background:linear-gradient(90deg,#e0f2ff,#e6fff2); font-size:.8rem; color:#44546a; text-align:center; }
/* streamlit widgets */
button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] { background:linear-gradient(90deg,#ff6a3d,#ff3d81); border:0; color:#fff; font-weight:700; border-radius:14px; padding:.6rem 1.6rem;
  box-shadow:0 8px 20px rgba(255,61,129,.35); transition: transform .12s ease, box-shadow .12s ease; }
button[data-testid="stBaseButton-primary"]:hover, .stButton > button[kind="primary"]:hover { transform: translateY(-2px); box-shadow:0 12px 26px rgba(255,61,129,.45); color:#fff; }
[data-testid="stDownloadButton"] button { border:2px solid #12a594; color:#0b6f63; font-weight:600; border-radius:14px; background:#fff; }
[data-testid="stDownloadButton"] button:hover { background:#12a594; color:#fff; border-color:#12a594; }
[data-testid="stMetric"] { background:#fff; border-radius:16px; padding:14px 18px; box-shadow:0 8px 22px rgba(20,33,61,.09); border-left:6px solid #2ec4b6; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color:#0b6fa4; font-weight:800; font-size:1.55rem !important; line-height:1.2 !important;
  white-space:normal !important; overflow:visible !important; text-overflow:clip !important; }
[data-testid="stDataFrame"] { border-radius:16px; overflow:hidden; box-shadow:0 8px 22px rgba(20,33,61,.10); border:1px solid #d7e8f5; }
[data-testid="stImage"] img { border-radius:18px; box-shadow:0 10px 28px rgba(20,33,61,.12); }
[data-testid="stAlert"] { border-radius:16px; }
[data-testid="stExpander"] { border-radius:16px; border:1px solid #d7e8f5; background:#fff; }
[data-baseweb="input"], [data-baseweb="input"] input, [data-baseweb="base-input"], [data-baseweb="select"] > div { background:#fff !important; color:#14213D !important; border-radius:12px; }
/* readable labels even if the browser is in dark mode and the theme file was not copied */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, [data-testid="stSidebar"] label, [data-testid="stSidebar"] label *, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color:#14213D !important; }
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p, [data-testid="stMain"] [data-testid="stMarkdownContainer"] li { color:#14213D; }
@media (max-width: 960px) { .steps, .kpis { grid-template-columns: repeat(2, 1fr); } .hero-text { max-width:100%; padding:26px 22px; } .hero-title { font-size:2.1rem; } .hero-key { display:none; } }
</style>"""


def hero_html(lat, lon, scenes=None):
    """Hero banner: title, and a dotted world map with a pulsing pin at (lat, lon). If scenes are given, the places the model was trained on are drawn too."""
    scenes = scenes or []
    px, py = _xy(lat, lon)
    boxes = ''.join(f'<rect x="{(lo0 + 180) * K:.1f}" y="{(LAT_TOP - la1) * K:.1f}" width="{(lo1 - lo0) * K:.1f}" height="{(la1 - la0) * K:.1f}" rx="6" fill="rgba(140,255,180,.10)" '
                    f'stroke="#8CFFB4" stroke-width="1.6" stroke-dasharray="6 5"/>' for (la0, la1, lo0, lo1) in REGIONS.values())
    dots = ''.join(f'<circle cx="{_xy(a, b)[0]:.1f}" cy="{_xy(a, b)[1]:.1f}" r="2.3" fill="{SPLIT_STYLE.get(s, ("#fff", ""))[0]}" stroke="rgba(8,36,84,.55)" stroke-width=".6"/>' for a, b, s in scenes)
    left = px > 760
    label = f'{abs(lat):.2f}°{"N" if lat >= 0 else "S"}, {abs(lon):.2f}°{"E" if lon >= 0 else "W"}'
    tx, anchor = (px - 14, 'end') if left else (px + 14, 'start')
    pin = (f'<g><circle class="ring" cx="{px:.1f}" cy="{py:.1f}" r="6" fill="none" stroke="#FFB703" stroke-width="2.2"/><circle class="ring r2" cx="{px:.1f}" cy="{py:.1f}" r="6" fill="none" stroke="#FFB703" stroke-width="2.2"/>'
           f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="#FFB703" stroke="#fff" stroke-width="2.4"/>'
           f'<text x="{tx:.1f}" y="{py + 4:.1f}" text-anchor="{anchor}" font-size="15" font-weight="700" fill="#fff" stroke="rgba(8,36,84,.65)" stroke-width="3" paint-order="stroke">{label}</text></g>')
    svg = (f'<svg class="hero-map" viewBox="0 0 {VB_W} {VB_H}" preserveAspectRatio="xMaxYMid meet" xmlns="http://www.w3.org/2000/svg">'
           f'<path d="{LAND_PATH}" fill="none" stroke="rgba(255,255,255,.62)" stroke-width="3.5" stroke-linecap="round"/>{boxes}{dots}{pin}</svg>')
    key = ''
    if scenes:
        seen = {}
        for _, _, s in scenes:
            col, name = SPLIT_STYLE.get(s, ('#fff', s)); seen[name] = (col, seen.get(name, (col, 0))[1] + 1)
        key = '<div class="hero-key">' + ''.join(f'<span><i style="background:{c}"></i>{_h.escape(n)} ({k})</span>' for n, (c, k) in seen.items()) + \
              '<span><i style="background:transparent;border:1.5px dashed #8CFFB4"></i>held-out regions</span><span><i style="background:#FFB703"></i>your place</span></div>'
    else:
        key = '<div class="hero-key"><span><i style="background:transparent;border:1.5px dashed #8CFFB4"></i>held-out regions</span><span><i style="background:#FFB703"></i>your place</span></div>'
    text = ('<div class="hero-text"><span class="hero-badge">🛰️ SENTINEL-1 RADAR · DEEP LEARNING</span>'
            '<div class="hero-title">Pick a place.<br>Get <span>12 monthly</span> <span style="white-space:nowrap">land-cover maps.</span></div>'
            '<div class="hero-sub">Radar sees through clouds. A deep-learning model turns Sentinel-1 backscatter into land-cover maps, class shares and change, at 10 m resolution.</div>'
            '<div class="hero-chips"><span class="hero-chip">📡 VV + VH radar only</span><span class="hero-chip">🌍 10 m</span><span class="hero-chip">🌳 6 classes</span><span class="hero-chip">🧠 FPN · ResNet-50</span></div></div>')
    return f'<div class="hero">{svg}<span class="sat">🛰️</span>{text}{key}</div>'


def legend_html(names, colors):
    return '<div class="legend">' + ''.join(f'<span class="chip"><i style="background:{c}"></i>{_h.escape(n)}</span>' for n, c in zip(names, colors)) + '</div>'


def steps_html():
    items = [('#0b6fa4', '📡', 'Radar', 'Sentinel-1 VV and VH backscatter, up to 5 passes per month.'),
             ('#12a594', '🧩', 'Composite', 'Monthly median at 10 m, plus the annual mean and spread as context.'),
             ('#ff9f1c', '🧠', 'Model', 'FPN with an ImageNet-pretrained ResNet-50, six land-cover classes.'),
             ('#ff3d81', '🗺️', 'Maps and change', '12 monthly maps, class shares, month-to-month and persistent change.')]
    return '<div class="steps">' + ''.join(f'<div class="step" style="--c:{c}"><span class="num">{i}</span><span class="ico">{ic}</span><b>{_h.escape(t)}</b><p>{_h.escape(d)}</p></div>' for i, (c, ic, t, d) in enumerate(items, 1)) + '</div>'


def kpi_html(cards):
    """cards: list of (label, value, sub, colour)."""
    return '<div class="kpis">' + ''.join(f'<div class="kpi" style="--c:{c}"><div class="l">{_h.escape(l)}</div><div class="v">{_h.escape(v)}</div><div class="s">{_h.escape(s)}</div></div>' for l, v, s, c in cards) + '</div>'


def footer_html():
    return ('<div class="foot">Radar data: Copernicus Sentinel-1 via Google Earth Engine · training labels: ESA WorldCover 2021 · case-study reference: CORINE Land Cover 2018 · '
            'map outline: Natural Earth · the maps are model output and an upper bound for real change, not ground truth</div>')
