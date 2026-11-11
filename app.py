"""
Flight Fare Prediction - Streamlit web app
Developed by: Sindhu S

Pages: Home | Quick Prediction | Smart Analytics | Plan Better | About

Run with:   streamlit run app.py

Files expected next to this script:
    Flight_Fare.xlsx            (the dataset used in the notebook)   <- required
    flight_price_model.pkl      (saved in the notebook)               <- optional
    feature_columns.pkl         (saved in the notebook)               <- optional
If the two .pkl files are missing, the app trains the same Tuned Random Forest
(max_depth=20, n_estimators=100) once at start-up.
"""
from __future__ import annotations

import base64
import calendar
import inspect
import io
import itertools
from datetime import date, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import flight_utils as fu

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
APP_TITLE = "Flight Fare Prediction"
DEVELOPER = "Sindhu S"
GITHUB_URL = "https://github.com/"  # <- paste your own GitHub repo link here (set "" to hide the icon)

PAGES = ["Home", "Quick Prediction", "Smart Analytics", "Plan Better", "About"]

PURPLE = "#6a40f5"
PURPLE_LIGHT = "#a78bfa"
BLUE = "#4f8cff"
INK = "#1c1b3a"
MUTED = "#6b6a8a"
GRID = "#ece9fb"
DONUT_COLORS = ["#4c3ae0", "#6a40f5", "#4f8cff", "#8b7cff", "#a9c4ff", "#c9b8ff"]

st.set_page_config(page_title=APP_TITLE, page_icon="✈️", layout="wide")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stApp * { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: linear-gradient(180deg,#ebe7ff 0%,#f5f3ff 42%,#eeeaff 100%); color:#1c1b3a; }
header[data-testid="stHeader"], footer, #MainMenu { display:none !important; visibility:hidden; }
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: none !important; width: 100% !important; margin: 0 !important; box-sizing: border-box !important;
  padding-top: 2.4rem !important; padding-bottom: 3rem !important;
  padding-left: max(1.5rem, calc((100vw - 1200px) / 2)) !important;
  padding-right: max(1.5rem, calc((100vw - 1200px) / 2)) !important;
}
[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main, .stApp { overflow-x: hidden; }

/* ---------- cards: card() via card() ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]:not(:has(div[class*="st-key-card_"])),
div[class*="st-key-card_"] {
  background: rgba(255,255,255,.74) !important; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,.85) !important;
  border-radius: 18px !important;
  box-shadow: 0 8px 26px rgba(106,64,245,.08);
}
/* if Streamlit wraps the keyed card in its own bordered wrapper, hide the wrapper's frame */
div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="st-key-card_"]) {
  background: transparent !important; border-color: transparent !important; box-shadow: none !important;
}

/* ---------- top navigation: plain buttons, centred between brand and GitHub ---------- */
div[class*="st-key-nav_bar"] [data-testid="stHorizontalBlock"],
div[data-testid="stHorizontalBlock"]:has(.nav-mark) { gap:.25rem !important; align-items:center; }
div[data-testid="stHorizontalBlock"]:has(.nav-mark) {
  background:rgba(255,255,255,.78); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  border:1px solid rgba(255,255,255,.9); border-radius:999px; padding:.55rem 1.5rem !important;
  box-shadow:0 8px 28px rgba(106,64,245,.14);
}
div[class*="st-key-nav_bar"] button,
div[data-testid="stHorizontalBlock"]:has(.nav-mark) button {
  background:transparent !important; box-shadow:none !important; border:none !important;
  border-bottom:2px solid transparent !important; border-radius:0 !important;
  color:#2b2a4c !important; padding:.4rem .1rem !important; white-space:nowrap;
}
div[class*="st-key-nav_bar"] button p,
div[data-testid="stHorizontalBlock"]:has(.nav-mark) button p { font-size:.9rem !important; font-weight:600 !important; margin:0 !important; color:inherit !important; }
div[class*="st-key-nav_bar"] button:hover,
div[data-testid="stHorizontalBlock"]:has(.nav-mark) button:hover { color:#6a40f5 !important; }
div[class*="st-key-nav_bar"] button[kind="primary"],
div[class*="st-key-nav_bar"] button[data-testid="stBaseButton-primary"],
div[data-testid="stHorizontalBlock"]:has(.nav-mark) button[kind="primary"],
div[data-testid="stHorizontalBlock"]:has(.nav-mark) button[data-testid="stBaseButton-primary"] {
  color:#6a40f5 !important; border-bottom:2px solid #6a40f5 !important; background:transparent !important; box-shadow:none !important;
}

.brand { display:flex; align-items:center; gap:.6rem; font-weight:800; font-size:1.12rem; color:#1c1b3a; transform:translateY(-.65rem); }
.brand svg { width:34px; height:34px; }
.gh { display:flex; justify-content:flex-end; align-items:center; transform:translateY(-.55rem); }
.gh a { color:#1c1b3a; }
.gh svg { width:30px; height:30px; }

/* ---------- typography ---------- */
h1.hero { font-size:4.8rem; line-height:1.04; font-weight:800; margin:.4rem 0 .6rem; letter-spacing:-.02em; color:#1c1b3a; padding:0; }
h1.page { font-size:3.2rem; line-height:1.05; font-weight:800; margin:.2rem 0 .5rem; letter-spacing:-.02em; color:#1c1b3a; padding:0; }
.grad { background:linear-gradient(90deg,#5b3df5,#8b5cf6); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
p.lead { font-size:1.05rem; color:#2b2a4c; max-width:34rem; margin:0 0 1rem; }
.badge { display:inline-block; background:#e6defd; color:#4a2fd1; font-weight:700; font-size:.8rem; padding:.28rem .8rem; border-radius:999px; }
.card-title { font-weight:800; font-size:1.08rem; color:#1c1b3a; margin-bottom:.4rem; }
.small { color:#6b6a8a; font-size:.82rem; }

/* ---------- widgets ---------- */
label p, .stSlider label p { font-weight:600 !important; font-size:.82rem !important; color:#2b2a4c !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"] { border-radius:10px !important; }
div[data-testid="stButton"], div[data-testid="stDownloadButton"], .stButton, .stDownloadButton { width:100% !important; }
.stButton > button, .stDownloadButton > button,
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
  width:100% !important; border-radius:12px; font-weight:700; padding:.55rem 1rem;
  border:1px solid #cfc4fb; background:#fff; color:#4a2fd1; justify-content:center; text-align:center;
}
.stButton > button:hover, .stDownloadButton > button:hover,
div[data-testid="stButton"] button:hover, div[data-testid="stDownloadButton"] button:hover { border-color:#6a40f5; color:#6a40f5; }
.stButton > button[kind="primary"], div[data-testid="stButton"] button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background:linear-gradient(90deg,#7b4dff,#5b3df5) !important; color:#fff !important; border:none !important;
  box-shadow:0 8px 18px rgba(106,64,245,.28);
}
.stDownloadButton > button, div[data-testid="stDownloadButton"] button { background:linear-gradient(90deg,#7b4dff,#5b3df5); color:#fff; border:none; }
.stDownloadButton > button:hover, div[data-testid="stDownloadButton"] button:hover { color:#fff; filter:brightness(1.06); }

/* ---------- small components ---------- */
.ficon { width:44px; height:44px; border-radius:12px; background:#efeaff; display:flex; align-items:center; justify-content:center; font-size:1.35rem; margin:0 auto .5rem; }
.ftitle { font-weight:800; font-size:1.05rem; text-align:center; }
.stat { display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; gap:.2rem; background:rgba(255,255,255,.74); backdrop-filter:blur(12px); border:1px solid rgba(255,255,255,.85); border-radius:16px; padding:.9rem 1rem; box-shadow:0 6px 18px rgba(106,64,245,.07); }
.stat .si { width:38px; height:38px; border-radius:10px; background:#efeaff; display:flex; align-items:center; justify-content:center; font-size:1.1rem; margin-bottom:.15rem; }
.stat .sl { font-size:.78rem; color:#6b6a8a; font-weight:600; text-align:center; }
.stat .sv { font-size:1.1rem; font-weight:800; color:#1c1b3a; text-align:center; }
.kpi { text-align:center; background:rgba(255,255,255,.74); backdrop-filter:blur(12px); border:1px solid rgba(255,255,255,.85); border-radius:16px; padding:.9rem .5rem; box-shadow:0 6px 18px rgba(106,64,245,.07); }
.kpi .ki { font-size:1.3rem; }
.kpi .kl { font-size:.85rem; font-weight:700; color:#2b2a4c; }
.kpi .kv { font-size:1.35rem; font-weight:800; color:#1c1b3a; }
.fare-box { text-align:center; padding:.3rem .2rem; }
.fare-box .fl { font-weight:700; color:#2b2a4c; }
.fare-box .fv { font-size:2.5rem; font-weight:800; color:#1c1b3a; letter-spacing:-.02em; }
.kv-row { display:flex; justify-content:space-between; font-size:.88rem; padding:.28rem 0; border-top:1px solid #eee9ff; color:#2b2a4c; }
.kv-row b { color:#1c1b3a; }
.newfare { background:#efe9ff; border-radius:14px; padding:.5rem .9rem; margin:.5rem 0 .2rem; }
.newfare .nfv { font-size:1.5rem; font-weight:800; color:#1c1b3a; }
.fare-sub { font-weight:800; color:#1c1b3a; text-align:center; margin:.5rem 0 .3rem; }
.kv-row.total { border-top:2px solid #d9d0fb; font-weight:700; }
div[class*="st-key-card_eq_result"] { background:linear-gradient(160deg,rgba(236,228,255,.92),rgba(247,243,255,.8)) !important; min-height:330px; }
div[class*="st-key-card_eq_whatif"], div[class*="st-key-card_eq_history"] { min-height:150px; }
div[data-testid="stDownloadButton"], .stDownloadButton { width:100% !important; }
div[data-testid="stDownloadButton"] button, .stDownloadButton > button {
  width:100% !important; min-height:2.7rem !important; border-radius:14px !important; border:none !important;
  background:linear-gradient(135deg,#7a58f6,#5a3cf0) !important; color:#fff !important; font-weight:700 !important;
  box-shadow:0 8px 20px rgba(106,64,245,.3) !important; white-space:nowrap !important; opacity:1 !important;
}
div[data-testid="stDownloadButton"] button p, .stDownloadButton > button p { color:#fff !important; margin:0 !important; font-size:.95rem !important; white-space:nowrap !important; }
div[data-testid="stDownloadButton"] button:disabled, .stDownloadButton > button:disabled { opacity:.55 !important; color:#fff !important; }
div[data-testid="stElementContainer"]:has(.stDownloadButton), div[data-testid="stElementContainer"]:has([data-testid="stDownloadButton"]) { width:100% !important; }
div[class*="st-key-predict_btn"], div[class*="st-key-predict_btn"] > div, div[class*="st-key-predict_btn"] button { width:100% !important; }
div[class*="st-key-predict_btn"] button { min-height:3rem !important; font-size:1rem !important; border-radius:14px !important; }
.tbl { width:100%; border-collapse:collapse; font-size:.86rem; }
.tbl th { text-align:left; color:#6b6a8a; font-weight:700; padding:.35rem .4rem; border-bottom:1px solid #e9e4fb; }
.tbl td { padding:.42rem .4rem; border-bottom:1px solid #f1eefc; color:#1c1b3a; }
.tbl tr.best td { background:#f3efff; }
.pill { display:inline-block; font-size:.74rem; font-weight:700; padding:.12rem .55rem; border-radius:999px; background:#dff7e7; color:#127a3c; border:1px solid #b6ebc7; }
.pill.warn { background:#fff1dc; color:#9a5b00; border-color:#ffd9a3; }
div[class*="st-key-card_eq_tips"] .tip { font-size:1.2rem; padding:.85rem 0; }
.check.check-big { font-size:1.2rem; margin:.45rem 0; }
.check.check-big i { width:24px; height:24px; }
.tip { display:flex; gap:.6rem; align-items:flex-start; padding:.42rem 0; font-size:.9rem; color:#2b2a4c; }
.tip .ti { width:30px; height:30px; flex:0 0 30px; border-radius:9px; background:#efeaff; display:flex; align-items:center; justify-content:center; }
.check { display:flex; align-items:center; gap:.55rem; font-size:.98rem; color:#2b2a4c; margin:.25rem 0; }
.check i { width:20px; height:20px; border-radius:50%; background:#6a40f5; color:#fff; font-style:normal; font-size:.7rem; display:flex; align-items:center; justify-content:center; }
.note { text-align:center; color:#6b6a8a; font-size:.78rem; margin-top:1.5rem; }
.herosvg svg { width:100%; height:auto; display:block; filter:drop-shadow(0 18px 30px rgba(106,64,245,.25)); }
/* hero photo: no frame, edges fade into the page background */
.himg { overflow:hidden; background:transparent; border:none; box-shadow:none;
  -webkit-mask-image: linear-gradient(to right, transparent 0%, #000 18%, #000 82%, transparent 100%),
                      linear-gradient(to bottom, transparent 0%, #000 16%, #000 84%, transparent 100%);
  -webkit-mask-composite: source-in;
          mask-image: linear-gradient(to right, transparent 0%, #000 18%, #000 82%, transparent 100%),
                      linear-gradient(to bottom, transparent 0%, #000 16%, #000 84%, transparent 100%);
          mask-composite: intersect;
  -webkit-mask-size:100% 100%; -webkit-mask-repeat:no-repeat; mask-size:100% 100%; mask-repeat:no-repeat; }
.himg img { width:100%; height:100%; object-fit:cover; display:block; }
/* ---------- Home: whole card is the click target ---------- */
div[class*="st-key-card_home_"] { position:relative; cursor:pointer; transition:transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
div[class*="st-key-card_home_"]:hover { transform:translateY(-3px); box-shadow:0 16px 36px rgba(106,64,245,.2) !important; border-color:#8b6cf7 !important; }
div[class*="st-key-homebtn_"] { position:absolute !important; inset:0; width:100% !important; height:100% !important; margin:0 !important; z-index:10; }
div[class*="st-key-homebtn_"] [data-testid="stButton"], div[class*="st-key-homebtn_"] .stButton { width:100% !important; height:100% !important; }
div[class*="st-key-homebtn_"] button { width:100% !important; height:100% !important; min-height:100% !important; opacity:0; cursor:pointer; outline:none !important; box-shadow:none !important; }
p.lead.hero-lead { max-width:28rem; line-height:1.5; font-size:1.3rem; }
div[class*="st-key-hero_home"] .badge { font-size:.95rem; padding:.4rem .9rem; }
div[class*="st-key-hero_try"] button { min-height:3.4rem !important; padding:0 2.4rem !important; font-size:1.15rem !important; border-radius:16px !important; }
div[data-testid="stElementContainer"]:has(style) { display:none !important; }
/* cards named card_eq_* stretch to the height of their neighbour in the same row */
div[data-testid="stHorizontalBlock"]:has([class*="st-key-card_eq"]) { align-items:stretch !important; }
div[data-testid="stHorizontalBlock"]:has([class*="st-key-card_eq"]) > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] { height:100%; }
div[data-testid="stHorizontalBlock"]:has([class*="st-key-card_eq"]) > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > *:has([class*="st-key-card_eq"]),
div[data-testid="stHorizontalBlock"]:has([class*="st-key-card_eq"]) > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > [class*="st-key-card_eq"] { flex:1 1 auto; display:flex; flex-direction:column; }
div[data-testid="stHorizontalBlock"]:has([class*="st-key-card_eq"]) [class*="st-key-card_eq"] { flex:1 1 auto; }
.mini-metrics { display:flex; gap:1.2rem; flex-wrap:wrap; margin:.2rem 0 .5rem; }
.mini-metrics div { display:flex; flex-direction:column; }
.mini-metrics span { font-size:.8rem; color:#6b6a8a; font-weight:600; }
.mini-metrics b { font-size:1.45rem; font-weight:800; color:#1c1b3a; white-space:nowrap; }
.page-head { text-align:center; margin:1.4rem auto 2.2rem; }
.page-head h1.page { font-size:2.2rem; margin:.1rem 0 .3rem; }
.page-head p { margin:.5rem auto 0; max-width:48rem; color:#4b4a6b; font-size:.95rem; }
h1 a, h2 a, h3 a, [data-testid="stHeaderActionElements"] { display:none !important; }
.herobanner svg { width:100%; height:210px; display:block; border-radius:24px; filter:drop-shadow(0 14px 26px rgba(106,64,245,.22)); }
</style>
"""

# --- banner start ----------------------------------------------------------
# Hero banner = one self-contained rounded card. The photo lives *inside* the card (always as
# tall as the card, pinned to its right edge) so it scales together with the card when the
# window is resized. The photo's left edge fades into the card's own sky colour behind the text.
BANNER_CSS = """
<style>
div[class*="st-key-__KEY__"] { position:relative; min-height:__MINH__; padding:__PAD__; }
div[class*="st-key-__KEY__"]::before { content:""; position:absolute; inset:0; z-index:0; pointer-events:none;
  border-radius:26px; box-shadow:0 14px 40px rgba(106,64,245,.16), inset 0 0 0 1px rgba(255,255,255,.7);
  background: linear-gradient(90deg, __WASH__ 0%, __WASH2__ 24%, rgba(255,255,255,0) 52%) , url("data:image/webp;base64,__B64__") right center / auto 100% no-repeat, __BASE__; }
div[class*="st-key-__KEY__"] > * { position:relative; z-index:1; }
__EXTRA__
</style>
"""

BANNER_BASE = {
    "hero_home": "linear-gradient(90deg,#c4b9f7 0%,#d3c4f7 35%,#e8c8e4 70%,#f0c6c8 100%)",
    "hero_predict": "linear-gradient(90deg,#dcd9fc 0%,#d6defc 40%,#bcd3fb 100%)",
}


WASH = {
    "hero_home": ("rgba(205,195,248,.72)", "rgba(205,195,248,.42)"),
    "hero_predict": ("rgba(228,229,253,.72)", "rgba(228,229,253,.4)"),
}


def banner_css(key: str, b64: str, min_h: str, pos: str, pad: str, extra: str = "") -> str:
    base = BANNER_BASE.get(key, BANNER_BASE["hero_home"])
    return (
        BANNER_CSS.replace("__KEY__", key).replace("__MINH__", min_h).replace("__PAD__", pad)
        .replace("__B64__", b64).replace("__BASE__", base).replace("__WASH__", WASH[key][0]).replace("__WASH2__", WASH[key][1]).replace("__EXTRA__", extra)
    )


# page background: plain soft sky (no photo), so nothing depends on the window size
PAGE_SKY = (
    "radial-gradient(ellipse 60vw 24vw at 100% 0%, rgba(180,170,250,.45), rgba(180,170,250,0) 70%),"
    "radial-gradient(ellipse 50vw 20vw at 0% 100%, rgba(255,255,255,.8), rgba(255,255,255,0) 70%),"
    "linear-gradient(180deg,#d9d2fb 0%,#e8e2fd 30%,#f1eeff 60%,#f3f0ff 100%)"
)


def page_bg_css(page: str) -> str:
    return "<style>\n.stApp { background: " + PAGE_SKY + " !important; }\n</style>"
# --- banner end ------------------------------------------------------------

LOGO_SVG = """
<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
<path d="M58 8 L6 29 L24 35 L31 56 L37 41 L50 50Z" fill="#4f46e5"/>
<path d="M24 35 L58 8 L31 41Z" fill="#6a40f5"/>
</svg>"""

GITHUB_SVG = """
<svg viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
<path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.38 7.86 10.9.58.1.79-.25.79-.56v-2c-3.2.7-3.87-1.37-3.87-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.04-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.69 5.39-5.25 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z"/>
</svg>"""

HERO_SVG = """
<svg viewBox="0 0 560 360" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#7f9cff"/><stop offset=".55" stop-color="#b9a6ff"/><stop offset="1" stop-color="#ffc7b0"/>
</linearGradient>
<radialGradient id="sun" cx=".5" cy=".5" r=".5">
<stop offset="0" stop-color="#fff3c4" stop-opacity=".95"/><stop offset="1" stop-color="#fff3c4" stop-opacity="0"/>
</radialGradient>
</defs>
<rect width="560" height="360" rx="30" fill="url(#sky)"/>
<circle cx="440" cy="270" r="170" fill="url(#sun)"/>
<g fill="#ffffff" opacity=".9">
<ellipse cx="90" cy="318" rx="140" ry="36"/><ellipse cx="210" cy="332" rx="150" ry="34"/>
<ellipse cx="380" cy="326" rx="150" ry="36"/><ellipse cx="510" cy="312" rx="110" ry="34"/>
</g>
<g fill="#ffffff" opacity=".55">
<ellipse cx="70" cy="80" rx="60" ry="15"/><ellipse cx="110" cy="68" rx="40" ry="12"/>
<ellipse cx="470" cy="62" rx="55" ry="14"/><ellipse cx="505" cy="50" rx="35" ry="10"/>
</g>
<g transform="translate(290 165) rotate(-13) scale(1.05)">
<path d="M-12 -12 L-92 -98 L-60 -98 L56 -14Z" fill="#dfe3ff"/>
<path d="M-165 -8 L-206 -66 L-168 -66 L-108 -14Z" fill="#6a40f5"/>
<path d="M-150 4 L-196 40 L-166 40 L-100 10Z" fill="#dfe3ff"/>
<path d="M-172 0 C-172 -15 -122 -23 -40 -23 L120 -23 C172 -23 202 -8 208 0 C202 8 172 23 120 23 L-40 23 C-122 23 -172 15 -172 0Z" fill="#ffffff"/>
<path d="M150 -15 C172 -15 192 -7 200 0 L150 0Z" fill="#7f9cff"/>
<path d="M-172 6 L208 6 L208 0 C202 8 172 23 120 23 L-40 23 C-122 23 -172 15 -172 6Z" fill="#e8ebff"/>
<path d="M-12 14 L-104 104 L-64 104 L62 16Z" fill="#ffffff" stroke="#cfd4ff" stroke-width="1.5"/>
<ellipse cx="-34" cy="52" rx="28" ry="12" fill="#eceeff" stroke="#cfd4ff" stroke-width="1.5"/>
<g fill="#7f9cff"><circle cx="-70" cy="-8" r="4"/><circle cx="-48" cy="-8" r="4"/><circle cx="-26" cy="-8" r="4"/><circle cx="-4" cy="-8" r="4"/><circle cx="18" cy="-8" r="4"/><circle cx="40" cy="-8" r="4"/><circle cx="62" cy="-8" r="4"/><circle cx="84" cy="-8" r="4"/><circle cx="106" cy="-8" r="4"/></g>
</g>
</svg>"""

# Wide, short crop of the hero illustration (keeps the plane centred)
HERO_BANNER_SVG = HERO_SVG.replace("<svg viewBox", '<svg preserveAspectRatio="xMidYMid slice" viewBox', 1)


def html(markup: str) -> None:
    """Render raw HTML. Leading spaces are stripped so Markdown never treats it as code."""
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


ASSETS_DIR = Path(__file__).resolve().parent / "assets"


@st.cache_data(show_spinner=False)
def _image_b64(name: str) -> str:
    path = ASSETS_DIR / name
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def hero_img(name: str, height: int, fallback: str = "", position: str = "center") -> str:
    """Photo from assets/ as a rounded card. Uses `fallback` HTML if the file is missing."""
    b64 = _image_b64(name)
    if not b64:
        return fallback
    return (
        f'<div class="himg" style="height:{height}px">'
        f'<img src="data:image/jpeg;base64,{b64}" style="object-position:{position}" alt=""></div>'
    )


def show_plot(fig: go.Figure, key: str | None = None) -> None:
    """st.plotly_chart that works on old and new Streamlit versions."""
    params = inspect.signature(st.plotly_chart).parameters
    kwargs = {"width": "stretch"} if "width" in params else {"use_container_width": True}
    st.plotly_chart(fig, key=key, config={"displayModeBar": False}, **kwargs)


def style_fig(fig: go.Figure, height: int = 300, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#4a4870", size=12),
        showlegend=legend,
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor="#d9d4f5", zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def inr(x: float) -> str:
    return f"₹{x:,.0f}"


def html_table(df: pd.DataFrame, best_row: int | None = None) -> str:
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for i, row in enumerate(df.itertuples(index=False)):
        cls = ' class="best"' if best_row is not None and i == best_row else ""
        body += f"<tr{cls}>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


_card_ids = itertools.count()


def card(name: str | None = None):
    """Bordered container with a class name we can style (st-key-card_*). Falls back on old Streamlit."""
    key = f"card_{name or next(_card_ids)}"
    try:
        return st.container(border=True, key=key)
    except TypeError:  # Streamlit < 1.39 has no `key` for containers
        return st.container(border=True)


def make_report(last: dict, history: list, mt: dict) -> tuple[bytes, str, str]:
    """PDF report of the latest prediction (+ session history). Falls back to Excel if reportlab is missing."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        buf = io.BytesIO()
        pd.DataFrame(history).to_excel(buf, index=False)
        return buf.getvalue(), "flight_fare_predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    from datetime import datetime

    purple, soft, ink = colors.HexColor("#6a40f5"), colors.HexColor("#efeaff"), colors.HexColor("#1c1b3a")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=ink, fontSize=22, leading=28, alignment=0, spaceAfter=6)
    sub_ = ParagraphStyle("sub", parent=ss["Normal"], textColor=colors.HexColor("#6b6a8a"), fontSize=9, leading=12)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=purple, fontSize=13, spaceBefore=10, spaceAfter=4)
    small = ParagraphStyle("small", parent=ss["Normal"], textColor=colors.HexColor("#6b6a8a"), fontSize=8, leading=11)

    def rs(x: float) -> str:  # built-in PDF fonts have no rupee glyph
        return f"Rs. {x:,.0f}"

    def grid(rows, widths, header=False):
        t = Table(rows, colWidths=widths)
        style = [
            ("FONTSIZE", (0, 0), (-1, -1), 9.5), ("TEXTCOLOR", (0, 0), (-1, -1), ink),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f7f5ff")]),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#e4defc")),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), purple), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                      ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        else:
            style += [("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6b6a8a"))]
        t.setStyle(TableStyle(style))
        return t

    route = f"{fu.city(last['source'])} to {fu.city(last['destination'])}"
    low, high = max(last["fare"] - mt["mae"], 0), last["fare"] + mt["mae"]
    vs = (last["fare"] - last["route_avg"]) / last["route_avg"] * 100
    story = [
        Paragraph("Flight Fare Prediction Report", h1),
        Paragraph(f"Generated on {datetime.now():%d %b %Y, %H:%M} &nbsp;|&nbsp; {APP_TITLE}", sub_),
        Spacer(1, 8),
    ]
    fare_box = Table(
        [[Paragraph("Predicted Fare", ParagraphStyle("f1", parent=ss["Normal"], textColor=colors.HexColor("#6b6a8a"), fontSize=10, leading=14, alignment=1))],
         [Paragraph(f"<b>{rs(last['fare'])}</b>", ParagraphStyle("f2", parent=ss["Normal"], textColor=ink, fontSize=30, leading=38, alignment=1))],
         [Paragraph(f"{route} | {last['airline']}", ParagraphStyle("f3", parent=ss["Normal"], textColor=colors.HexColor("#6b6a8a"), fontSize=9, leading=13, alignment=1))]],
        colWidths=[174 * mm],
    )
    fare_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), soft), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                  ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [fare_box, Paragraph("Flight Details", h2)]
    story.append(grid(
        [["Route", route], ["Airline", last["airline"]], ["Travel date", f"{last['date']:%d %b %Y}"],
         ["Departure time", last["dep"]], ["Arrival time", last["arrival"]],
         ["Duration", fu.hhmm(last["duration"])], ["Total stops", fu.STOP_LABELS[last["stops"]]],
         ["Additional info", last.get("info", "No info")]],
        [50 * mm, 124 * mm],
    ))
    story.append(Paragraph("Fare Details", h2))
    story.append(grid(
        [["Predicted fare", rs(last["fare"])], ["Likely range (+/- MAE)", f"{rs(low)} - {rs(high)}"],
         ["Route average fare", rs(last["route_avg"])], ["Compared with route average", f"{vs:+.0f}%"]],
        [50 * mm, 124 * mm],
    ))
    story.append(Paragraph("Model", h2))
    story.append(grid(
        [["Model", "Tuned Random Forest"], ["MAE", f"{mt['mae']:,.2f}"], ["RMSE", f"{mt['rmse']:,.2f}"], ["R2 score", f"{mt['r2']:.4f}"]],
        [50 * mm, 124 * mm],
    ))
    if len(history) > 1:
        story.append(Paragraph("Predictions in this session", h2))
        rows = [["Date", "Route", "Airline", "Stops", "Duration", "Fare"]]
        for h in history[-12:][::-1]:
            rows.append([h["Date"], h["Route"].replace("→", "to"), h["Airline"], h["Stops"], h["Duration"], rs(h["Predicted Fare (₹)"])])
        story.append(grid(rows, [24 * mm, 48 * mm, 30 * mm, 20 * mm, 22 * mm, 30 * mm], header=True))
    story += [Spacer(1, 10), Paragraph(
        "Predictions are estimates from a Random Forest model trained on historical Indian domestic flight data. "
        "Actual fares depend on airline pricing and availability.", small)]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                      title="Flight Fare Prediction Report").build(story)
    return buf.getvalue(), f"flight_fare_report_{last['source']}_{last['destination']}.pdf".replace(" ", "_"), "application/pdf"


def go_to(page: str) -> None:
    """Button callback used for navigation."""
    st.session_state["page"] = page


# --------------------------------------------------------------------------- #
# Data + model (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading dataset...")
def load_data(source) -> pd.DataFrame:
    """`source` is a file path (str) or the raw bytes of an uploaded Excel file."""
    raw = fu.read_raw(io.BytesIO(source) if isinstance(source, bytes) else source)
    return fu.clean_data(raw)


@st.cache_resource(show_spinner="Preparing the model...")
def load_bundle(data_key: str, _df: pd.DataFrame) -> dict:
    return fu.prepare_bundle(_df)


def get_dataset() -> pd.DataFrame:
    path = fu.find_dataset()
    if path is not None:
        return load_data(str(path))
    html(
        """
        <h1 class="page">Dataset <span class="grad">not found</span></h1>
        <p class="lead">Place <b>Flight_Fare.xlsx</b> (the dataset used in your notebook) next to
        <b>app.py</b>, or upload it below.</p>
        """
    )
    up = st.file_uploader("Upload Flight_Fare.xlsx", type=["xlsx"])
    if up is None:
        st.stop()
    return load_data(up.getvalue())


# --------------------------------------------------------------------------- #
# Shared page pieces
# --------------------------------------------------------------------------- #
def _supports_key() -> bool:
    """st.container(key=...) exists from Streamlit 1.39."""
    return "key" in inspect.signature(st.container).parameters


def _supports_va() -> bool:
    return "vertical_alignment" in inspect.signature(st.columns).parameters


def render_nav() -> None:
    """Flat top bar: brand on the left, page links on the right (no box)."""
    weights = [{"Home": 1.0, "Quick Prediction": 1.8, "Smart Analytics": 1.7, "Plan Better": 1.4, "About": 1.0}[p] for p in PAGES]
    sizes = [6.5] + weights + ([0.5] if GITHUB_URL else [])
    holder = st.container(key="nav_bar") if _supports_key() else st.container()
    with holder:
        cols = st.columns(sizes, vertical_alignment="center") if _supports_va() else st.columns(sizes)
        with cols[0]:
            html(f'<div class="brand"><span class="nav-mark"></span>{LOGO_SVG}<span>{APP_TITLE}</span></div>')
        for col, page in zip(cols[1 : 1 + len(PAGES)], PAGES):
            with col:
                st.button(
                    page, key=f"nav_{page}", on_click=go_to, args=(page,),
                    type="primary" if st.session_state.get("page") == page else "secondary",
                )
        if GITHUB_URL:
            with cols[-1]:
                html(f'<div class="gh"><a href="{GITHUB_URL}" target="_blank">{GITHUB_SVG}</a></div>')


def footer(df: pd.DataFrame) -> None:
    year = int(df["Date_of_Journey"].dt.year.min())
    months = sorted(df["Journey_Month"].unique())
    span = f"{calendar.month_name[months[0]]}–{calendar.month_name[months[-1]]} {year}"
    html(
        f"""<div class="note">Predictions are estimates from a Random Forest model trained on {len(df):,} flights
        (journeys in {span}). Actual fares depend on airline pricing and availability.</div>"""
    )


def default_travel_date(df: pd.DataFrame) -> date:
    """Next 15 May (inside the months the model was trained on)."""
    today = date.today()
    d = date(today.year, 5, 15)
    return d if d >= today else date(today.year + 1, 5, 15)


def date_warning(df: pd.DataFrame, d: date) -> None:
    months = sorted(df["Journey_Month"].unique())
    if d.month not in months:
        names = ", ".join(calendar.month_abbr[m] for m in months)
        st.info(
            f"The model was trained on journeys in {names}. For {calendar.month_name[d.month]} "
            "it can only approximate the fare from the closest months it has seen."
        )


# --------------------------------------------------------------------------- #
# Page: Home
# --------------------------------------------------------------------------- #
def _hero_copy() -> None:
    html(
        """
        <div class="badge">⚡ ML Powered ⭐</div>
        <h1 class="hero">Flight Fare<br><span class="grad">Prediction</span></h1>
        <p class="lead hero-lead">Get an accurate estimate of flight fares using Machine Learning technology.</p>
        """
    )
    b, _ = st.columns([1.1, 1.4])
    with b:
        st.button("Try Now  →", type="primary", key="hero_try", on_click=go_to, args=("Quick Prediction",))


def page_home(df: pd.DataFrame, bundle: dict) -> None:
    keyed = _supports_key()
    b64 = _image_b64("home_sky.webp")

    if keyed and b64:
        # Full-width banner: the photo covers the strip, lavender wash on the left carries the text.
        st.markdown(banner_css("hero_home", b64, "clamp(360px, 31vw, 420px)", "right", "2rem 2.4rem 1.8rem"), unsafe_allow_html=True)
        with st.container(key="hero_home"):
            left, _ = st.columns([1.5, 1], gap="large")
            with left:
                _hero_copy()
    else:
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            _hero_copy()
        with right:
            html(hero_img("home_hero.jpg", 330, f'<div class="herosvg">{HERO_SVG}</div>', "55% 50%"))

    st.write("")
    cards = [
        ("⚡", "Quick Prediction", "Get instant fare estimates.", "Quick Prediction", "quick"),
        ("📊", "Smart Analytics", "Explore insights from the data.", "Smart Analytics", "smart"),
        ("🧭", "Plan Better", "Compare airlines and dates.", "Plan Better", "plan"),
    ]
    cols = st.columns(3, gap="medium")
    for col, (icon, title, desc, target, slug) in zip(cols, cards):
        with col, card(f"home_{slug}"):
            html(
                f'<div style="text-align:center;padding:.4rem 0 .3rem"><div class="ficon">{icon}</div>'
                f'<div class="ftitle">{title}</div><div class="small">{desc}</div></div>'
            )
            if keyed:
                # invisible button stretched over the whole card (see CSS: st-key-homebtn_*)
                st.button(f"Open {title}", key=f"homebtn_{slug}", on_click=go_to, args=(target,))
            else:  # very old Streamlit: fall back to a visible button
                _, mid, _ = st.columns([1, 1.6, 1])
                with mid:
                    st.button("Open  →", key=f"open_{slug}", on_click=go_to, args=(target,))

    st.write("")
    top_route = df.groupby(["Source", "Destination"]).size().sort_values(ascending=False).index[0]
    stats = [
        ("📋", "Flights Analysed", f"{len(df):,}"),
        ("📍", "Popular Route", f"{fu.city(top_route[0])} → {fu.city(top_route[1])}"),
        ("⭐", "Model R² Score", f"{bundle['metrics']['r2']:.4f}"),
    ]
    cols = st.columns(3, gap="medium")
    for col, (icon, label, value) in zip(cols, stats):
        with col:
            html(f'<div class="stat"><div class="si">{icon}</div><div class="sl">{label}</div><div class="sv">{value}</div></div>')

    footer(df)


# --------------------------------------------------------------------------- #
# Page: Quick Prediction
# --------------------------------------------------------------------------- #
def page_predict(df: pd.DataFrame, bundle: dict) -> None:
    model, fcols, mae = bundle["model"], bundle["feature_columns"], bundle["metrics"]["mae"]
    mt_ = bundle["metrics"]
    rmap = fu.route_map(df)

    def _predict_copy() -> None:
        html(
            """
            <h1 class="page">Predict Your<br><span class="grad">Flight Fare</span></h1>
            <p class="lead hero-lead">Enter flight details and get an estimated fare using Machine Learning!</p>
            """
        )

    pb64 = _image_b64("predict_sky.webp")
    if _supports_key() and pb64:
        st.markdown(
            banner_css("hero_predict", pb64, "clamp(210px, 20vw, 270px)", "right", "1.6rem 2.2rem 1.4rem"),
            unsafe_allow_html=True,
        )
        with st.container(key="hero_predict"):
            left, _ = st.columns([1.3, 1], gap="large")
            with left:
                _predict_copy()
    else:
        h1, h2 = st.columns([1.3, 1], gap="large")
        with h1:
            _predict_copy()
        with h2:
            html(hero_img("predict_hero.jpg", 215, f'<div class="herobanner">{HERO_BANNER_SVG}</div>', "60% 50%"))

    main, side = st.columns([2.2, 1], gap="medium")

    # ----------------------------- inputs ---------------------------------- #
    with main:
        with card("eq_details"):
            html('<div class="card-title">✈️ Flight Details</div>')
            c1, c2, c3, c4 = st.columns(4)
            source = c1.selectbox("Source City", sorted(rmap), format_func=fu.city)
            destination = c2.selectbox("Destination City", rmap[source], format_func=fu.city)
            sub = fu.route_subset(df, source, destination)
            airline = c3.selectbox("Airline", sub["Airline"].value_counts().index.tolist())
            travel_date = c4.date_input("Travel Date", value=default_travel_date(df), format="YYYY-MM-DD")

            d1, d2, d3, d4 = st.columns(4)
            dep_time = d1.time_input("Departure Time", value=time(10, 30), step=300)
            dur_h = d2.number_input("Duration (hours)", min_value=0, max_value=47, value=2, step=1)
            dur_m = d3.number_input("Duration (minutes)", min_value=0, max_value=55, value=50, step=5)
            stop_vals = sorted(sub["Total_Stops"].map(fu.STOP_MAPPING).unique().tolist())
            common_stop = int(sub["Total_Stops"].map(fu.STOP_MAPPING).mode().iloc[0])
            total_stops = d4.selectbox(
                "Total Stops",
                stop_vals,
                index=stop_vals.index(common_stop),
                format_func=lambda s: fu.STOP_LABELS[s].capitalize(),
            )

            e1, e2 = st.columns([1.5, 2.5])
            infos = sub["Additional_Info"].value_counts().index.tolist()
            info = e1.selectbox("Additional Info", infos, index=infos.index("No info") if "No info" in infos else 0)

            duration_minutes = int(dur_h) * 60 + int(dur_m)
            arr_h, arr_m, plus_days = fu.compute_arrival(dep_time.hour, dep_time.minute, duration_minutes)
            arrival_txt = f"{arr_h:02d}:{arr_m:02d}" + (f" (+{plus_days} day)" if plus_days else "")
            prof = fu.route_profile(df, source, destination, airline)
            with e2:
                html(
                    f"""
                    <div class="small" style="padding-top:1.7rem">
                    🛬 Estimated arrival <b>{arrival_txt}</b> &nbsp;·&nbsp;
                    Typical for this route: <b>{fu.hhmm(prof['duration_minutes'])}</b>,
                    <b>{fu.STOP_LABELS[prof['total_stops']]}</b>
                    </div>
                    """
                )
            st.caption("Cities, airlines and options shown are the combinations that exist in the training data.")
            date_warning(df, travel_date)

            clicked = st.button("✈️  Predict Flight Fare", type="primary", key="predict_btn")

        if clicked:
            if duration_minutes == 0:
                st.warning("Please enter a flight duration greater than zero.")
            else:
                row = fu.make_row(
                    airline, source, destination, travel_date,
                    dep_time.hour, dep_time.minute, duration_minutes, total_stops, info,
                )
                fare = float(fu.predict_many(model, fcols, [row])[0])
                result = dict(
                    row=row, fare=fare, date=travel_date, source=source, destination=destination,
                    airline=airline, duration=duration_minutes, stops=total_stops,
                    dep=f"{dep_time.hour:02d}:{dep_time.minute:02d}", arrival=arrival_txt,
                    route_avg=float(sub["Price"].mean()), info=info,
                )
                st.session_state["last"] = result
                st.session_state.setdefault("history", []).append(
                    {
                        "Date": travel_date.isoformat(),
                        "Route": f"{fu.city(source)} → {fu.city(destination)}",
                        "Airline": airline,
                        "Stops": fu.STOP_LABELS[total_stops],
                        "Duration": fu.hhmm(duration_minutes),
                        "Predicted Fare (₹)": int(round(fare)),
                    }
                )

    # ----------------------------- result ---------------------------------- #
    with side:
        last = st.session_state.get("last")
        with card("eq_result"):
            html('<div style="text-align:center;font-size:1.7rem;margin-top:.4rem">🎫</div>')
            if not last:
                html(
                    """<div class="fare-box"><div class="fl">Predicted Fare</div>
                    <div class="fv">₹ —</div>
                    <div class="small">Fill in the flight details and press <b>Predict Flight Fare</b>.</div></div>"""
                )
            else:
                low, high = max(last["fare"] - mae, 0), last["fare"] + mae
                vs = (last["fare"] - last["route_avg"]) / last["route_avg"] * 100
                route = f"{fu.city(last['source'])} → {fu.city(last['destination'])}"
                html(
                    f"""
                    <div class="fare-box"><div class="fl">Predicted Fare</div>
                    <div class="fv">{inr(last['fare'])}</div>
                    <div class="small">{route} · {last['airline']}</div></div>
                    <div class="fare-sub">Fare Details</div>
                    <div class="kv-row"><span>Likely range (± MAE)</span><b>{inr(low)} – {inr(high)}</b></div>
                    <div class="kv-row"><span>Route average</span><b>{inr(last['route_avg'])}</b></div>
                    <div class="kv-row"><span>vs route average</span><b>{vs:+.0f}%</b></div>
                    <div class="kv-row"><span>Departure → Arrival</span><b>{last['dep']} → {last['arrival']}</b></div>
                    <div class="kv-row total"><span>Duration / Stops</span><b>{fu.hhmm(last['duration'])} · {fu.STOP_LABELS[last['stops']]}</b></div>
                    """
                )
            history = st.session_state.get("history", [])
            report, rname, rmime = make_report(last, history, mt_) if last else (b"", "flight_fare_report.pdf", "application/pdf")
            _l, _m, _r = st.columns([0.35, 3, 0.35])
            with _m:
                _dl = inspect.signature(st.download_button).parameters
                _stretch = {"width": "stretch"} if "width" in _dl else ({"use_container_width": True} if "use_container_width" in _dl else {})
                st.download_button(
                    "↓  Download Report", data=report, file_name=rname,
                    mime=rmime, disabled=not last, key="dl_report", **_stretch,
                )

    # ------------------- bottom row: what-if | history | download ---------- #
    last = st.session_state.get("last")
    wi_col, hist_col = st.columns([1, 1.3], gap="medium")

    with wi_col, card("eq_whatif"):
        html('<div class="card-title">📈 What-if Analysis</div><div class="small" style="margin:-.3rem 0 .4rem">See how changes affect the fare</div>')
        if not last:
            st.caption("Run a prediction to see how the fare changes with the travel date.")
        else:
            shift = st.slider("Change Travel Date (days):", -7, 14, 0, key="whatif")
            shifts = list(range(-7, 15))
            rows = []
            for s in shifts:
                r = dict(last["row"])
                d = last["date"] + timedelta(days=s)
                r["journey_day"], r["journey_month"] = d.day, d.month
                rows.append(r)
            preds = fu.predict_many(model, fcols, rows)
            cur = float(preds[shifts.index(shift)])
            new_date = last["date"] + timedelta(days=shift)
            delta = cur - last["fare"]
            html(
                f"""<div class="small">Fare on <b>{new_date:%d %b %Y}</b>
                ({'+' if delta >= 0 else '−'}{inr(abs(delta))} vs selected date)</div>
                <div class="newfare"><div class="small">New Estimated Fare:</div><div class="nfv">{inr(cur)}</div></div>"""
            )
            if st.checkbox("Show fare trend", key="show_trend"):
                fig = go.Figure(
                    go.Scatter(
                        x=[f"{(last['date'] + timedelta(days=s)):%d %b}" for s in shifts],
                        y=preds, mode="lines+markers",
                        line=dict(color=PURPLE, width=2), marker=dict(size=6, color=PURPLE),
                        hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[f"{new_date:%d %b}"], y=[cur], mode="markers",
                        marker=dict(size=12, color="#ffffff", line=dict(color=PURPLE, width=3)),
                        hoverinfo="skip",
                    )
                )
                style_fig(fig, height=170)
                fig.update_yaxes(tickprefix="₹", tickformat=",d")
                fig.update_xaxes(nticks=6)
                show_plot(fig, key="whatif_chart")
            st.caption("Only day and month are used by the model (not the year).")


    with hist_col, card("eq_history"):
        html('<div class="card-title">🕘 Prediction History</div><div class="small" style="margin:-.3rem 0 .4rem">Your recent predictions</div>')
        history = st.session_state.get("history", [])
        if not history:
            st.caption("Your predictions in this session will appear here.")
        else:
            recent = pd.DataFrame(history[-5:][::-1])[["Date", "Route", "Airline", "Predicted Fare (₹)"]]
            recent["Predicted Fare (₹)"] = recent["Predicted Fare (₹)"].map(lambda v: f"₹{v:,}")
            recent = recent.rename(columns={"Predicted Fare (₹)": "Fare"})
            html(html_table(recent))



    footer(df)


# --------------------------------------------------------------------------- #
# Page: Smart Analytics
# --------------------------------------------------------------------------- #
def kpi(icon: str, label: str, value: str) -> str:
    return f'<div class="kpi"><div class="ki">{icon}</div><div class="kl">{label}</div><div class="kv">{value}</div></div>'


def page_analytics(df: pd.DataFrame, bundle: dict) -> None:
    html(
        """
        <div class="page-head">
        <h1 class="page"><span class="grad">Flight Insights</span> &amp; Trends</h1>
        <p>Explore useful insights from the flight data.</p></div>
        """
    )

    f1, f2, f3 = st.columns(3)
    airline = f1.selectbox("Airline", ["All Airlines"] + sorted(df["Airline"].unique()))
    source = f2.selectbox("Source City", ["All Cities"] + sorted(df["Source"].unique()), format_func=fu.city)
    destination = f3.selectbox("Destination City", ["All Cities"] + sorted(df["Destination"].unique()), format_func=fu.city)

    d = df
    if airline != "All Airlines":
        d = d[d["Airline"] == airline]
    if source != "All Cities":
        d = d[d["Source"] == source]
    if destination != "All Cities":
        d = d[d["Destination"] == destination]
    if d.empty:
        st.warning("No flights match these filters. Try a different combination.")
        return

    k = st.columns(4, gap="medium")
    pop_air = d["Airline"].value_counts().index[0]
    for col, (icon, label, val) in zip(
        k,
        [
            ("✈️", "Average Fare", inr(d["Price"].mean())),
            ("🏆", "Highest Fare", inr(d["Price"].max())),
            ("⬇️", "Lowest Fare", inr(d["Price"].min())),
            ("⭐", "Popular Airline", pop_air),
        ],
    ):
        with col:
            html(kpi(icon, label, val))
    st.write("")

    # ---- row 1: trend + airline share ---- #
    c1, c2 = st.columns([1.45, 1], gap="medium")
    with c1, card():
        html('<div class="card-title">📈 Average Fare by Journey Month</div>')
        m = d.groupby("Journey_Month")["Price"].mean().sort_index()
        fig = go.Figure(
            go.Scatter(
                x=[calendar.month_abbr[i] for i in m.index], y=m.values, mode="lines+markers",
                line=dict(color=PURPLE, width=3), marker=dict(size=9, color=PURPLE, line=dict(color="#fff", width=2)),
                hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
            )
        )
        style_fig(fig, height=270)
        fig.update_yaxes(tickprefix="₹", tickformat=",d")
        show_plot(fig, key="trend")
    with c2, card():
        html('<div class="card-title">🍩 Airline Share of Flights</div>')
        counts = d["Airline"].value_counts()
        top = counts.head(5)
        if len(counts) > 5:
            top = pd.concat([top, pd.Series({"Others": counts.iloc[5:].sum()})])
        fig = go.Figure(
            go.Pie(
                labels=top.index.tolist(), values=top.values, hole=0.58, sort=False,
                marker=dict(colors=DONUT_COLORS[: len(top)], line=dict(color="#ffffff", width=2)),
                textinfo="percent", hovertemplate="%{label}: %{value:,} flights (%{percent})<extra></extra>",
            )
        )
        style_fig(fig, height=270, legend=True)
        fig.update_layout(legend=dict(orientation="v", x=1.0, y=0.5))
        show_plot(fig, key="donut")

    # ---- row 2: distribution, airline avg fare, routes ---- #
    c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
    with c1, card():
        html('<div class="card-title">📊 Fare Distribution</div>')
        fig = go.Figure(
            go.Histogram(x=d["Price"], nbinsx=40, marker=dict(color=PURPLE, line=dict(color="#fff", width=1)),
                         hovertemplate="₹%{x}: %{y} flights<extra></extra>")
        )
        style_fig(fig, height=250)
        fig.update_xaxes(tickprefix="₹", tickformat=",d")
        show_plot(fig, key="hist")
    with c2, card():
        html('<div class="card-title">💸 Average Fare by Airline</div>')
        a = d.groupby("Airline")["Price"].mean().sort_values().tail(8)
        fig = go.Figure(
            go.Bar(x=a.values, y=a.index.tolist(), orientation="h", marker=dict(color=PURPLE),
                   hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>")
        )
        style_fig(fig, height=250)
        fig.update_xaxes(tickprefix="₹", tickformat=",d", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        show_plot(fig, key="airline_bar")
    with c3, card():
        html('<div class="card-title">🧭 Popular Routes</div>')
        r = (
            d.groupby(["Source", "Destination"])["Price"]
            .agg(["count", "mean"]).sort_values("count", ascending=False).head(6).reset_index()
        )
        table = pd.DataFrame(
            {
                "Route": [f"{fu.city(s)} → {fu.city(t)}" for s, t in zip(r["Source"], r["Destination"])],
                "Flights": r["count"].map("{:,}".format),
                "Avg Fare": r["mean"].map(inr),
            }
        )
        html(html_table(table))

    # ---- row 3: model ---- #
    st.write("")
    html('<div class="card-title" style="font-size:1.35rem">🤖 Model Performance</div>')
    mt = bundle["metrics"]
    c1, c2 = st.columns([1, 1.25], gap="medium")
    with c1, card():
        html('<div class="card-title">Final model: Tuned Random Forest</div>')
        html(
            f"""<div class="mini-metrics">
            <div><span>MAE</span><b>{mt['mae']:,.2f}</b></div>
            <div><span>RMSE</span><b>{mt['rmse']:,.2f}</b></div>
            <div><span>R² Score</span><b>{mt['r2']:.4f}</b></div></div>"""
        )
        st.caption(f"{bundle['source']}. Metrics are on the 20% test split (random_state=42).")
        comp = fu.NOTEBOOK_MODEL_RESULTS.copy()
        best = int(comp["R2 Score"].idxmax())
        comp["MAE"] = comp["MAE"].map(lambda v: f"{v:,.2f}")
        comp["RMSE"] = comp["RMSE"].map(lambda v: f"{v:,.2f}")
        comp["R2 Score"] = comp["R2 Score"].map(lambda v: f"{v:.4f}")
        html('<div class="small" style="margin-top:.4rem"><b>Model comparison (from the notebook)</b></div>')
        html(html_table(comp, best_row=best))
    with c2, card():
        html('<div class="card-title">Top 10 Important Features</div>')
        imp = bundle["importance"].head(10).iloc[::-1]
        fig = go.Figure(
            go.Bar(x=imp["Importance"], y=[fu.pretty_feature(f) for f in imp["Feature"]], orientation="h",
                   marker=dict(color=PURPLE), hovertemplate="%{y}: %{x:.3f}<extra></extra>")
        )
        style_fig(fig, height=330)
        fig.update_xaxes(showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        show_plot(fig, key="importance")

    with card():
        html('<div class="card-title">Actual vs Predicted Price (test set)</div>')
        yt, yp = bundle["y_test"], bundle["y_pred"]
        lim = [0, float(max(yt.max(), yp.max()))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yt, y=yp, mode="markers", name="Flights",
                                 marker=dict(color=PURPLE, size=5, opacity=0.45),
                                 hovertemplate="Actual ₹%{x:,.0f}<br>Predicted ₹%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction",
                                 line=dict(color="#9a98b8", dash="dash", width=1.5), hoverinfo="skip"))
        style_fig(fig, height=330, legend=True)
        fig.update_xaxes(title="Actual price", tickprefix="₹", tickformat=",d")
        fig.update_yaxes(title="Predicted price", tickprefix="₹", tickformat=",d")
        fig.update_layout(legend=dict(orientation="h", y=1.1, x=0))
        show_plot(fig, key="avp")
    footer(df)


# --------------------------------------------------------------------------- #
# Page: Plan Better
# --------------------------------------------------------------------------- #
def page_plan(df: pd.DataFrame, bundle: dict) -> None:
    model, fcols = bundle["model"], bundle["feature_columns"]
    rmap = fu.route_map(df)

    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        html(
            """
            <div class="badge">💡 Travel Smart ⭐</div>
            <h1 class="page" style="font-size:3.9rem">Plan Better<br><span class="grad">with Data</span></h1>
            <p class="lead" style="font-size:1.35rem; max-width:none; line-height:1.55">Get useful travel tips and explore flight options<br>before planning your journey.</p>
            <div class="check check-big"><i>✓</i>Compare fares across airlines.</div>
            <div class="check check-big"><i>✓</i>Explore different travel dates.</div>
            <div class="check check-big"><i>✓</i>Stay within your budget.</div>
            <div class="check check-big"><i>✓</i>Review estimated flight prices.</div>
            """
        )
        st.write("")
        b, _ = st.columns([1.2, 1])
        with b:
            st.button("View Flight Prediction  →", type="primary", key="plan_to_predict", on_click=go_to, args=("Quick Prediction",))
    with right, card("eq_tips"):
        html('<div class="card-title">💡 Quick Tips <span class="small">(from the dataset)</span></div>')
        for icon, text in fu.quick_tips(df):
            html(f'<div class="tip"><div class="ti">{icon}</div><div>{text}</div></div>')

    st.write("")
    html(hero_img("plan_hero.jpg", 250, "", "50% 6%"))
    st.write("")
    with card():
        html('<div class="card-title">🧳 Travel Planner</div>')
        p1, p2, p3, p4 = st.columns(4)
        source = p1.selectbox("From", sorted(rmap), format_func=fu.city, key="plan_from")
        destination = p2.selectbox("To", rmap[source], format_func=fu.city, key="plan_to")
        travel_date = p3.date_input("Date", value=default_travel_date(df), format="YYYY-MM-DD", key="plan_date")
        budget = p4.number_input("Budget (₹)", min_value=1000, max_value=200000, value=10000, step=500, key="plan_budget")
        date_warning(df, travel_date)

    sub = fu.route_subset(df, source, destination)
    counts = sub["Airline"].value_counts()
    airlines = counts[counts >= 5].index.tolist() or counts.index.tolist()

    def rows_for(day: date) -> list[dict]:
        out = []
        for a in airlines:
            p = fu.route_profile(df, source, destination, a)
            out.append(
                fu.make_row(a, source, destination, day, p["dep_hour"], p["dep_minute"],
                            p["duration_minutes"], p["total_stops"], "No info")
            )
        return out

    # airline comparison for the chosen date
    fares = fu.predict_many(model, fcols, rows_for(travel_date))
    comp = pd.DataFrame({"Airline": airlines, "fare": fares}).sort_values("fare").reset_index(drop=True)

    # date comparison (selected date + next 4 days): average over airlines
    day_list = [travel_date + timedelta(days=i) for i in range(5)]
    day_avgs = [float(np.mean(fu.predict_many(model, fcols, rows_for(dd)))) for dd in day_list]
    cheapest_day = int(np.argmin(day_avgs))

    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1, card("eq_air"):
        html('<div class="card-title">🏷️ Airline Comparison</div>')
        shown = comp.head(8)
        table = pd.DataFrame(
            {
                "Airline": shown["Airline"],
                "Estimated Fare": shown["fare"].map(inr),
                "Status": [
                    ('<span class="pill">★ Best value</span> ' if i == 0 else "")
                    + ('<span class="pill">Within budget</span>' if f <= budget else '<span class="pill warn">Over budget</span>')
                    for i, f in enumerate(shown["fare"])
                ],
            }
        )
        html(html_table(table, best_row=0))
        prof = fu.route_profile(df, source, destination)
        st.caption(
            f"Estimates use the typical flight on {fu.city(source)} → {fu.city(destination)}: "
            f"about {fu.hhmm(prof['duration_minutes'])}, {fu.STOP_LABELS[prof['total_stops']]} (median of {prof['n']:,} flights)."
        )
    with c2, card("eq_date"):
        html('<div class="card-title">📆 Travel Date Comparison</div>')
        table = pd.DataFrame(
            {
                "Date": [f"{dd:%d %b %Y}" for dd in day_list],
                "Avg Fare": [inr(v) for v in day_avgs],
                "": ['<span class="pill">Cheapest</span>' if i == cheapest_day else "" for i in range(5)],
            }
        )
        html(html_table(table, best_row=cheapest_day))
        st.caption("Average predicted fare across the airlines on this route.")

    footer(df)


# --------------------------------------------------------------------------- #
# Page: About
# --------------------------------------------------------------------------- #
def page_about(df: pd.DataFrame, bundle: dict) -> None:
    mt = bundle["metrics"]
    html(
        f"""
        <div class="page-head">
        <h1 class="page">About this <span class="grad">Project</span></h1>
        <p>Predicting Indian domestic flight fares with Machine Learning, by <b>{DEVELOPER}</b>.</p></div>
        """
    )
    c1, c2 = st.columns(2, gap="medium")
    with c1, card("eq_ds"):
        html('<div class="card-title">🗂️ Dataset</div>')
        html(
            f"""
            <div class="kv-row"><span>Raw records</span><b>10,683</b></div>
            <div class="kv-row"><span>After cleaning (used here)</span><b>{len(df):,}</b></div>
            <div class="kv-row"><span>Airlines</span><b>{df['Airline'].nunique()}</b></div>
            <div class="kv-row"><span>Source / Destination cities</span><b>{df['Source'].nunique()} / {df['Destination'].nunique()}</b></div>
            <div class="kv-row"><span>Fare range</span><b>{inr(df['Price'].min())} – {inr(df['Price'].max())}</b></div>
            <div class="small" style="margin-top:.5rem">Cleaning: removed 1 row with missing values, 220 duplicate
            rows and standardised “No Info” → “No info”.</div>
            """
        )
    with c2, card("eq_pipe"):
        html('<div class="card-title">🛠️ Pipeline</div>')
        for step in [
            "Journey date → day and month",
            "Departure / arrival time → hour and minute",
            "Duration (e.g. 2h 50m) → total minutes",
            "Total stops → 0 to 4",
            "One-Hot Encoding: Airline, Source, Destination, Additional Info",
            "80/20 train–test split, 36 final features",
        ]:
            html(f'<div class="check"><i>✓</i>{step}</div>')

    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1, card("eq_mc"):
        html('<div class="card-title">🏁 Model comparison</div>')
        comp = fu.NOTEBOOK_MODEL_RESULTS.copy()
        best = int(comp["R2 Score"].idxmax())
        comp["MAE"] = comp["MAE"].map(lambda v: f"{v:,.2f}")
        comp["RMSE"] = comp["RMSE"].map(lambda v: f"{v:,.2f}")
        comp["R2 Score"] = comp["R2 Score"].map(lambda v: f"{v:.4f}")
        html(html_table(comp, best_row=best))
        st.caption(
            "Tuned with GridSearchCV (best: n_estimators=100, max_depth=20). "
            "5-fold cross-validation of the untuned forest gave a mean R² of 0.891."
        )
    with c2, card("eq_fm"):
        html('<div class="card-title">✅ Final model (this app)</div>')
        html(
            f"""
            <div class="kv-row"><span>Model</span><b>Tuned Random Forest</b></div>
            <div class="kv-row"><span>MAE</span><b>{mt['mae']:,.2f}</b></div>
            <div class="kv-row"><span>RMSE</span><b>{mt['rmse']:,.2f}</b></div>
            <div class="kv-row"><span>R² (test)</span><b>{mt['r2']:.4f}</b></div>
            <div class="kv-row"><span>R² (train)</span><b>{mt['train_r2']:.4f}</b></div>
            <div class="small" style="margin-top:.5rem">{bundle['source']}.</div>
            """
        )

    with card():
        html('<div class="card-title">🧩 Challenges and solutions</div>')
        items = [
            ("Missing values", "Route / Total_Stops were missing in one record → the record was removed."),
            ("Duplicate records", "220 exact duplicates → removed with drop_duplicates()."),
            ("Date and time columns", "Not usable directly → day, month, hour and minute extracted."),
            ("Duration as text", "Values like “2h 50m” → converted to total minutes."),
            ("Categorical features", "Airline, Source, Destination, Additional_Info → One-Hot Encoding."),
            ("Overfitting", "Random Forest train R² ≫ test R² → cross-validation and GridSearchCV."),
        ]
        a, b = st.columns(2)
        for i, (t, s) in enumerate(items):
            (a if i % 2 == 0 else b).markdown(f"**{t}** — {s}")

    with card():
        html('<div class="card-title">🚀 Future improvements</div>')
        st.markdown(
            "- Use larger and more recent datasets; add booking date, days left before departure, class and season.\n"
            "- Try XGBoost / Gradient Boosting and more extensive hyperparameter tuning.\n"
            "- Add real-time flight information and deploy the model online."
        )
    footer(df)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("page", "Home")

    df = get_dataset()
    data_key = f"{len(df)}-{int(df['Price'].sum())}"
    bundle = load_bundle(data_key, df)

    page = st.session_state["page"]
    st.markdown(page_bg_css(page), unsafe_allow_html=True)
    render_nav()

    if page == "Home":
        page_home(df, bundle)
    elif page == "Quick Prediction":
        page_predict(df, bundle)
    elif page == "Smart Analytics":
        page_analytics(df, bundle)
    elif page == "Plan Better":
        page_plan(df, bundle)
    else:
        page_about(df, bundle)


main()
