# ============================================================
# 🎬 CINEMATCH — Ultra Cinematic Movie Recommendation System
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import ast
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
img = get_base64("background_image.png")
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow+Condensed:wght@300;400;500;600;700;800&family=Share+Tech+Mono&display=swap');

/* ════════════════════════════
   DARK THEME VARIABLES (default)
════════════════════════════ */
:root {
    --bg-url: url("background_image.png");
    --surface:    rgba(255,255,255,0.030);
    --surface2:   rgba(255,255,255,0.055);
    --border:     rgba(255,255,255,0.07);
    --text:       #c8d6e8;
    --text-dim:   #4a5a6a;
    --text-mid:   #7a90a8;
    --red:        #E50914;
    --red-glow:   rgba(229,9,20,0.22);
    --cyan:       #00e5ff;
    --cyan-glow:  rgba(0,229,255,0.18);
    --purple:     #bf00ff;
    --gold:       #ffd700;
    --card-bg:    rgba(255,255,255,0.035);
    --card-border:rgba(255,255,255,0.065);
    --panel-bg:   rgba(10,10,22,0.97);
    --sb-bg:      #06060f;
    --ticker-fade:#03030a;
}

/* ════════════════════════════
   GLOBAL
════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    transition: background 0.4s ease, color 0.35s ease;
}

.stApp {
    background-image: var(--bg-url) !important;
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
}

/* ── DARK: aurora + grid ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 120% 60% at 50% -20%, rgba(229,9,20,0.20) 0%, transparent 65%),
        radial-gradient(ellipse 70%  50% at 100% 100%, rgba(0,229,255,0.09) 0%, transparent 55%),
        radial-gradient(ellipse 60%  40% at -10%  80%, rgba(191,0,255,0.08) 0%, transparent 55%);
    pointer-events: none; z-index: 0;
    animation: auroraShift 14s ease-in-out infinite alternate;
    opacity: 0.4;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: 0; left: -20%; right: -20%; height: 55vh;
    background:
        linear-gradient(transparent 0%, rgba(229,9,20,0.04) 100%),
        repeating-linear-gradient(90deg, transparent, transparent calc(6vw - 1px),
            rgba(229,9,20,0.06) calc(6vw - 1px), rgba(229,9,20,0.06) calc(6vw)),
        repeating-linear-gradient(180deg, transparent, transparent calc(4vh - 1px),
            rgba(229,9,20,0.04) calc(4vh - 1px), rgba(229,9,20,0.04) calc(4vh));
    transform: perspective(400px) rotateX(45deg);
    transform-origin: bottom center;
    pointer-events: none; z-index: 0; opacity: 0.55;
    animation: gridPulse 7s ease-in-out infinite;
    opacity: 0.2;
}

/* ── CRT scanlines ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 3px,
        rgba(0,0,0,0.06) 3px, rgba(0,0,0,0.06) 4px
    );
    pointer-events: none; z-index: 9997; opacity: 0.25;
    transition: opacity 0.4s ease;
}

/* ════════════════════════════
   KEYFRAMES
════════════════════════════ */
@keyframes auroraShift { 0%{opacity:.7;transform:scale(1);} 100%{opacity:1;transform:scale(1.05);} }
@keyframes gridPulse   { 0%,100%{opacity:.35;} 50%{opacity:.65;} }
@keyframes titleSlam   {
    0%  {opacity:0;letter-spacing:28px;transform:scaleY(1.3);filter:blur(12px);}
    60% {opacity:1;letter-spacing:-4px;transform:scaleY(1);filter:blur(0);}
    75% {letter-spacing:2px;}
    100%{letter-spacing:-2px;}
}
@keyframes subFade  { from{opacity:0;transform:translateY(14px);filter:blur(4px);} to{opacity:1;transform:none;filter:none;} }
@keyframes lineExp  { from{transform:scaleX(0);opacity:0;} to{transform:scaleX(1);opacity:1;} }
@keyframes redPulse {
    0%,100%{text-shadow:0 0 8px var(--red),0 0 24px rgba(229,9,20,.55),0 0 55px rgba(229,9,20,.22);}
    50%    {text-shadow:0 0 4px var(--red),0 0 14px rgba(229,9,20,.35);}
}
@keyframes holoPan   { 0%,100%{background-position:0% 50%;} 50%{background-position:100% 50%;} }
@keyframes bMarch    { 0%{background-position:0%;} 100%{background-position:200%;} }
@keyframes shimmer   { 0%{transform:translateX(-120%) skewX(-15deg);} 100%{transform:translateX(320%) skewX(-15deg);} }
@keyframes ticker    { from{transform:translateX(100%);} to{transform:translateX(-100%);} }
@keyframes spin3d    { from{transform:translate(-50%,-50%) rotate(0deg);} to{transform:translate(-50%,-50%) rotate(360deg);} }
@keyframes spin3dR   { from{transform:translate(-50%,-50%) rotate(0deg);} to{transform:translate(-50%,-50%) rotate(-360deg);} }
@keyframes blink     { 0%,90%,100%{opacity:1;} 95%{opacity:.5;} }

/* ════════════════════════════
   SCROLL-REVEAL BASE STATES
   JS picks one class per element randomly.
   Entering viewport → adds .sr-in
   Leaving viewport  → removes .sr-in (re-animates on next scroll)
════════════════════════════ */
.sr { will-change: transform, opacity; }

.sr-fade-up    { opacity:0; transform:translateY(44px); }
.sr-fade-down  { opacity:0; transform:translateY(-36px); }
.sr-slide-left { opacity:0; transform:translateX(-50px); }
.sr-slide-right{ opacity:0; transform:translateX(50px); }
.sr-zoom-in    { opacity:0; transform:scale(0.78); }
.sr-flip       { opacity:0; transform:rotateY(35deg) scale(0.9); }
.sr-tilt       { opacity:0; transform:rotate(-10deg) scale(0.88) translateY(20px); }
.sr-drop       { opacity:0; transform:translateY(-60px) scale(1.05); }

/* Transition applied when visible — delay injected via data-delay attr */
.sr.sr-in {
    opacity: 1;
    transform: none;
    transition:
        opacity   0.60s cubic-bezier(0.22, 1, 0.36, 1),
        transform 0.60s cubic-bezier(0.22, 1, 0.36, 1);
    transition-delay: var(--sr-delay, 0ms);
}
/* ════════════════════════════
   FILM TICKER
════════════════════════════ */
.filmstrip-ticker {
    width: 100%; overflow: hidden;
    background: linear-gradient(90deg, var(--bg), rgba(229,9,20,0.07), var(--bg));
    border-top:    1px solid rgba(229,9,20,0.17);
    border-bottom: 1px solid rgba(229,9,20,0.17);
    padding: 7px 0; position: relative;
}
.filmstrip-ticker::before,
.filmstrip-ticker::after {
    content:''; position:absolute; top:0; bottom:0; width:80px; z-index:2;
}
.filmstrip-ticker::before { left:0;  background: linear-gradient(90deg,  var(--ticker-fade), transparent); }
.filmstrip-ticker::after  { right:0; background: linear-gradient(-90deg, var(--ticker-fade), transparent); }
.ticker-inner {
    display: flex; white-space: nowrap;
    animation: ticker 32s linear infinite;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.70rem; letter-spacing: 3px;
    color: rgba(229,9,20,0.55); text-transform: uppercase;
}
.ticker-sep { margin: 0 20px; color: rgba(0,229,255,0.30); }

/* ════════════════════════════
   HERO
════════════════════════════ */
.hero-outer {
    position: relative; text-align: center;
    padding: 52px 0 40px; overflow: hidden;
}
.arc1 {
    position:absolute; top:50%; left:50%;
    width:600px; height:600px;
    border:1px solid rgba(229,9,20,0.05); border-radius:50%;
    animation: spin3d  42s linear infinite; pointer-events:none;
}
.arc2 {
    position:absolute; top:50%; left:50%;
    width:420px; height:420px;
    border:1px solid rgba(0,229,255,0.04); border-radius:50%;
    animation: spin3dR 27s linear infinite; pointer-events:none;
}
.hero-eyebrow {
    font-family:'Share Tech Mono',monospace;
    font-size:.70rem; letter-spacing:6px; text-transform:uppercase;
    color:rgba(229,9,20,0.72); margin-bottom:12px;
    animation: subFade .7s ease both;
}
.hero-eyebrow span { color:rgba(0,229,255,0.45); margin:0 8px; }
.hero-title-cine {
    font-family:'Bebas Neue',sans-serif;
    font-size: clamp(3.8rem,9vw,8rem);
    letter-spacing:-2px; line-height:.9; color:#fff; display:inline-block;
    animation: titleSlam 1.1s cubic-bezier(.16,1,.3,1) .1s both, redPulse 5s ease-in-out 1.4s infinite;
}
.hero-title-match {
    font-family:'Bebas Neue',sans-serif;
    font-size: clamp(3.8rem,9vw,8rem);
    letter-spacing:-2px; line-height:.9; display:inline-block;
    background: linear-gradient(135deg,#00e5ff 0%,#bf00ff 50%,#00e5ff 100%);
    background-size:200% 100%;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation: titleSlam 1.1s cubic-bezier(.16,1,.3,1) .2s both, holoPan 4s linear 1.5s infinite;
}
.hero-divider {
    width:200px; height:2px; margin:16px auto;
    background: linear-gradient(90deg,transparent,var(--red),var(--cyan),transparent);
    transform-origin:center;
    animation: lineExp .8s cubic-bezier(.16,1,.3,1) .7s both;
}
.hero-subtitle {
    font-family:'Barlow Condensed',sans-serif;
    font-size:1.0rem; letter-spacing:6px; text-transform:uppercase;
    color:rgba(200,214,232,0.40);
    animation: subFade .8s ease .9s both;
}

/* ════════════════════════════
   STAT CARDS
════════════════════════════ */
.stat-card {
    position:relative;
    background:var(--surface);
    border:1px solid var(--border); border-radius:4px;
    padding:20px 14px 16px; text-align:center; overflow:hidden;
    transition: transform .3s ease, border-color .3s ease, box-shadow .3s ease;
    cursor:default;
}
.stat-card:hover {
    transform:translateY(-4px);
    border-color:rgba(229,9,20,.30);
    box-shadow: 0 8px 28px rgba(229,9,20,.18), 0 0 0 1px rgba(229,9,20,.15);
}
.stat-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,transparent,var(--sc,#E50914),transparent);
    background-size:200%; animation:bMarch 3s linear infinite;
}
.stat-card::after {
    content:''; position:absolute; top:0; left:0; width:45%; height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.04),transparent);
    animation:shimmer 4s ease-in-out infinite; animation-delay:var(--sd,0s);
}
.stat-number {
    font-family:'Bebas Neue',sans-serif; font-size:2.6rem; line-height:1;
    color:var(--sc,#E50914); display:block;
    text-shadow:0 0 18px var(--sc,#E50914);
    animation:blink 4s ease-in-out infinite; animation-delay:var(--sd,0s);
}
.stat-label {
    font-family:'Share Tech Mono',monospace; font-size:.62rem;
    letter-spacing:3px; text-transform:uppercase; color:var(--text-dim); margin-top:5px;
}

/* ════════════════════════════
   SEARCH FRAME
════════════════════════════ */
.search-frame {
    position:relative;
    background:linear-gradient(135deg,rgba(229,9,20,.055),rgba(0,229,255,.03),rgba(191,0,255,.03));
    border:1px solid rgba(229,9,20,.22); border-radius:6px;
    padding:22px 28px 18px; margin:0 0 28px;
    box-shadow:0 0 50px rgba(229,9,20,.07), inset 0 1px 0 rgba(255,255,255,.05);
}
.search-frame::before {
    content:''; position:absolute; top:-1px; left:-1px; width:20px; height:20px;
    border-top:2px solid var(--cyan); border-left:2px solid var(--cyan);
    border-radius:2px 0 0 0; box-shadow:-2px -2px 8px rgba(0,229,255,.35);
}
.search-frame::after {
    content:''; position:absolute; bottom:-1px; right:-1px; width:20px; height:20px;
    border-bottom:2px solid var(--red); border-right:2px solid var(--red);
    border-radius:0 0 2px 0; box-shadow:2px 2px 8px rgba(229,9,20,.35);
}
.search-label {
    font-family:'Share Tech Mono',monospace; font-size:.70rem;
    letter-spacing:4px; text-transform:uppercase; color:var(--cyan); margin-bottom:12px;
}
.search-label::before { content:'// '; color:var(--red); }

/* selectbox */
div[data-testid="stSelectbox"] > div > div {
    background:rgba(0,0,0,.50) !important; border:1px solid rgba(229,9,20,.20) !important;
    border-radius:4px !important; color:var(--text) !important;
    font-family:'Barlow Condensed',sans-serif !important; font-size:1.05rem !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color:rgba(0,229,255,.40) !important;
    box-shadow:0 0 0 2px rgba(0,229,255,.08),0 0 20px rgba(0,229,255,.10) !important;
}

/* primary button */
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#E50914,#9b0000) !important;
    border:1px solid rgba(229,9,20,.50) !important; border-radius:4px !important;
    font-family:'Bebas Neue',sans-serif !important; font-size:1.1rem !important;
    letter-spacing:3px !important; color:#fff !important; height:46px !important;
    box-shadow:0 0 18px rgba(229,9,20,.38) !important; transition:all .2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background:linear-gradient(135deg,#ff1a2a,#E50914) !important;
    box-shadow:0 0 38px rgba(229,9,20,.65),0 0 70px rgba(229,9,20,.25) !important;
    transform:translateY(-2px) !important;
}
.stButton > button[kind="primary"]:active { transform:translateY(0) !important; }

/* ════════════════════════════
   SECTION HEADERS
════════════════════════════ */
.section-hdr {
    display:flex; align-items:center; gap:14px; margin:32px 0 18px;
}
.section-hdr-tag {
    font-family:'Share Tech Mono',monospace; font-size:.60rem;
    letter-spacing:3px; text-transform:uppercase;
    color:var(--red); background:rgba(229,9,20,.08);
    border:1px solid rgba(229,9,20,.22);
    padding:3px 10px 2px; border-radius:2px; flex-shrink:0;
}
.section-hdr-title {
    font-family:'Bebas Neue',sans-serif; font-size:1.05rem;
    letter-spacing:4px; color:var(--text); flex-shrink:0;
}
.section-hdr-line {
    flex:1; height:1px;
    background:linear-gradient(90deg,rgba(229,9,20,.40),rgba(0,229,255,.15),transparent);
}
.section-hdr-count {
    font-family:'Share Tech Mono',monospace; font-size:.62rem;
    letter-spacing:2px; color:var(--text-dim); flex-shrink:0;
}

/* ════════════════════════════
   SOURCE PANEL
════════════════════════════ */
.source-panel {
    position:relative;
    background:var(--panel-bg);
    border:1px solid rgba(229,9,20,.18); border-radius:6px;
    padding:28px; margin:10px 0 6px; overflow:hidden;
    box-shadow:0 25px 70px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.04);
    transition:box-shadow .4s ease;
}
.source-panel:hover {
    box-shadow:0 30px 90px rgba(0,0,0,.65),0 0 50px rgba(229,9,20,.10),
               inset 0 1px 0 rgba(255,255,255,.06);
}
.source-panel::before {
    content:''; position:absolute; top:0; left:0; width:3px; height:100%;
    background:linear-gradient(180deg,var(--red),var(--purple),var(--cyan));
    border-radius:6px 0 0 6px; box-shadow:0 0 12px var(--red);
}
.source-panel::after {
    content:'NOW PLAYING';
    position:absolute; top:50%; right:-18px;
    transform:translateY(-50%) rotate(90deg);
    font-family:'Bebas Neue',sans-serif; font-size:4.5rem;
    color:rgba(229,9,20,.025); letter-spacing:8px;
    pointer-events:none; user-select:none; white-space:nowrap;
}

.poster-container {
    position:relative; border-radius:6px; overflow:hidden;
    box-shadow:0 0 0 1px rgba(229,9,20,.15),0 18px 50px rgba(0,0,0,.75),
               0 0 35px rgba(229,9,20,.20);
    transition:box-shadow .4s ease;
}
.source-panel:hover .poster-container {
    box-shadow:0 0 0 1px rgba(229,9,20,.28),0 22px 65px rgba(0,0,0,.85),
               0 0 50px rgba(229,9,20,.28);
}
.poster-container img { width:100%; border-radius:6px; display:block; }
.poster-glow {
    position:absolute; bottom:-14px; left:50%; transform:translateX(-50%);
    width:80%; height:22px; background:rgba(229,9,20,.28);
    filter:blur(16px); border-radius:50%; pointer-events:none;
}

.src-title {
    font-family:'Bebas Neue',sans-serif; font-size:1.9rem; letter-spacing:2px;
    color:var(--text); line-height:1.05; margin:0 0 10px;
}
.src-rating {
    font-family:'Barlow Condensed',sans-serif; font-size:1.1rem;
    font-weight:600; color:var(--gold); letter-spacing:1px; margin-bottom:10px;
}
.src-rating .score {
    font-family:'Bebas Neue',sans-serif; font-size:1.7rem;
    text-shadow:0 0 12px rgba(255,215,0,.35); margin-left:4px;
}
.src-meta {
    font-family:'Share Tech Mono',monospace; font-size:.76rem;
    letter-spacing:2px; color:var(--text-dim); margin-bottom:12px;
}
.genre-wrap { display:flex; flex-wrap:wrap; gap:5px; margin-top:12px; }
.genre-pill {
    font-family:'Barlow Condensed',sans-serif; font-size:.80rem;
    font-weight:600; letter-spacing:1.5px; text-transform:uppercase;
    color:#bf80ff; background:rgba(191,0,255,.07);
    border:1px solid rgba(191,0,255,.20);
    padding:3px 11px 2px; border-radius:3px;
    transition:all .2s ease; cursor:default;
}
.genre-pill:hover {
    background:rgba(191,0,255,.15); border-color:rgba(191,0,255,.45);
    transform:translateY(-2px) scale(1.04);
    box-shadow:0 4px 12px rgba(191,0,255,.20);
}

/* recs divider */
.recs-lead {
    display:flex; align-items:center; gap:14px; margin:20px 0 16px;
}
.recs-lead-line {
    flex:1; height:1px;
    background:linear-gradient(90deg,rgba(229,9,20,.30),rgba(0,229,255,.12),transparent);
}
.recs-lead-count {
    font-family:'Share Tech Mono',monospace; font-size:.62rem;
    letter-spacing:3px; text-transform:uppercase; color:var(--text-dim); flex-shrink:0;
}

/* ════════════════════════════
   MOVIE CARDS
════════════════════════════ */
.movie-card {
    position:relative;
    background:var(--card-bg); border:1px solid var(--card-border);
    border-radius:6px; overflow:hidden; margin-bottom:6px;
    /* transform applied by tilt JS; transition only for non-tilt props */
    transition: border-color .3s ease, box-shadow .3s ease;
}

/* shine sweep on hover */
.movie-card::before {
    content:''; position:absolute; top:0; left:-90%; width:55%; height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.07),transparent);
    transform:skewX(-12deg);
    transition:left .5s ease; z-index:2; pointer-events:none;
}
.movie-card:hover::before { left:160%; }

/* glow ring on hover */
.movie-card::after {
    content:''; position:absolute; inset:-1px; border-radius:6px;
    opacity:0; transition:opacity .3s ease; pointer-events:none; z-index:3;
    background:linear-gradient(135deg,rgba(229,9,20,.10),rgba(0,229,255,.05));
    box-shadow:inset 0 0 0 1px rgba(229,9,20,.22);
}
.movie-card:hover::after { opacity:1; }
.movie-card:hover {
    border-color:rgba(229,9,20,.40);
    box-shadow:0 20px 50px rgba(0,0,0,.70),0 0 22px rgba(229,9,20,.22),
               0 0 0 1px rgba(229,9,20,.18);
    z-index:10;
}

/* bottom accent bar */
.card-bar {
    position:absolute; bottom:0; left:0; right:0; height:3px; z-index:4;
    background:linear-gradient(90deg,var(--red),var(--cyan));
    transform:scaleX(0); transform-origin:left;
    transition:transform .38s cubic-bezier(.22,1,.36,1);
}
.movie-card:hover .card-bar { transform:scaleX(1); }

/* gold badge fades in */
.card-badge {
    position:absolute; top:8px; right:8px; z-index:5;
    background:rgba(0,0,0,.75); backdrop-filter:blur(4px);
    border:1px solid rgba(255,215,0,.30); border-radius:3px;
    padding:2px 7px;
    font-family:'Bebas Neue',sans-serif; font-size:.85rem; color:var(--gold);
    opacity:0; transform:translateY(-4px);
    transition:opacity .25s ease, transform .25s ease;
}
.movie-card:hover .card-badge { opacity:1; transform:translateY(0); }

/* poster */
.card-poster-wrap {
    position:relative; overflow:hidden; aspect-ratio:2/3;
}
.card-poster-wrap img {
    width:100%; height:100%; object-fit:cover; display:block;
    transition:transform .45s ease, filter .45s ease;
}
.movie-card:hover .card-poster-wrap img {
    transform:scale(1.07); filter:brightness(1.08) saturate(1.15);
}
.card-poster-wrap::after {
    content:''; position:absolute; bottom:0; left:0; right:0; height:55%;
    background:linear-gradient(transparent,rgba(3,3,10,.94) 85%,rgba(3,3,10,1));
    pointer-events:none; transition:opacity .3s ease;
}

/* card text */
.card-info { padding:9px 10px 11px; }
.card-title {
    font-family:'Barlow Condensed',sans-serif; font-size:.86rem;
    font-weight:700; color:var(--text); line-height:1.3;
    min-height:34px; text-align:center; letter-spacing:.3px;
}
.card-rating {
    font-family:'Bebas Neue',sans-serif; font-size:.95rem; letter-spacing:1px;
    color:var(--gold); text-align:center; margin-top:3px;
    text-shadow:0 0 8px rgba(255,215,0,.25);
}
.card-match {
    font-family:'Share Tech Mono',monospace; font-size:.66rem; letter-spacing:1.5px;
    color:var(--cyan); text-align:center; margin-top:2px;
    text-shadow:0 0 7px rgba(0,229,255,.35);
}
.card-year {
    font-family:'Share Tech Mono',monospace; font-size:.62rem; letter-spacing:2px;
    color:var(--text-dim); text-align:center; margin-top:2px;
}

/* ════════════════════════════
   GENRE BANNER
════════════════════════════ */
.genre-banner {
    position:relative;
    background:linear-gradient(135deg,var(--surface),rgba(229,9,20,.04));
    border:1px solid var(--border); border-left:3px solid var(--red);
    border-radius:4px; padding:16px 22px;
    margin:8px 0 20px; overflow:hidden;
    display:flex; align-items:center; gap:18px;
}
.genre-banner::before {
    content:''; position:absolute; inset:0;
    background:linear-gradient(90deg,rgba(229,9,20,.03),transparent 60%);
    pointer-events:none;
}
.genre-banner-title {
    font-family:'Bebas Neue',sans-serif; font-size:1.5rem;
    letter-spacing:3px; color:var(--text); flex-shrink:0; line-height:1;
}
.genre-banner-sub {
    font-family:'Share Tech Mono',monospace; font-size:.60rem;
    letter-spacing:3px; text-transform:uppercase; color:var(--text-dim);
}
.genre-banner-line {
    flex:1; height:1px;
    background:linear-gradient(90deg,rgba(229,9,20,.30),transparent);
}
.genre-banner-badge {
    font-family:'Share Tech Mono',monospace; font-size:.58rem;
    letter-spacing:2px; text-transform:uppercase;
    color:var(--cyan); background:rgba(0,229,255,.08);
    border:1px solid rgba(0,229,255,.20);
    padding:3px 10px 2px; border-radius:2px; flex-shrink:0;
}

/* ════════════════════════════
   TABS
════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(0,0,0,.35) !important; border-radius:6px !important;
    padding:4px 5px !important; gap:2px !important;
    border:1px solid var(--border) !important;
    backdrop-filter:blur(8px) !important; flex-wrap:wrap !important;
}
.stTabs [data-baseweb="tab"] {
    color:var(--text-dim) !important; font-family:'Barlow Condensed',sans-serif !important;
    font-size:.84rem !important; font-weight:700 !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
    border-radius:4px !important; padding:6px 15px 5px !important;
    transition:all .2s ease !important; border:none !important; margin:2px !important;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--red) !important; background:rgba(229,9,20,.08) !important; }
.stTabs [aria-selected="true"] {
    color:#fff !important;
    background:linear-gradient(135deg,rgba(229,9,20,.20),rgba(229,9,20,.08)) !important;
    border:1px solid rgba(229,9,20,.28) !important;
    box-shadow:0 0 10px rgba(229,9,20,.20) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display:none !important; }

/* ════════════════════════════
   SIDEBAR
════════════════════════════ */
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,var(--sb-bg),#08080f) !important;
    border-right:1px solid rgba(229,9,20,.10) !important;
    transition:background .4s ease;
}
section[data-testid="stSidebar"]::after {
    content:''; position:absolute; top:0; right:-1px; width:1px; height:100%;
    background:linear-gradient(180deg,rgba(229,9,20,.40),rgba(0,229,255,.20),transparent);
    pointer-events:none;
}
.sb-brand {
    font-family:'Bebas Neue',sans-serif; font-size:1.8rem; letter-spacing:4px;
    background:linear-gradient(90deg,var(--red),#ff6b35);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    display:block; line-height:1; animation:redPulse 4s ease-in-out infinite;
}
.sb-ver {
    font-family:'Share Tech Mono',monospace; font-size:.60rem;
    letter-spacing:3px; color:var(--text-dim); text-transform:uppercase; margin-bottom:20px;
}
.sb-divider {
    height:1px; margin:12px 0;
    background:linear-gradient(90deg,rgba(229,9,20,.30),rgba(0,229,255,.10),transparent);
}
.sb-label {
    font-family:'Share Tech Mono',monospace; font-size:.60rem;
    letter-spacing:4px; text-transform:uppercase; color:var(--cyan); opacity:.70;
    margin-bottom:4px; display:block;
}
.sb-status-row { display:flex; gap:8px; margin-top:12px; }
.sb-status {
    flex:1; text-align:center; background:var(--surface);
    border:1px solid var(--border); border-radius:4px; padding:7px 4px;
    font-family:'Share Tech Mono',monospace; font-size:.58rem;
    letter-spacing:2px; color:var(--text-dim); text-transform:uppercase;
}
.sb-status-on  { color:#00ff9d; border-color:rgba(0,255,157,.15); }
.sb-status-off { color:var(--red); border-color:rgba(229,9,20,.15); }

/* slider */
[data-testid="stSlider"] > div > div > div > div { background:var(--red) !important; }
.stSlider [data-testid="stThumbValue"] {
    background:var(--red) !important;
    font-family:'Share Tech Mono',monospace !important; font-size:.68rem !important;
}

/* spinner */
.stSpinner > div {
    border-top-color:var(--red) !important;
    border-right-color:rgba(229,9,20,.12) !important;
    border-bottom-color:rgba(229,9,20,.12) !important;
    border-left-color:rgba(229,9,20,.12) !important;
}

/* scrollbar */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:linear-gradient(180deg,var(--red),var(--purple)); border-radius:3px; }

/* alert */
.stAlert {
    border-radius:4px !important; border:1px solid rgba(229,9,20,.28) !important;
    background:rgba(229,9,20,.05) !important; font-family:'Barlow Condensed',sans-serif !important;
}

/* ════════════════════════════
   FOOTER
════════════════════════════ */
.footer {
    text-align:center; padding:44px 0 24px; margin-top:44px;
    border-top:1px solid var(--border); position:relative;
}
.footer::before {
    content:''; position:absolute; top:0; left:10%; right:10%; height:1px;
    background:linear-gradient(90deg,transparent,var(--red),var(--cyan),transparent);
    opacity:.35;
}
.footer-logo {
    font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:6px;
    background:linear-gradient(135deg,var(--text) 30%,var(--text-dim) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.footer-stack {
    font-family:'Share Tech Mono',monospace; font-size:.62rem;
    letter-spacing:3px; text-transform:uppercase; color:var(--text-dim); margin-top:7px;
}
.footer-sep { color:var(--red); opacity:.35; margin:0 8px; }

/* misc */
h1,h2,h3,h4,h5 { font-family:'Barlow Condensed',sans-serif !important; color:var(--text) !important; }
header[data-testid="stHeader"] { background:transparent !important; border-bottom:none !important; }
.block-container { padding-top:0 !important; }
/* collapse Streamlit's gap around the search widget row */
.element-container { margin-bottom:0 !important; }
div[data-testid="stButton"]   { margin-top:0 !important; }
div[data-testid="stSelectbox"]{ margin-top:0 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# THEME TOGGLE + SCROLL-REVEAL + 3-D TILT  (all in one block)
# ============================================================
st.markdown("""
<script>
(function(){
/* ─────────────────────────────────────────
   LIGHT-MODE CSS OVERRIDE (injected as <style>)
───────────────────────────────────────── */
var LIGHT = `
  :root {
    --bg:#f0f2f8 !important; --bg2:#e4e8f2 !important;
    --surface:rgba(0,0,0,0.04) !important; --surface2:rgba(0,0,0,0.07) !important;
    --border:rgba(0,0,0,0.10) !important;
    --text:#1a2030 !important; --text-dim:#8090a8 !important; --text-mid:#4a5a7a !important;
    --red:#c0000c !important; --red-glow:rgba(192,0,12,0.18) !important;
    --cyan:#007acc !important; --cyan-glow:rgba(0,122,204,0.15) !important;
    --purple:#7000cc !important; --gold:#b8860b !important;
    --card-bg:rgba(255,255,255,0.80) !important;
    --card-border:rgba(0,0,0,0.09) !important;
    --panel-bg:rgba(255,255,255,0.94) !important;
    --sb-bg:#e8eaf4 !important;
    --ticker-fade:#f0f2f8 !important;
  }
  .stApp { background:var(--bg) !important; }
  .stApp::before, .stApp::after { opacity:0.20 !important; }
  [data-testid="stAppViewContainer"]::before { opacity:0 !important; }
  .card-poster-wrap::after {
    background:linear-gradient(transparent, rgba(235,238,248,0.93) 85%) !important;
  }
  section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#e8eaf4,#dde0f0) !important;
  }
  #cm-toggle {
    background:rgba(0,0,0,0.06) !important;
    border-color:rgba(0,0,0,0.14) !important;
    color:#2a3a5a !important;
  }
  .hero-title-cine { color:#1a2030 !important; }
  div[data-testid="stSelectbox"] > div > div {
    background:rgba(255,255,255,0.80) !important;
    border-color:rgba(0,0,0,0.15) !important;
    color:#1a2030 !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    background:rgba(255,255,255,0.60) !important;
  }
`;

var styleEl = null;
var isLight  = false;

function applyLight(){
  styleEl = document.getElementById('cm-light-css');
  if(!styleEl){
    styleEl = document.createElement('style');
    styleEl.id = 'cm-light-css';
    document.head.appendChild(styleEl);
  }
  styleEl.textContent = LIGHT;
}
function applyDark(){
  styleEl = document.getElementById('cm-light-css');
  if(styleEl) styleEl.textContent = '';
}

function setTheme(light){
  isLight = light;
  light ? applyLight() : applyDark();
  var icon  = document.getElementById('cm-toggle-icon');
  var label = document.getElementById('cm-toggle-label');
  if(icon)  icon.textContent  = light ? '🌙' : '☀️';
  if(label) label.textContent = light ? 'DARK MODE' : 'LIGHT MODE';
  try{ sessionStorage.setItem('cm-theme', light ? 'light' : 'dark'); }catch(e){}
}

function wireToggle(){
  var btn = document.getElementById('cm-toggle');
  if(!btn || btn.dataset.wired) return;
  btn.dataset.wired = '1';
  btn.addEventListener('click', function(){
    var isL = (sessionStorage.getItem('cm-theme') === 'light');
    setTheme(!isL);
  });
}

function restoreTheme(){
  try{
    if(sessionStorage.getItem('cm-theme') === 'light' && !isLight) setTheme(true);
  }catch(e){}
}

/* ─────────────────────────────────────────
   SCROLL-REVEAL
   8 random entrance classes — assigned once per element.
   IntersectionObserver adds .sr-in on enter, removes on exit → re-animates.
───────────────────────────────────────── */
var SR_CLASSES = [
  'sr-fade-up','sr-fade-down','sr-slide-left','sr-slide-right',
  'sr-zoom-in','sr-flip','sr-tilt','sr-drop'
];

var observer = null;

function buildObserver(){
  observer = new IntersectionObserver(function(entries){
    entries.forEach(function(en){
      var el = en.target;
      if(en.isIntersecting){
        var d = parseInt(el.dataset.delay || '0');
        clearTimeout(el._srT);
        el._srT = setTimeout(function(){
          el.style.setProperty('--sr-delay', d + 'ms');
          el.classList.add('sr-in');
        }, 0);
      } else {
        clearTimeout(el._srT);
        el.classList.remove('sr-in');
        /* clear inline delay so next enter uses data-delay again */
        el.style.removeProperty('--sr-delay');
      }
    });
  }, {
    root: null,
    threshold: 0.07,
    rootMargin: '0px 0px -30px 0px'
  });
}

function initSR(){
  if(!observer) buildObserver();
  document.querySelectorAll('.sr:not([data-sr-init])').forEach(function(el){
    el.dataset.srInit = '1';
    var cls = SR_CLASSES[Math.floor(Math.random() * SR_CLASSES.length)];
    el.classList.add(cls);
    observer.observe(el);
  });
}

/* ─────────────────────────────────────────
   3-D TILT on movie cards
───────────────────────────────────────── */
function initTilt(){
  document.querySelectorAll('.movie-card:not([data-tilt])').forEach(function(card){
    card.dataset.tilt = '1';
    card.addEventListener('mousemove', function(e){
      var r  = card.getBoundingClientRect();
      var dx = ((e.clientX - r.left) / r.width  - 0.5) * 2;
      var dy = ((e.clientY - r.top)  / r.height - 0.5) * 2;
      card.style.transform =
        'translateY(-8px) scale(1.03) perspective(700px)'+
        ' rotateY('+( dx*8)+'deg) rotateX('+(-dy*6)+'deg)';
    });
    card.addEventListener('mouseleave', function(){
      card.style.transform = '';
    });
  });
}

/* ─────────────────────────────────────────
   MASTER RUN
───────────────────────────────────────── */
function run(){
  wireToggle();
  restoreTheme();
  initSR();
  initTilt();
}

new MutationObserver(function(){ run(); })
  .observe(document.body, { childList:true, subtree:true });

setTimeout(run, 150);
setTimeout(run, 500);
setTimeout(run, 1200);

})();
</script>""", unsafe_allow_html=True)
# ============================================================
# CONSTANTS
# ============================================================
try:
    OMDB_API_KEY = st.secrets["OMDB_KEY"]
except:
    OMDB_API_KEY = "11880d33"   # ← paste your key here when ready

TMDB_IMG    = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://placehold.co/500x750/06060f/1a1a2e?text=🎬"

# ============================================================
# DATA LOADING  (unchanged)
# ============================================================
@st.cache_data(show_spinner=False)
def load_p1():
    try:
        df = pd.read_csv("https://drive.google.com/uc?id=16yOpEE7M8gbdb81IWYgm7pUpXNJ1LQRR")
        def tl(x):
            try: return ast.literal_eval(x)
            except: return []
        df['genres'] = df['genres'].apply(tl)
        df['cast']   = df['cast'].apply(tl)
        df['crew']   = df['crew'].apply(tl)
        df['tags']   = df['tags'].fillna('')
        tfidf  = TfidfVectorizer(max_features=5000, stop_words='english')
        matrix = tfidf.fit_transform(df['tags'])
        sim    = cosine_similarity(matrix, matrix)
        return df, sim, True
    except:
        return None, None, False

@st.cache_data(show_spinner=False)
def load_p2():
    try:
        df = pd.read_csv("https://drive.google.com/uc?id=1QvmvfDxTjAbQxSdJBrutPkUNQuz4tviT")
        def tl(x):
            try: return ast.literal_eval(x)
            except: return []
        df['genres_list'] = df['genres_list'].apply(tl)
        df['tags']        = df['tags'].fillna('')
        return df, True
    except:
        return None, False

# ============================================================
# POSTER FUNCTIONS  (unchanged)
# ============================================================
@st.cache_data(show_spinner=False)
def poster_p2(path):
    if path and str(path) not in ['nan', '', 'None']:
        return TMDB_IMG + str(path)
    return PLACEHOLDER

@st.cache_data(show_spinner=False)
def poster_p1(title, year=""):
    if not OMDB_API_KEY:
        return PLACEHOLDER
    try:
        url = (f"http://www.omdbapi.com/"
               f"?t={requests.utils.quote(str(title))}"
               f"&y={year}&apikey={OMDB_API_KEY}")
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            p = r.json().get('Poster', '')
            if p and p != 'N/A':
                return p
    except:
        pass
    return PLACEHOLDER

# ============================================================
# RECOMMEND FUNCTIONS  (unchanged)
# ============================================================
def rec_p1(title, n, year_range, df, sim):
    m = df[df['title'].str.lower() == title.lower()]
    if m.empty: return None, None
    idx    = m.index[0]
    scores = sorted(list(enumerate(sim[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
    inds   = [i[0] for i in scores]
    res    = df.iloc[inds][['movie_id','title','weighted_rating','release_year','genres']].copy()
    res['similarity'] = [round(s[1], 3) for s in scores]
    res = res[(res['release_year'] >= year_range[0]) & (res['release_year'] <= year_range[1])]
    res['poster'] = res.apply(
        lambda r: poster_p1(r['title'],
                            str(int(r['release_year'])) if pd.notna(r['release_year']) else ""),
        axis=1)
    return res, df.iloc[idx]

def rec_p2(title, n, min_votes, year_range, df, idx_faiss, norm):
    m = df[df['title'].str.lower() == title.lower()]
    if m.empty: return None, None
    midx        = m.index[0]
    qv          = norm[midx:midx+1]
    dists, inds = idx_faiss.search(qv, 150)
    res = df.iloc[inds[0][1:]][
        ['movie_id','title','weighted_rating','vote_count',
         'release_date','genres_list','original_language','poster_path']
    ].copy()
    res['similarity'] = dists[0][1:].round(3)
    res['year']       = pd.to_datetime(res['release_date'], errors='coerce').dt.year
    res = res[res['vote_count'] >= min_votes]
    res = res[(res['year'] >= year_range[0]) & (res['year'] <= year_range[1])]
    mv  = res['vote_count'].max() if len(res) > 0 else 1
    res['final_score'] = (res['similarity'] * 0.7 + res['vote_count'] / max(mv, 1) * 0.3)
    res = res.sort_values('final_score', ascending=False).head(n).reset_index(drop=True)
    res['poster'] = res['poster_path'].apply(poster_p2)
    return res, df.iloc[midx]

# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("INITIALISING NEURAL MATRIX..."):
    df1, sim1, ok1   = load_p1()
    df2, ok2         = load_p2()
    fi, fn, ok_faiss = None, None, False

# ============================================================
# COMBINED TITLE LIST
# ============================================================
all_titles = []
if ok1: all_titles += df1['title'].dropna().tolist()
if ok2:
    p1set       = set(all_titles)
    all_titles += [t for t in df2['title'].dropna().tolist() if t not in p1set]
all_titles = sorted(set(all_titles))

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<span class="sb-brand">⬡ CINEMATCH</span>'
        '<div class="sb-ver">v2.0 &nbsp;·&nbsp; DUAL ENGINE</div>',
        unsafe_allow_html=True)
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sb-label">// RESULT COUNT</span>', unsafe_allow_html=True)
    n_recs = st.slider("Recommendations", min_value=5, max_value=20, value=10, step=1,
                       label_visibility="collapsed")
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sb-label">// RELEASE YEAR</span>', unsafe_allow_html=True)
    year_range = st.slider("Year range", min_value=1950, max_value=2024,
                           value=(1990, 2024), step=1, label_visibility="collapsed")
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sb-label">// QUALITY THRESHOLD</span>', unsafe_allow_html=True)
    min_votes = st.slider("Min votes", min_value=100, max_value=5000,
                          value=500, step=100, label_visibility="collapsed")
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    e1c = "sb-status-on"  if ok1              else "sb-status-off"
    e2c = "sb-status-off"
    st.markdown(f"""
    <div class="sb-status-row">
        <div class="sb-status {e1c}">{'●' if ok1 else '○'} COSINE</div>
        <div class="sb-status {e2c}">{'●' if ok2 and ok_faiss else '○'} FAISS</div>
    </div>""", unsafe_allow_html=True)

# ============================================================
# FILM TICKER
# ============================================================
sep   = '<span class="ticker-sep"> ◆◆◆ </span>'
items = ["Phase 1 · 4,800 Films","Phase 2 · 26,000 Films",
         "TF-IDF Cosine Similarity","FAISS Vector Search",
         "SVD Dimensionality Reduction","TMDB Dataset",
         "OMDb Poster API","Python · Streamlit · scikit-learn"]
st.markdown(f"""
<div class="filmstrip-ticker">
  <div class="ticker-inner">{sep.join(items)*3}</div>
</div>""", unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero-outer">
  <div class="arc1"></div>
  <div class="arc2"></div>
  <div class="hero-eyebrow">
    ⬡ NEURAL ENGINE<span>◆</span>AI-POWERED<span>◆</span>DUAL-PHASE SEARCH ⬡
  </div>
  <div>
    <span class="hero-title-cine">CINE</span><span class="hero-title-match">MATCH</span>
  </div>
  <div class="hero-divider"></div>
  <div class="hero-subtitle">Discover &nbsp;·&nbsp; Explore &nbsp;·&nbsp; Experience Cinema</div>
</div>""", unsafe_allow_html=True)

# ── STATS ────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
for i, (col, num, lbl, clr, dl) in enumerate(zip(
    [s1,s2,s3,s4],
    ["26K+","2","~2ms","20+"],
    ["FILMS INDEXED","NEURAL ENGINES","SEARCH LATENCY","GENRES"],
    ["#E50914","#00e5ff","#bf00ff","#00ff9d"],
    ["0.0s","0.4s","0.8s","1.2s"]
)):
    with col:
        st.markdown(f"""
        <div class="stat-card sr" data-delay="{i*100}"
             style="--sc:{clr};--sd:{dl}">
            <span class="stat-number" style="color:{clr};text-shadow:0 0 18px {clr}">{num}</span>
            <div class="stat-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

# ============================================================
# SEARCH
# ============================================================
st.markdown('<div class="search-frame">', unsafe_allow_html=True)
st.markdown('<div class="search-label">QUERY FILM DATABASE</div>', unsafe_allow_html=True)
sc1, sc2 = st.columns([4, 1])
with sc1:
    selected = st.selectbox("Search:", [""] + all_titles, index=0,
                            label_visibility="collapsed")
with sc2:
    btn = st.button("▶ SEARCH", use_container_width=True, type="primary")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
for k, v in [('last_movie',''),('results',None),('source',None),('engine','')]:
    if k not in st.session_state:
        st.session_state[k] = v

if btn and selected:
    st.session_state.last_movie = selected

if st.session_state.last_movie:
    movie = st.session_state.last_movie

    with st.spinner("SCANNING NEURAL DATABASE..."):
        res, src, eng = None, None, ""
        if ok1 and sim1 is not None:
            m = df1[df1['title'].str.lower() == movie.lower()]
            if not m.empty:
                res, src = rec_p1(movie, n_recs, year_range, df1, sim1)
                eng = "phase1"
        if res is None and ok2:
            m = df2[df2['title'].str.lower() == movie.lower()]
            if not m.empty:
                #res, src = rec_p2(movie, n_recs, min_votes, year_range, df2, fi, fn)
                eng = "phase2"

    if res is None or src is None:
        st.error(f"⚠  '{movie}' was not found in the database.")
    else:
        # ── SOURCE HEADER ─────────────────────────────────
        st.markdown("""
        <div class="section-hdr sr" data-delay="0">
            <span class="section-hdr-tag">SELECTED</span>
            <span class="section-hdr-title">SOURCE FILM</span>
            <div class="section-hdr-line"></div>
        </div>""", unsafe_allow_html=True)

        # ── SOURCE PANEL ──────────────────────────────────
        st.markdown('<div class="source-panel sr" data-delay="60">', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 4])

        with c1:
            if eng == "phase2":
                p = poster_p2(src.get('poster_path', ''))
            else:
                try:
                    yr = str(int(src['release_year'])) if pd.notna(src['release_year']) else ""
                except:
                    yr = ""
                p = poster_p1(src['title'], yr)
            st.markdown(
                f"<div class='poster-container'>"
                f"<img src='{p}' alt='poster'/>"
                f"<div class='poster-glow'></div></div>",
                unsafe_allow_html=True)

        with c2:
            st.markdown(f"<div class='src-title'>{src['title']}</div>", unsafe_allow_html=True)

            # Rating
            try:
                rat    = float(src['weighted_rating'])
                filled = int(round(rat / 2))
                stars  = "★" * filled + "☆" * (5 - filled)
                st.markdown(
                    f"<div class='src-rating'>{stars}"
                    f"<span class='score'>{rat:.1f}</span>"
                    f"<span style='font-size:.85rem;color:rgba(255,215,0,.45)'>/10</span></div>",
                    unsafe_allow_html=True)
            except:
                pass

            # Year only — NO engine badge
            if eng == "phase2":
                try:
                    yr = str(src['release_date'])[:4]
                    st.markdown(f"<div class='src-meta'>📅 {yr}</div>", unsafe_allow_html=True)
                except:
                    pass
            else:
                try:
                    st.markdown(f"<div class='src-meta'>📅 {int(src['release_year'])}</div>",
                                unsafe_allow_html=True)
                except:
                    pass

            # Genres
            genres = src.get('genres_list', []) if eng == "phase2" else src.get('genres', [])
            if isinstance(genres, str):
                try:    genres = ast.literal_eval(genres)
                except: genres = []
            if genres and isinstance(genres, list):
                pills = "".join([f"<span class='genre-pill'>{g}</span>" for g in genres[:6]])
                st.markdown(f"<div class='genre-wrap'>{pills}</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # source-panel

        # ── RECS DIVIDER ──────────────────────────────────
        st.markdown(f"""
        <div class="recs-lead sr" data-delay="40">
            <div class="recs-lead-line"></div>
            <span class="recs-lead-count">{len(res)} FILMS MATCHED</span>
            <div class="recs-lead-line"
                 style="background:linear-gradient(270deg,rgba(229,9,20,.30),rgba(0,229,255,.10),transparent)">
            </div>
        </div>""", unsafe_allow_html=True)

        # ── RECOMMENDATION GRID ───────────────────────────
        # Each card gets a unique index used for stagger delay.
        # The scroll-reveal JS will also assign a random direction class.
        cols_per_row = 5
        card_idx     = 0

        for row_i in range(0, len(res), cols_per_row):
            chunk = res.iloc[row_i:row_i + cols_per_row]
            cols  = st.columns(cols_per_row)
            for col_i, (col, (_, mv)) in enumerate(zip(cols, chunk.iterrows())):
                with col:
                    r_html, b_html, m_html = "", "", ""
                    try:
                        r      = float(mv['weighted_rating'])
                        r_html = f"<div class='card-rating'>★ {r:.1f}</div>"
                        b_html = f"<div class='card-badge'>★ {r:.1f}</div>"
                    except:
                        pass
                    try:
                        sc     = float(mv.get('final_score', mv.get('similarity', 0)))
                        m_html = f"<div class='card-match'>◈ {sc:.0%} MATCH</div>"
                    except:
                        pass

                    # Stagger: rows cascade, columns offset within row
                    delay_ms = row_i * 80 + col_i * 65

                    st.markdown(
                        f"<div class='movie-card sr' data-delay='{delay_ms}'>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card-poster-wrap'>"
                        f"<img src='{mv['poster']}' alt='{str(mv['title'])[:30]}'/>"
                        f"</div>"
                        f"{b_html}"
                        f"<div class='card-bar'></div>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card-info'>"
                        f"<div class='card-title'>{str(mv['title'])[:28]}</div>"
                        f"{r_html}{m_html}"
                        f"</div>",
                        unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    card_idx += 1

# ============================================================
# BROWSE — Top Rated by Genre
# ============================================================
st.markdown("""
<div class="genre-banner sr" data-delay="0" style="margin-top:52px">
    <div>
        <div class="genre-banner-title">🎭 TOP RATED BY GENRE</div>
        <div class="genre-banner-sub">Curated from both databases · 26K+ titles</div>
    </div>
    <div class="genre-banner-line"></div>
    <div class="genre-banner-badge">◈ BROWSE MODE</div>
</div>""", unsafe_allow_html=True)

all_genres_set = set()
if ok1:
    for g in df1['genres']:
        if isinstance(g, list): all_genres_set.update(g)
if ok2:
    for g in df2['genres_list']:
        if isinstance(g, list): all_genres_set.update(g)

top_genres = sorted(all_genres_set)[:10]
tabs       = st.tabs(top_genres)

for tab, genre in zip(tabs, top_genres):
    with tab:
        combined_movies = []

        if ok1:
            p1g = df1[df1['genres'].apply(
                lambda x: genre in x if isinstance(x, list) else False
            )].nlargest(20, 'weighted_rating')
            for _, row in p1g.iterrows():
                try:    yr = str(int(row['release_year'])) if pd.notna(row['release_year']) else ""
                except: yr = ""
                combined_movies.append({
                    'title': row['title'], 'weighted_rating': row['weighted_rating'],
                    'year': yr, 'poster': poster_p1(row['title'], yr), 'source': 'P1'
                })

        if ok2:
            p2g = df2[df2['genres_list'].apply(
                lambda x: genre in x if isinstance(x, list) else False
            )].nlargest(20, 'weighted_rating')
            for _, row in p2g.iterrows():
                try:    yr = str(row['release_date'])[:4]
                except: yr = ""
                combined_movies.append({
                    'title': row['title'], 'weighted_rating': row['weighted_rating'],
                    'year': yr, 'poster': poster_p2(row.get('poster_path', '')), 'source': 'P2'
                })

        combined_df = pd.DataFrame(combined_movies)
        if len(combined_df) > 0:
            combined_df = combined_df.drop_duplicates(subset='title').nlargest(10, 'weighted_rating')
            cols = st.columns(5)
            for i, (_, mv) in enumerate(combined_df.iterrows()):
                with cols[i % 5]:
                    try:
                        r  = float(mv['weighted_rating'])
                        rh = f"<div class='card-rating'>★ {r:.1f}</div>"
                        bh = f"<div class='card-badge'>★ {r:.1f}</div>"
                    except:
                        rh = bh = ""
                    yh = f"<div class='card-year'>{mv['year']}</div>" if mv['year'] else ""

                    delay_ms = (i % 5) * 65

                    st.markdown(
                        f"<div class='movie-card sr' data-delay='{delay_ms}'>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card-poster-wrap'>"
                        f"<img src='{mv['poster']}' alt='{str(mv['title'])[:22]}'/>"
                        f"</div>"
                        f"{bh}"
                        f"<div class='card-bar'></div>",
                        unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='card-info'>"
                        f"<div class='card-title'>{str(mv['title'])[:22]}</div>"
                        f"{rh}{yh}"
                        f"</div>",
                        unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer sr" data-delay="0">
    <div class="footer-logo"> CINEMATCH </div>
    <div class="footer-stack">
        Python<span class="footer-sep">◆</span>Streamlit<span class="footer-sep">◆</span>
        FAISS<span class="footer-sep">◆</span>scikit-learn<span class="footer-sep">◆</span>TMDB
    </div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:.58rem;
                letter-spacing:3px;text-transform:uppercase;
                color:var(--text-dim);opacity:.4;margin-top:5px">
        PHASE I · TF-IDF COSINE &nbsp;|&nbsp; PHASE II · TF-IDF + SVD + FAISS
    </div>
</div>""", unsafe_allow_html=True)
