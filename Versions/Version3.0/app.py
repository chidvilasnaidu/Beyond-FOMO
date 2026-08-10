import os
import uuid
import html

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from fomo_agent import build_graph


load_dotenv()

st.set_page_config(
    page_title="Beyond FOMO",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
<style>
:root{
    --ink:#161d1b;
    --teal:#0E8C8F;
    --tan:#E8B98A;
    --peach:#FF9F5A;
    --mint:#EAF7F2;
    --glass:rgba(234,247,242,0.09);
    --glass-strong:rgba(234,247,242,0.14);
    --border:rgba(234,247,242,0.25);
    --navbar-h:54px;
    --footer-h:42px;
}

/* ---------- Global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html{
    font-size:18px;
}

html, body, [class*="css"]{
    font-family:'Inter', sans-serif;
    letter-spacing:-0.01em;
}

html, body, .stApp{
    min-height:100vh;
}

.stApp{
    background:
        radial-gradient(circle at 10% 6%, rgba(14,140,143,0.62), transparent 40%),
        radial-gradient(circle at 92% 10%, rgba(255,159,90,0.32), transparent 42%),
        radial-gradient(circle at 50% 105%, rgba(232,185,138,0.28), transparent 48%),
        linear-gradient(155deg, #12211f 0%, #16302e 45%, #103533 100%);
    color:var(--mint);
}

#MainMenu,
footer,
header,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"]{
    visibility:hidden !important;
    display:none !important;
}

.block-container{
    padding:0 1.25rem 0 !important;
    max-width:100% !important;
}

/* Transparent Streamlit wrappers */
div[data-testid="stAppViewContainer"],
div[data-testid="stMain"],
section.main,
div[data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlockBorderWrapper"]{
    background:transparent !important;
}


/* ---------- Shared glass ---------- */
.glass{
    background:var(--glass);
    border:1px solid var(--border);
    border-radius:22px;
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
    box-shadow:0 8px 32px rgba(0,0,0,0.30);
}


/* ---------- Navbar ---------- */
.navbar{
    position:fixed;
    top:0;
    left:0;
    right:0;
    height:var(--navbar-h);
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:0 1.5rem;
    background:rgba(22,29,27,0.60);
    border-bottom:1px solid var(--border);
    backdrop-filter:blur(16px);
    -webkit-backdrop-filter:blur(16px);
    z-index:999;
}

.navbar .brand{
    display:flex;
    align-items:center;
    gap:.5rem;
    font-family:'Space Grotesk',sans-serif;
    font-weight:700;
    font-size:1.05rem;
    color:var(--mint);
}

.navbar .brand .b-dot{
    width:9px;
    height:9px;
    border-radius:50%;
    background:var(--peach);
    box-shadow:0 0 10px var(--peach);
}

.navbar .links{
    display:flex;
    align-items:center;
    gap:1.6rem;
}

.navbar .links a{
    font-size:.86rem;
    color:rgba(234,247,242,0.68);
    text-decoration:none;
    font-weight:500;
}

.navbar .links a:hover{
    color:var(--peach);
}

.navbar .status{
    display:flex;
    align-items:center;
    gap:.4rem;
    font-size:.78rem;
    color:rgba(234,247,242,0.78);
    background:rgba(255,159,90,0.12);
    border:1px solid rgba(255,159,90,0.35);
    padding:.3rem .7rem;
    border-radius:999px;
}

.navbar .status .dot{
    width:6px;
    height:6px;
    border-radius:50%;
    background:#7CFFB2;
    box-shadow:0 0 8px #7CFFB2;
    animation:pulse-dot 1.6s ease-in-out infinite;
}

@keyframes pulse-dot{
    0%,100%{opacity:1}
    50%{opacity:.25}
}


/* ---------- Main top-level columns ---------- */
/* Only the first/main horizontal block gets the viewport layout.
   Nested blocks inside the form are explicitly excluded. */
div[data-testid="stHorizontalBlock"]:has(.title-block){
    box-sizing:border-box;
    height:calc(100vh - var(--navbar-h) - var(--footer-h) - .35rem);
    min-height:calc(100vh - var(--navbar-h) - var(--footer-h) - .35rem);
    margin-top:var(--navbar-h);
    margin-bottom:var(--footer-h);
    padding:1.15rem 0 1rem;
    gap:1.35rem;
}

div[data-testid="stHorizontalBlock"]:has(.title-block)
> div[data-testid="column"]{
    height:100%;
    min-height:0;
}

div[data-testid="stHorizontalBlock"]:has(.title-block)
> div[data-testid="column"]:first-child{
    overflow:hidden !important;
    min-height:0 !important;
}

div[data-testid="stHorizontalBlock"]:has(.title-block){
    overflow:hidden !important;
}

div[data-testid="stHorizontalBlock"]:has(.title-block)
> div[data-testid="column"]:last-child{
    overflow:hidden;
}


@keyframes titleGlow{
    0%,100%{
        filter:drop-shadow(0 5px 16px rgba(14,140,143,.20));
    }
    50%{
        filter:drop-shadow(0 7px 24px rgba(255,159,90,.30));
    }
}

/* ---------- Left side ---------- */
.title-block{
    padding:.2rem .25rem 1rem .25rem;
}

.title-block h1{
    position:relative;
    display:inline-block;
    font-family:'Space Grotesk',sans-serif;
    font-weight:800;
    font-size:3.55rem;
    line-height:.94;
    margin:0 0 1.15rem 0;
    letter-spacing:-.07em;
    font-style:italic;
    text-shadow:0 10px 30px rgba(0,0,0,.22);
}

.title-block h1::after{
    content:"";
    position:absolute;
    left:2px;
    bottom:-.42rem;
    width:78%;
    height:4px;
    border-radius:999px;
    background:linear-gradient(90deg,var(--peach),var(--teal),transparent);
    box-shadow:0 0 15px rgba(255,159,90,.30);
}

.title-block h1 span{
    background:linear-gradient(
        100deg,
        #FF9F5A 0%,
        #E8B98A 34%,
        #8CE7E8 72%,
        #EAF7F2 100%
    );
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
    filter:drop-shadow(0 5px 18px rgba(14,140,143,.28));
    animation:titleGlow 4s ease-in-out infinite;
}

.title-block p{
    max-width:95%;
    font-size:1.02rem;
    line-height:1.62;
    color:rgba(234,247,242,.88);
    margin:0;
}

.feature-list{
    height:calc(100% - 10.5rem);
    min-height:0;
    overflow-y:auto;
    overflow-x:hidden;
    padding:.15rem .3rem .5rem 0;
    display:flex;
    flex-direction:column;
    gap:.55rem;
    scrollbar-width:thin;
    scrollbar-color:rgba(140,231,232,.65) rgba(234,247,242,.06);
}

.feature-list::-webkit-scrollbar{
    width:8px;
}

.feature-list::-webkit-scrollbar-track{
    background:rgba(234,247,242,.045);
    border-radius:999px;
}

.feature-list::-webkit-scrollbar-thumb{
    background:linear-gradient(180deg,#8CE7E8,#FF9F5A);
    border-radius:999px;
    border:2px solid rgba(16,53,51,.72);
}

.feature-list::-webkit-scrollbar-thumb:hover{
    background:linear-gradient(180deg,#B9FFFF,#FFC48D);
}

.feat-row{
    padding:.68rem .82rem;
    display:flex;
    align-items:center;
    gap:.7rem;
    flex:0 0 auto;
    transition:transform .15s ease,border-color .15s ease;
}

.feat-row:hover{
    transform:translateX(3px);
    border-color:rgba(255,159,90,.55);
}

.feat-row .icon{
    flex:0 0 auto;
    width:36px;
    height:36px;
    border-radius:10px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:1.05rem;
    background:linear-gradient(135deg,rgba(14,140,143,.65),rgba(255,159,90,.32));
    border:1px solid var(--border);
}

.feat-row .txt h4{
    font-family:'Space Grotesk',sans-serif;
    font-size:1.02rem;
    margin:0 0 .14rem 0;
    color:var(--mint);
}

.feat-row .txt p{
    font-size:.86rem;
    line-height:1.45;
    margin:0;
    color:rgba(234,247,242,.64);
}


/* ---------- Right side chat ---------- */
.right-stack{
    display:flex;
    flex-direction:column;
    height:100%;
    min-height:0;
    gap:1.35rem;
}

.chat-card{
    flex:1 1 auto;
    min-height:0;
    height:100%;
    min-height:0;
    display:flex;
    flex-direction:column;
    padding:1.5rem 1.5rem 1.15rem 1.5rem;
}

.bot-stage{
    display:flex;
    align-items:center;
    justify-content:center;
    height:310px;
    flex:0 0 auto;
}

.mascot{
    animation:float 4.5s ease-in-out infinite;
    transform-origin:center;
}

@keyframes float{
    0%,100%{transform:translateY(0) rotate(-1deg)}
    50%{transform:translateY(-10px) rotate(1deg)}
}

.mascot .antenna-dot{
    animation:blink-fast 2.2s ease-in-out infinite;
}

.mascot .eye{
    animation:blink 4.8s ease-in-out infinite;
    transform-origin:center;
}

@keyframes blink{
    0%,92%,100%{transform:scaleY(1)}
    95%{transform:scaleY(.12)}
}

@keyframes blink-fast{
    0%,100%{opacity:1;filter:drop-shadow(0 0 8px var(--peach))}
    50%{opacity:.45;filter:drop-shadow(0 0 1px var(--peach))}
}

.mascot .visor{
    animation:glow 3.2s ease-in-out infinite;
}

@keyframes glow{
    0%,100%{filter:drop-shadow(0 0 8px rgba(14,140,143,.7))}
    50%{filter:drop-shadow(0 0 18px rgba(14,140,143,1))}
}

.mascot .mic-wave{
    animation:wave 1.8s ease-in-out infinite;
    transform-origin:center;
}

@keyframes wave{
    0%,100%{transform:scaleY(.6);opacity:.5}
    50%{transform:scaleY(1);opacity:1}
}

.bot-caption{
    text-align:center;
    font-size:.94rem;
    color:rgba(234,247,242,.66);
    margin:-.15rem 0 .45rem;
    flex:0 0 auto;
}

.chat-scroll{
    flex:1 1 auto;
    min-height:0;
    overflow-y:auto;
    padding:.15rem .3rem .35rem .1rem;
    display:flex;
    flex-direction:column;
    scrollbar-width:thin;
}

.bubble-row{
    display:flex;
    margin:.38rem 0;
}

.bubble-row.user{
    justify-content:flex-end;
}

.bubble{
    max-width:78%;
    padding:.65rem .9rem;
    border-radius:17px;
    font-size:1rem;
    line-height:1.58;
    white-space:pre-wrap;
    overflow-wrap:anywhere;
}

.bubble.assistant{
    background:var(--glass-strong);
    border:1px solid var(--border);
    border-top-left-radius:4px;
    color:var(--mint);
}

.bubble.user{
    background:linear-gradient(135deg,rgba(14,140,143,.9),rgba(232,185,138,.6));
    border:1px solid rgba(255,159,90,.45);
    border-top-right-radius:4px;
    color:#0d1817;
    font-weight:500;
}



/* ---------- Right-side real Streamlit container ---------- */
div[data-testid="stVerticalBlock"].st-key-right_chat{
    height:100% !important;
    min-height:0 !important;
    min-height:0 !important;
    display:flex !important;
    flex-direction:column !important;
    gap:1.1rem !important;
    overflow:visible !important;
}

div[data-testid="stVerticalBlock"].st-key-right_chat
> div[data-testid="stVerticalBlockBorderWrapper"]{
    height:100% !important;
    min-height:0 !important;
}

div[data-testid="stVerticalBlock"].st-key-right_chat
> div[data-testid="stVerticalBlockBorderWrapper"]{
    height:100% !important;
    min-height:0 !important;
}

div[data-testid="stVerticalBlock"].st-key-right_chat
div.element-container:has(.chat-card){
    flex:1 1 auto !important;
    min-height:0 !important;
    display:flex !important;
    flex-direction:column !important;
    overflow:hidden !important;
}

div[data-testid="stVerticalBlock"].st-key-right_chat
div.element-container:has(.chat-card) > div{
    flex:1 1 auto !important;
    min-height:0 !important;
    display:flex !important;
    flex-direction:column !important;
}

div[data-testid="stVerticalBlock"].st-key-right_chat
div.element-container:has(div[data-testid="stForm"]),
div[data-testid="stVerticalBlock"].st-key-right_chat
div.element-container:has(.composer-meta){
    flex:0 0 auto !important;
    min-height:0 !important;
    overflow:visible !important;
}

/* ---------- Composer ---------- */
/* It is intentionally OUTSIDE the chat card. */
div[data-testid="stForm"]{
    margin-top:0 !important;
    margin-bottom:0 !important;
    background:linear-gradient(
        145deg,
        rgba(234,247,242,.15),
        rgba(14,140,143,.08)
    ) !important;
    border:1px solid rgba(140,231,232,.28) !important;
    border-radius:24px !important;
    padding:.9rem 1rem .8rem !important;
    min-height:126px !important;
    flex:0 0 auto !important;
    backdrop-filter:blur(16px);
    -webkit-backdrop-filter:blur(16px);
    box-shadow:0 10px 30px rgba(0,0,0,.28);
}

div[data-testid="stForm"] .stTextInput{
    width:100%;
}

div[data-testid="stForm"] .stTextInput > div{
    background:
        linear-gradient(145deg,rgba(5,24,24,.72),rgba(17,49,47,.52))
        !important;
    border:1px solid rgba(140,231,232,.30) !important;
    border-radius:17px !important;
    min-height:64px;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.07),
        inset 0 -8px 20px rgba(0,0,0,.08),
        0 8px 22px rgba(0,0,0,.14) !important;
    transition:border-color .2s ease,box-shadow .2s ease;
}

div[data-testid="stForm"] .stTextInput input,
div[data-testid="stForm"] .stTextInput input:focus{
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
    color:var(--mint) !important;
    -webkit-text-fill-color:var(--mint) !important;
    caret-color:var(--peach) !important;
    font-size:1rem !important;
    padding:.88rem 1rem !important;
}

div[data-testid="stForm"] .stTextInput > div:focus-within{
    border-color:rgba(140,231,232,.72) !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.09),
        0 0 0 3px rgba(14,140,143,.10),
        0 10px 30px rgba(0,0,0,.18) !important;
}

div[data-testid="stForm"] .stTextInput input::placeholder{
    color:rgba(234,247,242,.52) !important;
    -webkit-text-fill-color:rgba(234,247,242,.52) !important;
}

div[data-testid="stForm"] div[data-testid="stHorizontalBlock"]{
    height:auto !important;
    min-height:0 !important;
    margin:0 !important;
    padding:0 !important;
    gap:.35rem !important;
    align-items:center !important;
}

div[data-testid="stForm"] button{
    min-width:54px !important;
    min-height:54px !important;
    border:1px solid rgba(234,247,242,.20) !important;
    background:rgba(234,247,242,.07) !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.08),
        0 8px 22px rgba(0,0,0,.18) !important;
    color:rgba(234,247,242,.90) !important;
    font-size:1.45rem !important;
    line-height:1 !important;
    padding:.25rem .65rem !important;
    border-radius:16px !important;
    transition:
        transform .18s ease,
        background .18s ease,
        border-color .18s ease,
        box-shadow .18s ease !important;
}

div[data-testid="stForm"] button:hover{
    transform:translateY(-2px);
    color:var(--mint) !important;
    background:rgba(234,247,242,.13) !important;
    border-color:rgba(255,159,90,.42) !important;
    box-shadow:
        0 10px 28px rgba(0,0,0,.25),
        0 0 18px rgba(255,159,90,.10) !important;
}

div[data-testid="stForm"] div[data-testid="column"]:first-child button{
    font-size:1.25rem !important;
}

div[data-testid="stForm"] div[data-testid="column"]:nth-child(3) button{
    min-width:76px !important;
    min-height:70px !important;
    width:76px !important;
    color:#8CE7E8 !important;
    border-color:rgba(140,231,232,.38) !important;
    background:linear-gradient(
        145deg,
        rgba(14,140,143,.22),
        rgba(234,247,242,.06)
    ) !important;
    font-size:2.45rem !important;
    text-shadow:0 0 18px rgba(140,231,232,.48);
}

div[data-testid="stForm"] div[data-testid="column"]:nth-child(3) button:hover{
    border-color:rgba(140,231,232,.70) !important;
    box-shadow:
        0 10px 30px rgba(0,0,0,.25),
        0 0 24px rgba(14,140,143,.28) !important;
}

div[data-testid="stForm"] div[data-testid="column"]:last-child button{
    min-width:82px !important;
    min-height:70px !important;
    width:82px !important;
    border-color:rgba(255,159,90,.60) !important;
    background:linear-gradient(145deg,#FFB06E,#FF9F5A) !important;
    color:#1a1108 !important;
    font-size:2.65rem !important;
    font-weight:800;
    border-radius:17px !important;
    box-shadow:
        0 8px 26px rgba(255,159,90,.20),
        inset 0 1px 0 rgba(255,255,255,.35) !important;
}


/* Streamlit puts form-button labels inside nested elements.
   These rules force the microphone and arrow to stay large. */
div[data-testid="stForm"] button p,
div[data-testid="stForm"] button span,
div[data-testid="stForm"] button div{
    line-height:1 !important;
}

div[data-testid="stForm"] div[data-testid="column"]:nth-child(3) button p,
div[data-testid="stForm"] div[data-testid="column"]:nth-child(3) button span{
    font-size:2.45rem !important;
    transform:scale(1.08);
}

div[data-testid="stForm"] div[data-testid="column"]:last-child button p,
div[data-testid="stForm"] div[data-testid="column"]:last-child button span{
    font-size:2.65rem !important;
    transform:scale(1.08);
}

div[data-testid="stForm"] div[data-testid="column"]:last-child button:hover{
    background:linear-gradient(145deg,#FFD09D,#E8B98A) !important;
    box-shadow:
        0 12px 32px rgba(255,159,90,.30),
        0 0 24px rgba(255,159,90,.18) !important;
}

.composer-meta{
    display:flex;
    align-items:center;
    justify-content:space-between;
    font-size:.76rem;
    color:rgba(234,247,242,.48);
    margin:.3rem .2rem 0;
}

.composer-meta .badge{
    background:rgba(234,247,242,.07);
    border:1px solid var(--border);
    border-radius:999px;
    padding:.12rem .5rem;
    color:rgba(234,247,242,.58);
}


/* ---------- Footer ---------- */
.footer{
    position:fixed;
    left:0;
    right:0;
    bottom:0;
    height:var(--footer-h);
    display:flex;
    align-items:center;
    justify-content:center;
    gap:.35rem;
    background:rgba(22,29,27,.60);
    border-top:1px solid var(--border);
    backdrop-filter:blur(16px);
    -webkit-backdrop-filter:blur(16px);
    font-size:.78rem;
    color:rgba(234,247,242,.60);
    z-index:999;
}

.footer b{
    color:var(--peach);
    font-weight:600;
}


/* ---------- Responsive ---------- */
@media (max-width:900px){
    .navbar .links{
        display:none;
    }

    .navbar{
        padding:0 1rem;
    }

    div[data-testid="stHorizontalBlock"]:has(.title-block){
        min-height:auto;
        padding-top:1rem;
        padding-bottom:1rem;
    }

    div[data-testid="stHorizontalBlock"]:has(.title-block)
    > div[data-testid="column"]{
        min-height:auto;
        overflow:visible;
    }

    .feature-list{
        height:auto;
        max-height:360px;
        overflow-y:auto;
    }

    .right-stack{
        height:auto;
        min-height:760px;
        gap:1rem;
    }

    .chat-card{
        min-height:590px;
    }

    .bot-stage{
        height:240px;
    }

    .title-block h1{
        font-size:2.65rem;
    }

    .footer{
        height:36px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Navbar
# ============================================================
st.markdown(
    '<div class="navbar">'
    '<div class="brand"><span class="b-dot"></span> 🛰️ BEYOND FOMO</div>'
    '<div class="links">'
    '<a href="#features">Features</a>'
    '<a href="#sectors">Sectors</a>'
    '<a href="#how-it-works">How it works</a>'
    '</div>'
    '<div class="status"><span class="dot"></span> Guardian online</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Mascot
# ============================================================
MASCOT_SVG = """
<svg class="mascot" width="205" height="205" viewBox="0 0 220 220"
     xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="110" cy="196" rx="55" ry="10" fill="#0d1817" opacity="0.35"/>
  <line x1="110" y1="26" x2="110" y2="6" stroke="#EAF7F2"
        stroke-width="3" stroke-linecap="round"/>
  <circle class="antenna-dot" cx="110" cy="6" r="6" fill="#FF9F5A"/>
  <circle cx="50" cy="98" r="10" fill="#0E8C8F"
          stroke="#EAF7F2" stroke-width="2"/>
  <circle cx="170" cy="98" r="10" fill="#0E8C8F"
          stroke="#EAF7F2" stroke-width="2"/>
  <rect x="45" y="34" width="130" height="110" rx="38"
        fill="#EAF7F2" stroke="#161d1b" stroke-width="3"/>
  <rect x="45" y="34" width="130" height="110" rx="38"
        fill="url(#headGrad)" opacity="0.5"/>
  <rect class="visor" x="64" y="60" width="92" height="52"
        rx="22" fill="#0E8C8F"/>
  <ellipse class="eye" cx="92" cy="86" rx="8" ry="10" fill="#EAF7F2"/>
  <ellipse class="eye" cx="128" cy="86" rx="8" ry="10" fill="#EAF7F2"/>
  <path d="M96 100 Q110 108 124 100" stroke="#EAF7F2"
        stroke-width="3" fill="none" stroke-linecap="round"/>
  <path d="M85 144 L110 158 L135 144 L128 168 L92 168 Z"
        fill="#FF9F5A" stroke="#161d1b" stroke-width="2"/>
  <path d="M35 210 Q40 158 90 152 L130 152 Q180 158 185 210 Z"
        fill="#0E8C8F" stroke="#161d1b" stroke-width="3"/>
  <path d="M65 152 Q110 178 155 152" stroke="#E8B98A"
        stroke-width="4" fill="none" opacity="0.55"/>
  <g transform="translate(150,150) rotate(18)">
    <rect x="-7" y="-28" width="14" height="30" rx="7" fill="#161d1b"/>
    <rect class="mic-wave" x="-4" y="-24" width="8" height="20"
          rx="4" fill="#FF9F5A"/>
    <line x1="0" y1="2" x2="0" y2="26" stroke="#161d1b"
          stroke-width="3" stroke-linecap="round"/>
  </g>
  <defs>
    <linearGradient id="headGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#FF9F5A"/>
      <stop offset="100%" stop-color="#0E8C8F"/>
    </linearGradient>
  </defs>
</svg>
"""


FEATURES = [
    ("📡", "Live Pulse", "Fresh news & events, not stale memory."),
    ("🔎", "Deep Lookup", "Evergreen answers — rankings, comparisons, how-tos."),
    ("🗂️", "Category Feeds", "Jobs, visas, scholarships, AI, business on demand."),
    ("🚦", "Priority Signal", "Critical, Important, Can Wait, or Ignore — tagged."),
    ("🎓", "Opportunity Radar", "Scholarships, internships, visas, deadlines surfaced."),
    ("🧼", "Clean Read", "Source clutter and unverified claims filtered."),
    ("🌐", "Multilingual", "Updates translated into your language."),
    ("🌦️", "Quick Utilities", "Weather checks & one-line summaries, on request."),
]


# ============================================================
# Session state
# ============================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


def content_to_text(content):
    """Convert common LangChain content formats into displayable text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


# ============================================================
# Main two-column layout
# ============================================================
left_col, right_col = st.columns([3, 7], gap="medium")


with left_col:
    st.markdown(
        '<div id="features" class="title-block">'
        '<h1>Beyond <span>FOMO</span></h1>'
        '<p>Every feed feeds the fear of missing out — quitting just swaps it '
        'for the fear of missing what actually matters. This is the middle '
        'path: agents that filter the noise and hand you the signal.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    cards = "".join(
        f'<div class="glass feat-row">'
        f'<div class="icon">{html.escape(icon)}</div>'
        f'<div class="txt"><h4>{html.escape(title)}</h4>'
        f'<p>{html.escape(desc)}</p></div></div>'
        for icon, title, desc in FEATURES
    )

    st.markdown(
        f'<div id="sectors" class="feature-list">{cards}</div>',
        unsafe_allow_html=True,
    )


with right_col:
    api_key_ready = bool(os.environ.get("GOOGLE_API_KEY"))

    # One real Streamlit container owns the entire right side.
    # CSS can therefore make the chat card flexible while keeping
    # the composer at the bottom and fully visible.
    with st.container(key="right_chat"):
        chat_placeholder = st.empty()

        # ----------------------------------------------------
        # Form is intentionally OUTSIDE the chat card.
        # This prevents the input from overlapping the bot area.
        # ----------------------------------------------------
        with st.form("chat_form", clear_on_submit=True, border=False):
            user_text = st.text_input(
                "message",
                placeholder="Ask me anything — jobs, visas, AI, business, weather…",
                label_visibility="collapsed",
            )

            c_attach, c_spacer, c_mic, c_send = st.columns([1, 8, 1, 1.3])

            with c_attach:
                st.form_submit_button("＋", help="Attach — coming soon")

            with c_mic:
                st.form_submit_button("🎙️", help="Voice search — coming soon")

            with c_send:
                sent = st.form_submit_button("➡️", type="primary")

        st.markdown(
            '<div class="composer-meta">'
            '<span>🎙️ Voice input &amp; spoken responses are on the way</span>'
            '<span class="badge">FOMO Guardian</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        if not api_key_ready:
            st.warning(
                "No GOOGLE_API_KEY found in .env — add it, then rerun.",
                icon="🔑",
            )

        # --------------------------------------------------------
        # Process current turn
        # --------------------------------------------------------
        if sent and user_text.strip():
            clean_user_text = user_text.strip()
            st.session_state.history.append(("user", clean_user_text))

            if not api_key_ready:
                st.session_state.history.append(
                    (
                        "assistant",
                        "I can't reach the model yet — add GOOGLE_API_KEY "
                        "to your .env file. Anything else I can help with?",
                    )
                )
            else:
                try:
                    graph = get_graph()

                    with st.spinner("Filtering the noise…"):
                        # IMPORTANT:
                        # Do NOT send preferences={} on every turn.
                        # The checkpointer keeps preferences for this thread.
                        result = graph.invoke(
                            {
                                "messages": [
                                    HumanMessage(content=clean_user_text)
                                ]
                            },
                            {
                                "configurable": {
                                    "thread_id": st.session_state.thread_id
                                }
                            },
                        )

                    reply = content_to_text(result["messages"][-1].content)

                except Exception as e:
                    reply = (
                        "Something broke on my end. "
                        f"Please try again.\n\nError: {e}"
                    )

                st.session_state.history.append(("assistant", reply))

        # --------------------------------------------------------
        # Render chat card
        # --------------------------------------------------------
        bubbles = ""

        if not st.session_state.history:
            bubbles += (
                '<div class="bubble-row">'
                '<div class="bubble assistant">'
                "Ask me for today's jobs, visa, AI, or business updates, "
                "or tell me what to focus on. Anything else I can help with?"
                "</div></div>"
            )

        for role, text in st.session_state.history:
            row_cls = "user" if role == "user" else "assistant"
            safe_text = html.escape(str(text))
            bubbles += (
                f'<div class="bubble-row {row_cls}">'
                f'<div class="bubble {row_cls}">{safe_text}</div>'
                "</div>"
            )

        chat_card_html = (
            '<div id="how-it-works" class="glass chat-card">'
            f'<div class="bot-stage">{MASCOT_SVG}</div>'
            '<div class="bot-caption">'
            "Hey — I'm your FOMO Guardian. Ask me anything below."
            "</div>"
            f'<div class="chat-scroll">{bubbles}</div>'
            "</div>"
        )

        # Put the card in the placeholder after processing the turn.
        chat_placeholder.markdown(
            chat_card_html,
            unsafe_allow_html=True,
        )

        if sent and user_text.strip():
            st.rerun()


# ============================================================
# Footer
# ============================================================
st.markdown(
    '<div class="footer">🛰️ <b>Beyond FOMO</b> — '
    'filtering the noise, not the signal</div>',
    unsafe_allow_html=True,
)
