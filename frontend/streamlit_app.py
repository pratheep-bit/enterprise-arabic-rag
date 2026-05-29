"""
streamlit_app.py — Arabic Document Q&A Frontend
Clean rewrite — Supports English (default) and Arabic UI.
"""

import html
import os
import time
import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# ──────────────────────────────────────────────────────────────
# Translations & Locale
# ──────────────────────────────────────────────────────────────
T = {
    "en": {
        "page_title": "Arabic Q&A · RAG",
        "header_title": "📜 Arabic Document Q&A System",
        "header_sub": "Retrieval-Augmented Generation",
        "sidebar_docs": "📁 Documents",
        "upload_label": "Upload PDF",
        "upload_btn": "📤 Process Document",
        "processing": "Processing...",
        "upload_success": "✅ Successfully processed",
        "pages": "pages",
        "chunks": "chunks",
        "no_docs": "No documents uploaded yet.",
        "health": "🔧 System Health",
        "status": "Status",
        "model": "Model",
        "docs_count": "Documents",
        "online": "🟢 Online",
        "offline": "🔴 Offline",
        "partial": "🟡 Degraded",
        "q_label": "Question",
        "q_placeholder": "Type your question here (Arabic or English)...",
        "ask_btn": "🔍 Ask Question",
        "top_k": "Results",
        "doc_filter": "Filter by doc",
        "all_docs": "All Documents",
        "empty_q": "⚠️ Please enter a question first.",
        "searching": "🔄 Searching & Answering...",
        "dialect_notice": "🗣️ Gulf dialect detected — Normalized to MSA:",
        "no_answer": "⚠️ No answer found.",
        "conf_hi": "● High Confidence",
        "conf_mid": "◐ Medium Confidence",
        "conf_lo": "○ Low Confidence",
        "translate_btn": "🌐 Translate Answer to English",
        "translation_label": "English Translation",
        "confidence_bar": "Confidence Score",
        "sources_label": "Sources",
        "page_str": "p.",
        "view_json": "🔍 View full JSON",
        "footer": "ARABIC Q&A · RAG v1.0 · FastAPI · LangChain · ChromaDB · Streamlit",
        "change_lang": "🌍 التبديل إلى العربية",
        "lang_code": "ar",
        "dir": "ltr",
        "align": "left"
    },
    "ar": {
        "page_title": "نظام الإجابة على الأسئلة",
        "header_title": "📜 نظام الإجابة على الأسئلة العربية",
        "header_sub": "Arabic Document Q&A — Retrieval-Augmented Generation",
        "sidebar_docs": "📁 المستندات",
        "upload_label": "ارفع ملف PDF",
        "upload_btn": "📤 معالجة المستند",
        "processing": "جاري المعالجة…",
        "upload_success": "✅ تم التحميل بنجاح",
        "pages": "صفحة",
        "chunks": "جزء",
        "no_docs": "لم يتم رفع أي مستندات بعد.",
        "health": "🔧 الحالة",
        "status": "الحالة",
        "model": "النموذج",
        "docs_count": "المستندات",
        "online": "🟢 متصل",
        "offline": "🔴 غير متصل",
        "partial": "🟡 جزئي",
        "q_label": "السؤال",
        "q_placeholder": "اكتب سؤالك باللغة العربية (فصحى أو عامية خليجية)…",
        "ask_btn": "🔍 ابحث عن الإجابة",
        "top_k": "نتائج",
        "doc_filter": "مستند",
        "all_docs": "كل المستندات",
        "empty_q": "⚠️ يرجى كتابة سؤال أولاً.",
        "searching": "🔄 جاري البحث والإجابة…",
        "dialect_notice": "🗣️ تم اكتشاف لهجة خليجية — السؤال بالفصحى:",
        "no_answer": "⚠️ لم يتم العثور على إجابة.",
        "conf_hi": "● ثقة عالية",
        "conf_mid": "◐ ثقة متوسطة",
        "conf_lo": "○ ثقة منخفضة",
        "translate_btn": "🌐 Translate to English",
        "translation_label": "English Translation",
        "confidence_bar": "درجة الثقة",
        "sources_label": "المراجع",
        "page_str": "ص",
        "view_json": "🔍 عرض JSON الكامل",
        "footer": "ARABIC Q&A · RAG v1.0 · FastAPI · LangChain · ChromaDB · Streamlit",
        "change_lang": "🌍 Change to English",
        "lang_code": "en",
        "dir": "rtl",
        "align": "right"
    }
}

# ──────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arabic Q&A · RAG",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────
_DEFAULTS = {
    "ui_lang":        "en",   # "en" or "ar"
    "qa_result":      None,   # full API response dict from /ask
    "qa_elapsed":     0.0,    # seconds taken
    "translation":    None,   # string or "" after translate
    "docs_cache":     None,   # list from /documents
    "upload_ok":      None,   # success message after PDF upload
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

L = T[st.session_state.ui_lang]

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def api(method: str, endpoint: str, **kwargs) -> dict:
    """Call backend. Returns {success, data} or {success:False, error}."""
    try:
        r = getattr(requests, method)(
            f"{API_BASE_URL}{endpoint}", timeout=120, **kwargs
        )
        if r.status_code == 200:
            return {"success": True, "data": r.json()}
        ct = r.headers.get("content-type", "")
        body = r.json() if "application/json" in ct else {"detail": r.text}
        return {"success": False, "error": body}
    except requests.ConnectionError:
        err = f"لا يمكن الاتصال بالخادم ({API_BASE_URL})" if st.session_state.ui_lang == "ar" else f"Cannot connect to server ({API_BASE_URL})"
        return {"success": False, "error": {"detail": err}}
    except requests.Timeout:
        err = "انتهت مهلة الطلب — حاول مرة أخرى" if st.session_state.ui_lang == "ar" else "Request timeout — please try again"
        return {"success": False, "error": {"detail": err}}
    except Exception as e:
        return {"success": False, "error": {"detail": "Request failed"}}

def err_msg(e) -> str:
    if isinstance(e, str):    return e
    if isinstance(e, dict):
        d = e.get("detail", "")
        fallback = "خطأ" if st.session_state.ui_lang == "ar" else "Error"
        return d.get("detail", fallback) if isinstance(d, dict) else str(d)
    return "خطأ غير معروف" if st.session_state.ui_lang == "ar" else "Unknown error"


def esc(value) -> str:
    """Escape dynamic values before rendering inside raw HTML blocks."""
    return html.escape(str(value), quote=True)

def docs_list() -> list:
    """Return cached document list, fetch if missing."""
    if st.session_state.docs_cache is None:
        r = api("get", "/documents")
        st.session_state.docs_cache = (
            r["data"].get("documents", []) if r["success"] else []
        )
    return st.session_state.docs_cache or []

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────
css_dir = L["dir"]
css_align = L["align"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

:root {{
  --ink:      #1a1a2e;
  --ink2:     #2d2d44;
  --paper:    #faf8f4;
  --paper2:   #f2ede4;
  --paper3:   #e8e0d0;
  --gold:     #b8860b;
  --gold-l:   #d4a017;
  --teal:     #0d7377;
  --teal-l:   #14a085;
  --red:      #c0392b;
  --muted:    #6b6570;
  --border:   #d5cec2;
  --radius:   10px;
  --font-ar:  'IBM Plex Sans Arabic', 'Amiri', sans-serif;
  --font-en:  'IBM Plex Mono', monospace;
  --font-body:'Lora', Georgia, serif;
}}

/* ── Shell ── */
.stApp {{ background: var(--paper) !important; }}
.main .block-container {{
  padding: 1.25rem 2.5rem 4rem !important;
  max-width: 1280px !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
  background: var(--ink) !important;
  border-right: none !important;
}}
[data-testid="stSidebar"] * {{ color: #d0cec8 !important; }}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
  color: var(--gold-l) !important;
  font-family: var(--font-en) !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}}
[data-testid="stSidebar"] hr {{
  border-color: #333350 !important;
  margin: 0.9rem 0 !important;
}}

/* ── All text ── */
p, span, div, label, li {{
  font-family: var(--font-ar) !important;
  color: var(--ink) !important;
}}

/* ── Fix Material Icons ── */
.stIconMaterial, [class*="material-symbols"], [data-testid="stIconMaterial"] {{
  font-family: "Material Symbols Rounded" !important;
}}

/* ── Global RTL/LTR based on language ── */
.bidi-box {{
    direction: {css_dir};
    text-align: {css_align};
}}

/* ── Widgets on light bg ── */
.stTextArea textarea {{
  background: #fff !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--ink) !important;
  font-family: var(--font-ar) !important;
  font-size: 1.05rem !important;
  direction: {css_dir};
  text-align: {css_align};
  padding: 0.9rem 1rem !important;
  transition: border-color .2s;
}}
.stTextArea textarea:focus {{
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px rgba(13,115,119,.1) !important;
}}
.stSelectbox > div > div {{
  background: #fff !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--ink) !important;
}}
[data-testid="stFileUploader"] {{
  background: #fff !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: var(--radius) !important;
}}

/* Ensure header and hamburger icon are visible */
[data-testid="stHeader"] {{
  background: transparent !important;
}}
[data-testid="collapsedControl"], [data-testid="stHeader"] button {{
  color: var(--ink) !important;
}}

/* ── Sidebar widgets (dark) ── */
[data-testid="stSidebar"] .stSelectbox > div > div {{
  background: var(--ink2) !important;
  border: 1px solid #444 !important;
  color: #d0cec8 !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
  background: var(--ink2) !important;
  border: 1.5px dashed #444 !important;
}}
/* Sidebar hamburger/close icon color */
[data-testid="stSidebar"] button {{
  color: #d0cec8 !important;
}}

/* ── Primary button ── */
.stButton > button {{
  background: var(--teal) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: var(--font-ar) !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  padding: 0.55rem 1.6rem !important;
  transition: background .2s, transform .15s !important;
  letter-spacing: 0.01em;
}}
.stButton > button:hover {{
  background: var(--teal-l) !important;
  transform: translateY(-1px) !important;
}}

/* ── Progress bar ── */
.stProgress > div > div > div {{
  background: linear-gradient(90deg, var(--teal), var(--gold)) !important;
}}
.stProgress > div > div {{ background: var(--paper3) !important; }}

/* ── Alerts ── */
.stAlert {{ border-radius: var(--radius) !important; }}

/* ── Expander ── */
.streamlit-expanderHeader {{
  background: var(--paper2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-family: var(--font-en) !important;
  font-size: 0.78rem !important;
  color: var(--muted) !important;
}}

/* ── Custom components ── */
.page-header {{
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  border-bottom: 2px solid var(--ink);
  padding-bottom: 1rem;
  margin-bottom: 1.75rem;
  direction: {css_dir};
  flex-direction: {"row-reverse" if css_dir == "rtl" else "row"};
}}
.page-header-title {{
  font-family: var(--font-ar) !important;
  font-size: 1.65rem;
  font-weight: 700;
  color: var(--ink) !important;
  line-height: 1.2;
}}
.page-header-sub {{
  font-family: var(--font-en) !important;
  font-size: 0.72rem;
  color: var(--muted) !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 0.3rem;
}}
.page-header-logo {{
  font-family: var(--font-en) !important;
  font-size: 0.68rem;
  color: var(--muted) !important;
  letter-spacing: 0.12em;
  text-align: {css_align};
  white-space: nowrap;
}}

/* answer card */
.answer-wrap {{
  background: #fff;
  border: 1.5px solid var(--border);
  border-radius: 12px;
  padding: 1.75rem 2rem;
  margin: 1rem 0 0.75rem;
  box-shadow: 0 2px 12px rgba(26,26,46,.06);
  direction: rtl; /* Arabic answer always RTL */
}}
.answer-meta-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
  padding-bottom: 0.85rem;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
  gap: 0.5rem;
  direction: {css_dir};
  flex-direction: {"row-reverse" if css_dir == "rtl" else "row"};
}}
.answer-body {{
  font-family: var(--font-ar) !important;
  font-size: 1.15rem !important;
  line-height: 2.1 !important;
  color: var(--ink) !important;
  direction: rtl; /* answer is Arabic */
  text-align: right;
}}
.badge {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.28rem 0.85rem;
  border-radius: 20px;
  font-family: var(--font-ar) !important;
  font-size: 0.82rem;
  font-weight: 600;
}}
.badge-hi  {{ background:#e8f5e9; color:#1b5e20 !important; border:1px solid #a5d6a7; }}
.badge-mid {{ background:#fff8e1; color:#6d4c00 !important; border:1px solid #ffe082; }}
.badge-lo  {{ background:#fce4ec; color:#880e4f !important; border:1px solid #f48fb1; }}
.timing {{
  font-family: var(--font-en) !important;
  font-size: 0.72rem;
  color: var(--muted) !important;
  background: var(--paper2);
  border: 1px solid var(--border);
  padding: 0.22rem 0.6rem;
  border-radius: 4px;
}}

/* no-answer */
.no-answer {{
  background: #fff8f8;
  border: 1.5px solid #f5c6c6;
  border-radius: 10px;
  padding: 1rem 1.25rem;
  direction: {css_dir};
  text-align: {css_align};
  font-family: var(--font-ar) !important;
  color: var(--red) !important;
  font-size: 0.95rem;
}}

/* translation */
.translation-box {{
  background: var(--paper2);
  border: 1.5px solid var(--border);
  border-left: 4px solid var(--teal);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  margin-top: 0.6rem;
  font-family: var(--font-body) !important;
  font-size: 0.98rem;
  color: var(--ink2) !important;
  line-height: 1.8;
  direction: ltr; /* translation always LTR */
  text-align: left;
}}
.translation-box .tl-label {{
  font-family: var(--font-en) !important;
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--teal) !important;
  display: block;
  margin-bottom: 0.5rem;
}}

/* JSON view */
[data-testid="stJson"] {{
  background: var(--ink) !important;
  padding: 1rem !important;
  border-radius: 8px !important;
  direction: ltr !important;
  text-align: left !important;
}}
[data-testid="stJson"] * {{
  color: #e2e8f0 !important;
  font-family: var(--font-en) !important;
}}

/* dialect notice */
.dialect-box {{
  background: #e8f4f8;
  border: 1px solid #b3d9e8;
  border-radius: 8px;
  padding: 0.65rem 1rem;
  direction: {css_dir};
  text-align: {css_align};
  font-family: var(--font-ar) !important;
  font-size: 0.88rem;
  color: #0a4d68 !important;
  margin-bottom: 0.85rem;
}}

/* source cards */
.src-card {{
  background: var(--paper2);
  border: 1px solid var(--border);
  border-right: 4px solid var(--gold);
  border-radius: 8px;
  padding: 0.9rem 1.1rem;
  margin: 0.5rem 0;
  direction: rtl; /* source text is Arabic */
  text-align: right;
}}
.src-hdr {{
  font-family: var(--font-en) !important;
  font-size: 0.75rem;
  color: var(--gold) !important;
  font-weight: 600;
  letter-spacing: 0.04em;
  margin-bottom: 0.4rem;
}}
.src-text {{
  font-family: var(--font-ar) !important;
  font-size: 0.9rem;
  color: var(--ink2) !important;
  line-height: 1.85;
  font-style: italic;
}}

/* section label */
.sec-lbl {{
  font-family: var(--font-en) !important;
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted) !important;
  display: block;
  margin: 1.25rem 0 0.5rem;
  direction: {css_dir};
  text-align: {css_align};
}}

/* sidebar doc item */
.s-doc {{
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.15);
  border-radius: 6px;
  padding: 0.55rem 0.8rem;
  margin: 0.3rem 0;
  direction: {css_dir};
  text-align: {css_align};
}}
.s-doc .s-name {{ font-size:0.82rem; font-weight:700; color:#f0ece4 !important; }}
.s-doc .s-meta {{ font-size:0.72rem; color:#b0a898 !important; margin-top:0.15rem; }}

/* sidebar stat */
.s-stat {{
  display:flex;
  justify-content:space-between;
  align-items:center;
  padding: 0.4rem 0;
  border-bottom: 1px solid rgba(255,255,255,.06);
  font-size:0.8rem;
  direction: {css_dir};
}}
.s-stat .k {{ color:#7a7870 !important; }}
.s-stat .v {{ color:#d0cec8 !important; font-weight:600; }}

/* upload success */
.up-ok {{
  background: rgba(20,160,133,.1);
  border: 1px solid rgba(20,160,133,.35);
  border-radius: 8px;
  padding: 0.85rem 1rem;
  direction: {css_dir};
  text-align: {css_align};
  font-size:0.83rem;
  line-height:1.8;
  color: #0a6657 !important;
  margin-top: 0.6rem;
}}

footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# HEADER (Top) & Language Toggle
# ──────────────────────────────────────────────────────────────
_, col_lang = st.columns([8, 2])
with col_lang:
    if st.button(L["change_lang"], use_container_width=True):
        st.session_state.ui_lang = L["lang_code"]
        st.rerun()

st.markdown(
    f'<div class="page-header">'
    f'  <div>'
    f'    <div class="page-header-title">{L["header_title"]}</div>'
    f'    <div class="page-header-sub">{L["header_sub"]}</div>'
    f'  </div>'
    f'  <div class="page-header-logo">RAG v1.0 · FastAPI · ChromaDB</div>'
    f'</div>',
    unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="bidi-box"><h3>{L["sidebar_docs"]}</h3></div>', unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader(
        L["upload_label"],
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        if st.button(L["upload_btn"], use_container_width=True):
            with st.spinner(L["processing"]):
                r = api("post", "/upload",
                        files={"file": (uploaded_file.name,
                                        uploaded_file.getvalue(),
                                        "application/pdf")})
            if r["success"]:
                d = r["data"]
                st.session_state.docs_cache = None   # invalidate — re-fetched on rerun
                st.session_state.upload_ok = (
                    f'{esc(L["upload_success"])}<br>'
                    f'PDF: {esc(d.get("filename",""))}<br>'
                    f'{esc(d.get("page_count","?"))} {esc(L["pages"])} · {esc(d.get("chunk_count","?"))} {esc(L["chunks"])}'
                )
                st.rerun()   # force sidebar doc list to refresh immediately
            else:
                st.session_state.upload_ok = None
                st.error(f"❌ {err_msg(r.get('error', {}))}")

    # Upload success message (persists across reruns)
    if st.session_state.get("upload_ok"):
        st.markdown(
            f'<div class="up-ok">{st.session_state.upload_ok}</div>',
            unsafe_allow_html=True)

    st.markdown("---")

    # Document list
    _docs = docs_list()
    if _docs:
        for _d in _docs:
            st.markdown(
                f'<div class="s-doc">'
                f'<div class="s-name">PDF: {esc(_d.get("filename",""))}</div>'
                f'<div class="s-meta">{esc(_d.get("page_count","?"))} {esc(L["pages"])} · {esc(_d.get("chunk_count","?"))} {esc(L["chunks"])}</div>'
                f'</div>', unsafe_allow_html=True)
    else:
        st.caption(f'<div class="bidi-box">{L["no_docs"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Health
    st.markdown(f'<div class="bidi-box"><h3>{L["health"]}</h3></div>', unsafe_allow_html=True)
    _h = api("get", "/health")
    if _h["success"]:
        _hd = _h["data"]
        _ok = _hd.get("status") == "healthy"
        st.markdown(
            f'<div class="s-stat"><span class="k">{L["status"]}</span>'
            f'<span class="v" style="color:{"#39d353" if _ok else "#d29922"} !important;">'
            f'{L["online"] if _ok else L["partial"]}</span></div>'
            f'<div class="s-stat"><span class="k">{L["model"]}</span>'
            f'<span class="v">{esc(_hd.get("llm_model","N/A"))}</span></div>'
            f'<div class="s-stat"><span class="k">{L["docs_count"]}</span>'
            f'<span class="v">{esc(_hd.get("documents_count",0))}</span></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="s-stat"><span class="k">{L["status"]}</span>'
            f'<span class="v" style="color:#f85149 !important;">{L["offline"]}</span></div>',
            unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# MAIN — Question form
# ──────────────────────────────────────────────────────────────
st.markdown(f'<span class="sec-lbl">{L["q_label"]}</span>', unsafe_allow_html=True)

question = st.text_area(
    label="q",
    placeholder=L["q_placeholder"],
    height=115,
    label_visibility="collapsed",
    key="q_input",
)

if css_dir == "rtl":
    c1, c2, c3 = st.columns([3, 1, 1])
else:
    # LTR puts button on left
    c1, c2, c3 = st.columns([3, 1, 1])

with c1:
    ask = st.button(L["ask_btn"], use_container_width=True)
with c2:
    top_k = st.selectbox(L["top_k"], [3, 5, 7, 10], index=1)
with c3:
    _filter_opts = [L["all_docs"]] + [
        d.get("filename", "") for d in docs_list() if d.get("filename")
    ]
    doc_filter = st.selectbox(L["doc_filter"], _filter_opts, index=0)

# ──────────────────────────────────────────────────────────────
# HANDLE ASK — write to state ONLY, never render here
# ──────────────────────────────────────────────────────────────
if ask:
    if not question.strip():
        st.warning(L["empty_q"])
    else:
        body: dict = {"question": question.strip(), "top_k": top_k}
        if doc_filter != L["all_docs"]:
            for _d in docs_list():
                if _d.get("filename") == doc_filter:
                    body["document_id"] = _d.get("document_id")
                    break

        with st.spinner(L["searching"]):
            _t = time.time()
            _r = api("post", "/ask", json=body)
            _elapsed = time.time() - _t

        # Store — reset translation for new question
        st.session_state.qa_result  = _r
        st.session_state.qa_elapsed = _elapsed
        st.session_state.translation = None


# ──────────────────────────────────────────────────────────────
# RENDER RESULT — reads ONLY from session_state
# Runs on EVERY rerun (including translate button click)
# ──────────────────────────────────────────────────────────────
if st.session_state.qa_result is not None:
    _res     = st.session_state.qa_result
    _elapsed = st.session_state.qa_elapsed

    st.markdown('<hr style="margin:1.4rem 0 0;">', unsafe_allow_html=True)

    if not _res["success"]:
        st.error(f"❌  {err_msg(_res.get('error', {}))}")
    else:
        _d     = _res["data"]
        _ans   = (_d.get("answer") or "").strip()
        _conf  = float(_d.get("confidence") or 0.0)
        _srcs  = _d.get("sources") or []
        _dial  = bool(_d.get("dialect_detected"))
        _norm  = (_d.get("query_normalized") or "").strip()

        # dialect notice
        if _dial and _norm:
            st.markdown(
                f'<div class="dialect-box">'
                f'{esc(L["dialect_notice"])} <strong>{esc(_norm)}</strong>'
                f'</div>', unsafe_allow_html=True)

        _no_ans = bool(_ans) and "المعلومات غير موجودة" in _ans

        # ── No answer ──────────────────────────────────────
        if not _ans or _no_ans:
            st.markdown(
                f'<div class="no-answer">{L["no_answer"]}</div>',
                unsafe_allow_html=True)

        # ── Answer card ────────────────────────────────────
        else:
            # badge
            if _conf >= 0.7:
                _badge = f'<span class="badge badge-hi">{L["conf_hi"]} — {_conf:.0%}</span>'
            elif _conf >= 0.4:
                _badge = f'<span class="badge badge-mid">{L["conf_mid"]} — {_conf:.0%}</span>'
            else:
                _badge = f'<span class="badge badge-lo">{L["conf_lo"]} — {_conf:.0%}</span>'

            st.markdown(
                f'<div class="answer-wrap">'
                f'  <div class="answer-meta-row">'
                f'    {_badge}'
                f'    <span class="timing">⏱ {_elapsed:.1f}s</span>'
                f'  </div>'
                f'  <div class="answer-body">{esc(_ans)}</div>'
                f'</div>',
                unsafe_allow_html=True)

            # ── Translation ────────────────────────────────
            # Button always rendered here. On click: calls API, stores result.
            _tc1, _tc2 = st.columns([3, 2] if css_dir == "rtl" else [2, 3])
            _tr_col = _tc2 if css_dir == "rtl" else _tc1
            with _tr_col:
                _do_translate = st.button(
                    L["translate_btn"],
                    key="tr_btn",
                    use_container_width=True,
                )

            if _do_translate:
                with st.spinner("Translating…"):
                    _tr = api("post", "/translate", json={"text": _ans})
                if _tr["success"]:
                    st.session_state.translation = _tr["data"].get("translation", "")
                else:
                    st.session_state.translation = ""
                    st.error(f"Translation failed: {err_msg(_tr.get('error', {}))}")

            if st.session_state.translation:
                st.markdown(
                    f'<div class="translation-box">'
                    f'<span class="tl-label">{esc(L["translation_label"])}</span>'
                    f'{esc(st.session_state.translation)}'
                    f'</div>',
                    unsafe_allow_html=True)

            # confidence bar
            st.markdown(f'<span class="sec-lbl">{L["confidence_bar"]}</span>', unsafe_allow_html=True)
            st.progress(_conf)

        # ── Sources ────────────────────────────────────────
        if _srcs:
            st.markdown(f'<span class="sec-lbl">{L["sources_label"]}</span>', unsafe_allow_html=True)
            for _i, _s in enumerate(_srcs, 1):
                _pg   = _s.get("page", "?")
                _doc  = _s.get("document", "")
                _exc  = (_s.get("excerpt") or "").strip()
                _sc   = _s.get("similarity_score")          # None check, not falsy
                _pct  = f" · {_sc:.0%}" if _sc is not None else ""
                st.markdown(
                    f'<div class="src-card">'
                    f'<div class="src-hdr">[{_i}] {esc(_doc)} — {esc(L["page_str"])} {esc(_pg)}{esc(_pct)}</div>'
                    f'<div class="src-text">"{esc(_exc)}"</div>'
                    f'</div>',
                    unsafe_allow_html=True)

        # raw JSON
        with st.expander(L["view_json"]):
            st.json(_d)


# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown(
    '<hr style="margin-top:3rem;">'
    f'<p style="text-align:center; font-family:\'IBM Plex Mono\',monospace; '
    f'font-size:0.68rem; letter-spacing:.1em; color:#9a9490; padding:.5rem 0;">'
    f'{L["footer"]}'
    f'</p>',
    unsafe_allow_html=True)
