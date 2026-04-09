import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="NER Demo", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}
.stApp { background-color: #0a0a0f; }
#MainMenu, footer, header { visibility: hidden; }

.ner-header {
    text-align: center;
    padding: 48px 0 32px;
}
.ner-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.ner-header p {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

.stTextArea textarea {
    background: #13131a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    padding: 16px !important;
    transition: border 0.2s;
}
.stTextArea textarea:focus {
    border: 1px solid #60a5fa !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,0.08) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 36px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    width: 100% !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

.legend-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 24px;
}
.legend-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}

.highlight-box {
    background: #0f0f17;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 1.05rem;
    line-height: 2.2;
    letter-spacing: 0.2px;
}

.ent-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.92rem;
    margin: 0 2px;
}
.ent-label {
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    vertical-align: super;
    margin-left: 3px;
    opacity: 0.85;
}

.ent-list-item {
    display: grid;
    grid-template-columns: 1fr auto auto;
    align-items: center;
    gap: 16px;
    background: #0f0f17;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 10px;
}
.ent-word {
    font-size: 1rem;
    font-weight: 500;
    color: #f1f5f9;
}
.ent-type-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
    letter-spacing: 0.5px;
}
.ent-score {
    font-size: 0.82rem;
    color: #475569;
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 120px;
}

.conf-bar-bg {
    background: #1e1e2e;
    border-radius: 999px;
    height: 4px;
    width: 60px;
    display: inline-block;
    vertical-align: middle;
}
.conf-bar-fill {
    height: 4px;
    border-radius: 999px;
    display: block;
}

.stats-row {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}
.stat-box {
    flex: 1;
    background: #0f0f17;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stat-number {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 2px;
    letter-spacing: 0.5px;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e1e2e;
}
</style>
""", unsafe_allow_html=True)


# ── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="first")


# ── Colors ───────────────────────────────────────────────────────────────────
COLORS = {
    "PER":  {"bg": "rgba(96,165,250,0.18)",  "text": "#60a5fa"},
    "ORG":  {"bg": "rgba(52,211,153,0.18)",  "text": "#34d399"},
    "LOC":  {"bg": "rgba(245,158,11,0.18)",  "text": "#f59e0b"},
    "MISC": {"bg": "rgba(244,114,182,0.18)", "text": "#f472b6"},
}


# ── Highlight ────────────────────────────────────────────────────────────────
def highlight_text(text, entities):
    for ent in sorted(entities, key=lambda x: -len(x["word"])):
        word  = ent["word"].replace("##", "").strip()
        label = ent["entity_group"]
        c     = COLORS.get(label, {"bg": "#1e1e2e", "text": "#e2e8f0"})
        span  = (
            f"<span class='ent-tag' style='background:{c['bg']};color:{c['text']};'>"
            f"{word}<span class='ent-label'>{label}</span></span>"
        )
        text = text.replace(ent["word"], span, 1)
    return text


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='ner-header'>
    <h1>Named Entity Recognition</h1>
    <p>BERT · dslim/bert-base-NER · CoNLL-2003</p>
</div>
""", unsafe_allow_html=True)


# ── Legend ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='legend-row'>
    <span class='legend-pill' style='background:rgba(96,165,250,0.12);color:#60a5fa;'>● PER &nbsp; Person</span>
    <span class='legend-pill' style='background:rgba(52,211,153,0.12);color:#34d399;'>● ORG &nbsp; Organization</span>
    <span class='legend-pill' style='background:rgba(245,158,11,0.12);color:#f59e0b;'>● LOC &nbsp; Location</span>
    <span class='legend-pill' style='background:rgba(244,114,182,0.12);color:#f472b6;'>● MISC &nbsp; Miscellaneous</span>
</div>
""", unsafe_allow_html=True)


# ── Input ────────────────────────────────────────────────────────────────────
text = st.text_area(
    label="Input Text",
    placeholder="e.g. Maybe Sundar Pichai joined Google in Mountain View.",
    height=120,
    label_visibility="collapsed"
)

analyze = st.button("ANALYZE →")


# ── Output ───────────────────────────────────────────────────────────────────
if analyze:
    if text.strip():
        ner      = load_model()
        entities = ner(text)

        if entities:
            per_count  = sum(1 for e in entities if e["entity_group"] == "PER")
            org_count  = sum(1 for e in entities if e["entity_group"] == "ORG")
            loc_count  = sum(1 for e in entities if e["entity_group"] == "LOC")
            misc_count = sum(1 for e in entities if e["entity_group"] == "MISC")
            avg_conf   = sum(e["score"] for e in entities) / len(entities)

            st.markdown(f"""
            <div class='stats-row'>
                <div class='stat-box'><div class='stat-number'>{len(entities)}</div><div class='stat-label'>Entities</div></div>
                <div class='stat-box'><div class='stat-number'>{per_count}</div><div class='stat-label'>Persons</div></div>
                <div class='stat-box'><div class='stat-number'>{org_count}</div><div class='stat-label'>Orgs</div></div>
                <div class='stat-box'><div class='stat-number'>{loc_count}</div><div class='stat-label'>Locations</div></div>
                <div class='stat-box'><div class='stat-number'>{avg_conf:.0%}</div><div class='stat-label'>Avg Confidence</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-label'>Highlighted Output</div>", unsafe_allow_html=True)
            highlighted = highlight_text(text, entities)
            st.markdown(f"<div class='highlight-box'>{highlighted}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("<div class='section-label'>Detected Entities</div>", unsafe_allow_html=True)
            for ent in entities:
                word  = ent["word"].replace("##", "").strip()
                c     = COLORS.get(ent["entity_group"], {"bg": "#1e1e2e", "text": "#e2e8f0"})
                score = ent["score"]
                bar_w = int(score * 60)
                st.markdown(f"""
                <div class='ent-list-item'>
                    <span class='ent-word'>{word}</span>
                    <span class='ent-type-badge' style='background:{c['bg']};color:{c['text']};'>{ent['entity_group']}</span>
                    <span class='ent-score'>
                        {score:.2%}
                        <span class='conf-bar-bg'>
                            <span class='conf-bar-fill' style='width:{bar_w}px;background:{c['text']};'></span>
                        </span>
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No entities detected in the text.")
    else:
        st.warning("Please enter some text first.")


#---------------------
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"