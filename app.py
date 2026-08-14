"""
Run with:  streamlit run app.py
"""
import io
import json
import logging
import threading
import time
from pathlib import Path

import fitz  
import streamlit as st
from PIL import Image

import config
from src.pipeline import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="Academic Literature RAG", layout="wide")

# Global CSS
st.markdown(
    """
    <style>
    /* Scrollable chat pane fills remaining viewport height */
    div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"]
        div[data-testid="stContainer"] {
        scrollbar-width: thin;
    }

    /* Polish the native (already-pinned) chat input bar */
    [data-testid="stBottom"] {
        background: linear-gradient(to top, rgba(14,17,23,0.97) 60%, rgba(14,17,23,0.75));
        backdrop-filter: blur(6px);
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
    }

    /* Pulsing animated glow bar shown while a question is being processed */
    .glow-bar-wrap { padding: 4px 0 10px 0; }
    .glow-bar {
        height: 5px;
        width: 100%;
        border-radius: 999px;
        background: linear-gradient(270deg, #7C6CFF, #17C3FF, #7C6CFF);
        background-size: 600% 600%;
        animation: glow-move 2.2s ease infinite, glow-shadow 2.2s ease infinite;
    }
    @keyframes glow-move {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes glow-shadow {
        0%, 100% { box-shadow: 0 0 4px 1px rgba(124,108,255,0.45); }
        50%      { box-shadow: 0 0 12px 3px rgba(23,195,255,0.75); }
    }

    /* Blinking typing cursor used during progressive answer reveal */
    .type-cursor { animation: blink 0.9s steps(1) infinite; opacity: 0.6; }
    @keyframes blink { 50% { opacity: 0; } }
    </style>
    """,
    unsafe_allow_html=True,
)
# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, citations, query_type}
if "preview" not in st.session_state:
    st.session_state.preview = None  # {source, page, snippet, paper_title}


def set_preview(citation: dict):
    st.session_state.preview = citation


# PDF page rendering helper
@st.cache_data(show_spinner=False)
def render_pdf_page(source_path: str, page_number: int, snippet: str = "") -> bytes:
    """Render a single PDF page to a PNG, highlighting the cited snippet
    (a "hover-to-see" style preview: the exact passage is boxed in yellow)."""
    doc = fitz.open(source_path)
    page = doc[page_number - 1]

    if snippet:
        # Search for a short, distinctive slice of the snippet (full snippets
        # rarely match exactly due to whitespace/line-break differences).
        needle = " ".join(snippet.split())[:80]
        for rect in page.search_for(needle):
            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=(1, 0.85, 0))
            highlight.update()

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


# Layout
def render_citation_buttons(msg):
    if not msg.get("citations"):
        return
    st.caption("Sources:")
    cols = st.columns(min(len(msg["citations"]), 4) or 1)
    for i, cit in enumerate(msg["citations"]):
        label = f"[{cit['paper_title']}, p.{cit['page']}]"
        with cols[i % len(cols)]:
            if st.button(label, key=f"cite_{id(msg)}_{i}"):
                set_preview(cit)
st.title("📚 Academic Literature RAG Assistant")
st.caption(
    "Ask questions across your entire related-work / bibliography folder. "
    "Click any citation tag to preview the exact page it came from."
)

if not config.GOOGLE_API_KEY:
    st.warning(
        "GOOGLE_API_KEY is not set. Add it to a `.env` file (see `.env.example`) "
        "before asking questions.",
        icon="⚠️",
    )

if not config.CHROMA_DIR.exists() or not any(config.CHROMA_DIR.iterdir() if config.CHROMA_DIR.exists() else []):
    st.info(
        "No index found yet. Add PDFs to `data/pdfs/` and run "
        "`python scripts/build_index.py` from a terminal, then reload this page.",
        icon="ℹ️",
    )

left, right = st.columns([3, 2], gap="large")

with left:
    chat_container = st.container(height=560)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                render_citation_buttons(msg)

    question = st.chat_input("Ask about your papers (e.g. 'What dataset did the ViT paper use?')")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                # Interactive "reasoning" status dropdown 
                result_holder = {}

                def _worker():
                    try:
                        result_holder["result"] = answer_question(question)
                    except Exception as exc:  # noqa: BLE001
                        result_holder["error"] = exc

                worker_thread = threading.Thread(target=_worker, daemon=True)
                worker_thread.start()

                stages = [
                    "Understanding your question…",
                    "Routing to the best retrieval strategy…",
                    "Searching your papers (hybrid search)…",
                    "Re-ranking the most relevant passages…",
                    "Drafting a grounded answer…",
                ]

                with st.status("🧠 Thinking…", expanded=True) as status:
                    glow_slot = st.empty()
                    glow_slot.markdown(
                        '<div class="glow-bar-wrap"><div class="glow-bar"></div></div>',
                        unsafe_allow_html=True,
                    )
                    i = 0
                    while worker_thread.is_alive():
                        status.update(label=f"🧠 {stages[i % len(stages)]}")
                        i += 1
                        time.sleep(0.85)
                    worker_thread.join()
                    glow_slot.empty()

                    if "error" in result_holder:
                        status.update(label="⚠️ Something went wrong", state="error", expanded=False)
                    else:
                        status.update(label="✅ Answer ready", state="complete", expanded=False)

                # --- Answer: progressive, "typed" reveal --------------------
                answer_slot = st.empty()
                if "error" in result_holder:
                    full_answer = f"⚠️ Error: {result_holder['error']}"
                    answer_slot.markdown(full_answer)
                    citations, query_type = [], None
                else:
                    result = result_holder["result"]
                    full_answer = result.answer
                    citations = result.citations
                    query_type = getattr(result, "query_type", None)

                    displayed = ""
                    words = full_answer.split(" ")
                    step = 3  # words revealed per tick - tune for pacing
                    for idx in range(0, len(words), step):
                        displayed += (" " if displayed else "") + " ".join(words[idx: idx + step])
                        answer_slot.markdown(displayed + ' <span class="type-cursor">▌</span>', unsafe_allow_html=True)
                        time.sleep(0.025)
                    answer_slot.markdown(displayed)

                new_msg = {
                    "role": "assistant",
                    "content": full_answer,
                    "citations": citations,
                    "query_type": query_type,
                }
                render_citation_buttons(new_msg)

        st.session_state.messages.append(new_msg)
        if citations:
            set_preview(citations[0])
        st.rerun()
with right:
    st.subheader("📄 Source preview")
    preview = st.session_state.preview
    if preview is None:
        st.info("Click a citation tag on the left to preview the source page here.")
    else:
        st.markdown(f"**{preview['paper_title']}** — page {preview['page']}")
        source_path = preview.get("source")
        try:
            img_bytes = render_pdf_page(source_path, int(preview["page"]), preview.get("snippet", ""))
            st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not render page: {e}")
        with st.expander("Cited snippet (text)"):
            st.write(preview.get("snippet", ""))


# Sidebar: corpus status / admin
with st.sidebar:
    st.header("Corpus")
    pdf_count = len(list(config.PDF_DIR.glob("*.pdf"))) if config.PDF_DIR.exists() else 0
    st.metric("PDFs in data/pdfs/", pdf_count)

    if config.SKIPPED_LOG_PATH.exists():
        skipped = json.loads(config.SKIPPED_LOG_PATH.read_text())
        st.metric("Skipped (scanned/unreadable)", len(skipped))
        if skipped:
            with st.expander("View skipped files"):
                for s in skipped:
                    st.write(f"- **{s['file']}** — {s['reason']}")

    st.divider()
    st.caption(
        "To (re)index after adding/removing PDFs, run in a terminal:\n\n"
        "`python scripts/build_index.py`"
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.preview = None
        st.rerun()
