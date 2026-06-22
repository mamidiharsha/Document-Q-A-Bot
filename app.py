"""
Streamlit Web UI — Document Q&A Bot

A polished web interface for the RAG Document Q&A Bot featuring:
- Chat-style conversation interface
- Source citations in expandable sections
- Sidebar with index statistics and configuration
- Session-based conversation history

Usage:
    streamlit run app.py
"""

import streamlit as st
from src.pipeline import RAGPipeline
from src.config import TOP_K


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="📚 Document Q&A Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }

    /* Chat message styling */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Source cards */
    .source-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.85em;
    }

    .source-header {
        color: #a5b4fc;
        font-weight: 600;
        font-size: 0.9em;
        margin-bottom: 6px;
    }

    .source-excerpt {
        color: #94a3b8;
        font-size: 0.83em;
        line-height: 1.5;
    }

    .score-badge {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78em;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e3f 0%, #0f0f23 100%) !important;
    }

    /* Stats card */
    .stats-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        text-align: center;
    }

    .stats-number {
        font-size: 2em;
        font-weight: 700;
        color: #a5b4fc;
    }

    .stats-label {
        font-size: 0.85em;
        color: #64748b;
        margin-top: 4px;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
    }

    .main-header h1 {
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        font-weight: 800;
        margin-bottom: 0.3em;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1.05em;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None


# ---------------------------------------------------------------------------
# Pipeline Initialization
# ---------------------------------------------------------------------------


@st.cache_resource
def init_pipeline():
    """Initialize the RAG pipeline (cached across sessions)."""
    try:
        pipeline = RAGPipeline()
        return pipeline, None
    except SystemExit as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


pipeline, error = init_pipeline()

if error:
    st.error(f"❌ Failed to initialize pipeline: {error}")
    st.info(
        "Make sure you have set up your `.env` file with a valid "
        "`GOOGLE_API_KEY`. See `.env.example` for the template."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 2.5em;">🤖</span>
            <h2 style="margin: 0.3em 0 0.1em; color: #a5b4fc;">Document Q&A Bot</h2>
            <p style="color: #64748b; font-size: 0.9em;">Powered by RAG Pipeline</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Index statistics
    if pipeline and pipeline.vector_store.is_indexed():
        stats = pipeline.get_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="stats-card">
                    <div class="stats-number">{stats['indexed_chunks']}</div>
                    <div class="stats-label">Chunks Indexed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="stats-card">
                    <div class="stats-number">{len(stats['sources'])}</div>
                    <div class="stats-label">Documents</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Indexed documents list
        st.markdown("##### 📁 Indexed Documents")
        for source in stats["sources"]:
            ext = source.rsplit(".", 1)[-1].upper() if "." in source else "?"
            icon = {"PDF": "📕", "TXT": "📄", "DOCX": "📘"}.get(ext, "📄")
            st.markdown(f"{icon} `{source}`")

        st.divider()

    # Configuration
    st.markdown("##### ⚙️ Settings")
    top_k = st.slider(
        "Number of source chunks (top-k)",
        min_value=1,
        max_value=10,
        value=TOP_K,
        help="How many document chunks to retrieve for context",
    )

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Example questions
    st.markdown("##### 💡 Example Questions")
    example_questions = [
        "What health benefits does coffee provide?",
        "How does wind energy compare to solar energy?",
        "What is the habit loop?",
        "Which planet has the tallest mountain?",
        "What is active listening?",
    ]

    for q in example_questions:
        if st.button(f"❓ {q}", use_container_width=True, key=f"example_{q}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

# Header
if not st.session_state.messages:
    st.markdown(
        """
        <div class="main-header">
            <h1>📚 Document Q&A Bot</h1>
            <p>Ask questions about your document collection. Answers are grounded in source material with citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Check if documents are indexed
if not pipeline.vector_store.is_indexed():
    st.warning(
        "⚠️ No documents have been indexed yet. "
        "Please run `python index.py` first to index your documents."
    )
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources if available
        if message.get("sources"):
            with st.expander("📚 View Source Citations", expanded=False):
                for src in message["sources"]:
                    page_info = ""
                    if src.get("section"):
                        page_info = src["section"]
                    elif src.get("page"):
                        page_info = f"Page {src['page']}"

                    score = src.get("score", 0)

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-header">
                                [Source {src.get('index', '?')}] {src.get('source', 'unknown')}
                                {f' — {page_info}' if page_info else ''}
                                <span class="score-badge">{score:.0%} match</span>
                            </div>
                            <div class="source-excerpt">{src.get('excerpt', '')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                answer = pipeline.query(prompt, top_k=top_k)

                st.markdown(answer.answer)

                # Display sources
                if answer.sources:
                    with st.expander("📚 View Source Citations", expanded=True):
                        for src in answer.sources:
                            page_info = ""
                            if src.get("section"):
                                page_info = src["section"]
                            elif src.get("page"):
                                page_info = f"Page {src['page']}"

                            score = src.get("score", 0)

                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <div class="source-header">
                                        [Source {src.get('index', '?')}] {src.get('source', 'unknown')}
                                        {f' — {page_info}' if page_info else ''}
                                        <span class="score-badge">{score:.0%} match</span>
                                    </div>
                                    <div class="source-excerpt">{src.get('excerpt', '')}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Save to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer.answer,
                        "sources": answer.sources,
                    }
                )

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
