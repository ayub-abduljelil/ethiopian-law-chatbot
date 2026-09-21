"""
Ethiopian Legal RAG Assistant — Streamlit frontend.
"""

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from local_llm import AiModel
from embedding import Embedding
from auth import register, login
from db_pgvector import (
    init_db,
    search_containers,
    search_chunks,
    count_chunks,
    count_containers,
    create_chat,
    list_chats,
    load_chat_messages,
    save_chat_messages,
    delete_chat,
)

# ---------------------------------------------------------------------------
# Retrieval config
# ---------------------------------------------------------------------------

TOP_CONTAINERS  = 3   # kc
CHUNKS_PER_CONT = 9   # kch
DIRECT_CHUNKS   = 5
MAX_CHUNKS      = 12

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ethiopian Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# DB bootstrap
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _ensure_db():
    init_db()

_ensure_db()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

for key, default in [
    ("user",       None),
    ("chat_id",    None),
    ("messages",   []),
    ("embedding",  None),
    ("auth_page",  "login"),
    ("show_auth",  False),
    ("chunk_count", 0),
    ("cont_count",  0),
    ("dark_mode",  True),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Theme CSS
# ---------------------------------------------------------------------------

def apply_theme_css():
    """Apply consistent theme CSS that actually works."""
    dark = st.session_state.dark_mode
    
    if dark:
        # DARK THEME
        bg_main = "#0e1117"
        bg_secondary = "#1a1d28"
        bg_tertiary = "#262a38"
        text_primary = "#fafafa"
        text_secondary = "#a0a0a0"
        border_color = "#2e3347"
        accent_color = "#4c8bf5"
        accent_hover = "#3a7be0"
        hover_bg = "#1e2235"
        input_bg = "#1a1d28"
    else:
        # LIGHT THEME
        bg_main = "#ffffff"
        bg_secondary = "#f8f9fa"
        bg_tertiary = "#e9ecef"
        text_primary = "#1a1a1a"
        text_secondary = "#6c757d"
        border_color = "#dee2e6"
        accent_color = "#0d6efd"
        accent_hover = "#0b5ed7"
        hover_bg = "#f1f3f5"
        input_bg = "#ffffff"
    
    st.markdown(f"""
    <style>
    /* RESET & BASE */
    * {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }}
    
    html, body, [data-testid="stApp"], .main {{
        background-color: {bg_main} !important;
        color: {text_primary} !important;
    }}
    
    .main .block-container {{
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {{
        background-color: {bg_secondary} !important;
        border-right: 1px solid {border_color} !important;
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: {text_primary} !important;
    }}
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption {{
        color: {text_secondary} !important;
    }}
    
    [data-testid="stSidebar"] .stButton > button {{
        background-color: transparent !important;
        color: {text_primary} !important;
        border: 1px solid transparent !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.75rem !important;
        font-size: 0.875rem !important;
        width: 100% !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
    }}
    
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {hover_bg} !important;
        border-color: {border_color} !important;
    }}
    
    /* NEW CHAT BUTTON */
    .new-chat-btn .stButton > button {{
        background-color: {accent_color} !important;
        color: #ffffff !important;
        border-color: {accent_color} !important;
        font-weight: 500 !important;
    }}
    
    .new-chat-btn .stButton > button:hover {{
        background-color: {accent_hover} !important;
        border-color: {accent_hover} !important;
    }}
    
    .new-chat-btn .stButton > button:disabled {{
        background-color: {bg_tertiary} !important;
        color: {text_secondary} !important;
        border-color: {border_color} !important;
        opacity: 0.6 !important;
    }}
    
    /* ACTIVE CHAT */
    .active-chat .stButton > button {{
        background-color: {hover_bg} !important;
        border-color: {border_color} !important;
        font-weight: 500 !important;
    }}
    
    /* DELETE BUTTON */
    .del-btn .stButton > button {{
        color: {text_secondary} !important;
        font-size: 1.2rem !important;
        padding: 0.25rem 0.5rem !important;
        min-width: auto !important;
        width: auto !important;
    }}
    
    .del-btn .stButton > button:hover {{
        color: #dc3545 !important;
        background-color: rgba(220, 53, 69, 0.1) !important;
    }}
    
    /* THEME TOGGLE */
    .theme-btn .stButton > button {{
        font-size: 1.2rem !important;
        padding: 0.25rem 0.5rem !important;
        min-width: auto !important;
        width: auto !important;
    }}
    
    /* DIVIDERS */
    [data-testid="stSidebar"] hr,
    .main hr {{
        border-color: {border_color} !important;
        margin: 1rem 0 !important;
    }}
    
    /* CHAT MESSAGES */
    [data-testid="stChatMessage"] {{
        background-color: transparent !important;
        padding: 0.75rem 0 !important;
    }}
    
    [data-testid="stChatMessageContent"] {{
        background-color: {bg_secondary} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        color: {text_primary} !important;
    }}
    
    [data-testid="stChatMessageContent"] p {{
        color: {text_primary} !important;
        margin: 0 !important;
        line-height: 1.6 !important;
    }}
    
    /* CHAT INPUT */
    [data-testid="stChatInput"] {{
        background-color: transparent !important;
    }}
    
    [data-testid="stChatInput"] > div {{
        background-color: {bg_secondary} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
    }}
    
    [data-testid="stChatInput"] textarea {{
        background-color: transparent !important;
        color: {text_primary} !important;
        border: none !important;
    }}
    
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {text_secondary} !important;
    }}
    
    [data-testid="stChatInput"] button {{
        background-color: {accent_color} !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }}
    
    [data-testid="stChatInput"] button:hover {{
        background-color: {accent_hover} !important;
    }}
    
    /* INPUTS */
    input, textarea, .stTextInput input, .stTextArea textarea {{
        background-color: {input_bg} !important;
        color: {text_primary} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
    }}
    
    input:focus, textarea:focus {{
        border-color: {accent_color} !important;
        box-shadow: 0 0 0 2px {accent_color}33 !important;
        outline: none !important;
    }}
    
    input::placeholder, textarea::placeholder {{
        color: {text_secondary} !important;
    }}
    
    /* BUTTONS */
    .stButton > button {{
        background-color: {bg_secondary} !important;
        color: {text_primary} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}
    
    .stButton > button:hover {{
        background-color: {hover_bg} !important;
        border-color: {accent_color} !important;
    }}
    
    .stButton > button[kind="primary"] {{
        background-color: {accent_color} !important;
        color: #ffffff !important;
        border-color: {accent_color} !important;
    }}
    
    .stButton > button[kind="primary"]:hover {{
        background-color: {accent_hover} !important;
        border-color: {accent_hover} !important;
    }}
    
    /* STATUS */
    [data-testid="stStatus"] {{
        background-color: {bg_secondary} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
    }}
    
    [data-testid="stStatusWidget"] {{
        color: {text_primary} !important;
    }}
    
    /* HEADINGS */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_primary} !important;
    }}
    
    /* CAPTIONS */
    .stCaption {{
        color: {text_secondary} !important;
    }}
    
    /* AUTH CARD */
    .auth-card {{
        background-color: {bg_secondary} !important;
        border: 1px solid {border_color} !important;
        border-radius: 16px !important;
        padding: 3rem 2.5rem !important;
        max-width: 440px !important;
        margin: 4rem auto !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1) !important;
    }}
    
    .auth-heading {{
        color: {text_primary} !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        margin-bottom: 0.5rem !important;
    }}
    
    .auth-sub {{
        color: {text_secondary} !important;
        font-size: 0.875rem !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
    }}
    
    /* HIDE STREAMLIT STUFF */
    [data-testid="InputInstructions"],
    [data-testid="stChatMessageAvatarContainer"] {{
        display: none !important;
    }}
    
    /* SCROLLBAR */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {bg_main};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {border_color};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {text_secondary};
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme_css()

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_ai_model() -> AiModel:
    load_dotenv()
    return AiModel()

@st.cache_data(show_spinner=False)
def load_doc_meta() -> dict:
    p = Path("pdfs/metadata.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def doc_title(source: str, meta: dict) -> str:
    stem = source.replace(".pdf", "")
    return meta.get(stem, {}).get("title", stem.replace("_", " "))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(qv: list, meta: dict) -> str:
    seen:   set  = set()
    ranked: list = []

    for container, _ in search_containers(qv, k=TOP_CONTAINERS):
        path = container.get("hierarchy_path", "")
        for chunk, dist in search_chunks(qv, k=CHUNKS_PER_CONT,
                                         container_id=container["id"]):
            if chunk["id"] not in seen:
                seen.add(chunk["id"])
                ranked.append((dist, chunk, path))

    for chunk, dist in search_chunks(qv, k=DIRECT_CHUNKS):
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            ranked.append((dist, chunk, ""))

    ranked.sort(key=lambda x: x[0])
    ranked = ranked[:MAX_CHUNKS]
    if not ranked:
        return ""

    blocks = []
    for _, chunk, path in ranked:
        title  = doc_title(chunk.get("source", ""), meta)
        header = f"[{title}]" + (f" › {path}" if path else "")
        blocks.append(f"--- {header} ---\n{chunk.get('content','')}")
    return "\n\n".join(blocks)

# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------

def _switch_chat(chat_id: str) -> None:
    st.session_state.chat_id  = chat_id
    st.session_state.messages = load_chat_messages(chat_id)

def _new_chat() -> None:
    st.session_state.chat_id  = None
    st.session_state.messages = []

def _sidebar_label(title: str) -> str:
    title = (title or "New chat").strip()
    return title[:40] + "..." if len(title) > 40 else title

# ---------------------------------------------------------------------------
# Auth pages
# ---------------------------------------------------------------------------

def show_login():
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:3rem;margin-bottom:1rem;">⚖️</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-heading">Ethiopian Legal Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Sign in to your account</div>', unsafe_allow_html=True)

        u = st.text_input("Username", key="li_u", label_visibility="collapsed", placeholder="Username")
        p = st.text_input("Password", type="password", key="li_p", label_visibility="collapsed", placeholder="Password")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign in", use_container_width=True, type="primary"):
            if not u or not p:
                st.error("Please enter your username and password.")
            else:
                user, err = login(u, p)
                if err:
                    st.error(err)
                else:
                    st.session_state.user = user
                    _new_chat()
                    st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create account", use_container_width=True):
                st.session_state.auth_page = "register"
                st.rerun()
        with col2:
            if st.button("← Back", use_container_width=True):
                st.session_state.show_auth = False
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


def show_register():
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:3rem;margin-bottom:1rem;">⚖️</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-heading">Create account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Start using the Ethiopian Legal Assistant</div>', unsafe_allow_html=True)

        nu = st.text_input("Username", key="re_u", label_visibility="collapsed", placeholder="Username")
        ne = st.text_input("Email", key="re_e", label_visibility="collapsed", placeholder="Email")
        np = st.text_input("Password", type="password", key="re_p", label_visibility="collapsed", placeholder="Password")
        np2 = st.text_input("Confirm password", type="password", key="re_p2", label_visibility="collapsed", placeholder="Confirm password")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create account", use_container_width=True, type="primary"):
            user, err = register(nu, ne, np, np2)
            if err:
                st.error(err)
            else:
                st.session_state.user = user
                st.rerun()

        if st.button("← Back to Sign in", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

def show_landing():
    # Hide sidebar on landing
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)
    
    # HEADER / NAV
    logo_path = Path("assets/logo.png")
    logo_col, _, tagline_col = st.columns([1, 4, 2])
    with logo_col:
        if logo_path.exists():
            st.image(str(logo_path), width=140)
        else:
            st.markdown('**⚖️ Ethiopian Legal**', unsafe_allow_html=True)
    with tagline_col:
        st.markdown(
            '<div style="text-align:right;padding-top:1rem;font-size:0.875rem;font-weight:500;">Ethiopian Legal Intelligence Platform</div>',
            unsafe_allow_html=True,
        )
    st.divider()
    
    # HERO SECTION
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem 3rem;max-width:900px;margin:0 auto;">
        <div style="font-size:0.875rem;font-weight:600;letter-spacing:0.05em;color:#CF0001;margin-bottom:1.5rem;">
            SEARCH ETHIOPIAN LAW ACROSS ONE CONNECTED KNOWLEDGE BASE
        </div>
        <h1 style="font-size:clamp(2.5rem,5vw,4rem);font-weight:700;line-height:1.15;margin-bottom:1.5rem;">
            Ethiopian law,<br>answered <span style="color:#CF0001;">instantly</span>
        </h1>
        <p style="font-size:1.125rem;line-height:1.7;max-width:700px;margin:0 auto 2.5rem;opacity:0.85;">
            A legal research platform that uses semantic search to help you find relevant provisions 
            across Ethiopian law. Built for students, researchers, and legal professionals who need 
            a faster way to navigate and understand large bodies of legal text.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA BUTTONS
    c1, c2, c3, c4, c5 = st.columns([2, 1, 0.5, 1, 2])
    with c2:
        if st.button("Get Started →", use_container_width=True, type="primary"):
            st.session_state.show_auth = True
            st.session_state.auth_page = "register"
            st.rerun()
    with c4:
        if st.button("Sign In", use_container_width=True):
            st.session_state.show_auth = True
            st.session_state.auth_page = "login"
            st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # DEMO VIDEO
    video_path = Path("assets/demo.mp4")
    if video_path.exists():
        _, vid_col, _ = st.columns([0.5, 9, 0.5])
        with vid_col:
            st.video(str(video_path))
    else:
        st.info("💡 Drop a demo video at `assets/demo.mp4` to showcase the platform")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # FEATURES SECTION
    st.markdown("""
    <div style="text-align:center;margin:4rem 0 3rem;">
        <div style="font-size:0.75rem;font-weight:700;letter-spacing:0.15em;color:#CF0001;margin-bottom:0.75rem;">
            FEATURES
        </div>
        <h2 style="font-size:2rem;font-weight:700;margin-bottom:1rem;">
            Everything you need to navigate Ethiopian law
        </h2>
        <p style="font-size:1rem;max-width:600px;margin:0 auto 3rem;opacity:0.75;">
            The platform brings Ethiopia's Constitution and major legal codes into 
            a single structured, searchable knowledge base.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🏛 Full Legal Corpus
        Civil Code, Criminal Code, Commercial Code, Family Code, Maritime Code, 
        Civil Procedure Code, and the Constitution — fully indexed and searchable.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        ### 📎 Article Citations
        Every answer references the specific article and document it came from — 
        no guessing, no hallucination.
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Hierarchical Retrieval
        Two-step search: first identifies the relevant legal area, then retrieves 
        the exact articles — not just keyword matches.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        ### 💬 Persistent History
        All your conversations are saved per account. Pick up any previous research 
        session exactly where you left off.
        """)
    
    with col3:
        st.markdown("""
        ### ⚡ Instant Answers
        Ask in plain language. The system rewrites your query, retrieves the right 
        provisions, and generates a cited answer in seconds.
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔒 Secure Accounts
        Designed to work with structured Ethiopian legal documents and return relevant 
        legal provisions with their surrounding context.
        """)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    
    # HOW IT WORKS
    st.markdown("""
    <div style="text-align:center;margin:4rem 0 3rem;">
        <div style="font-size:0.75rem;font-weight:700;letter-spacing:0.15em;color:#CF0001;margin-bottom:0.75rem;">
            HOW IT WORKS
        </div>
        <h2 style="font-size:2rem;font-weight:700;margin-bottom:3rem;">
            From question to cited answer
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align:center;">
            <div style="width:50px;height:50px;border-radius:50%;background:#CF0001;color:white;
                        display:flex;align-items:center;justify-content:center;font-size:1.5rem;
                        font-weight:700;margin:0 auto 1rem;">1</div>
            <h3 style="font-size:1.125rem;font-weight:600;margin-bottom:0.5rem;">Ask</h3>
            <p style="font-size:0.9375rem;opacity:0.75;">
                Type your legal question in plain language — no legal jargon required.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align:center;">
            <div style="width:50px;height:50px;border-radius:50%;background:#CF0001;color:white;
                        display:flex;align-items:center;justify-content:center;font-size:1.5rem;
                        font-weight:700;margin:0 auto 1rem;">2</div>
            <h3 style="font-size:1.125rem;font-weight:600;margin-bottom:0.5rem;">Retrieve</h3>
            <p style="font-size:0.9375rem;opacity:0.75;">
                The system searches across 5,000+ articles from 7 Ethiopian legal codes.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align:center;">
            <div style="width:50px;height:50px;border-radius:50%;background:#CF0001;color:white;
                        display:flex;align-items:center;justify-content:center;font-size:1.5rem;
                        font-weight:700;margin:0 auto 1rem;">3</div>
            <h3 style="font-size:1.125rem;font-weight:600;margin-bottom:0.5rem;">Answer</h3>
            <p style="font-size:0.9375rem;opacity:0.75;">
                AI synthesizes a precise answer with article citations you can verify.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.divider()
    
    # FINAL CTA
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;">
        <h2 style="font-size:2.25rem;font-weight:700;margin-bottom:1rem;">
            From scattered legal texts to one searchable system
        </h2>
        <p style="font-size:1.125rem;max-width:650px;margin:0 auto 2.5rem;opacity:0.75;">
            Explore Ethiopian legal sources in one place and retrieve the provisions 
            most relevant to your question.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if st.button("Create Free Account", use_container_width=True, type="primary", key="cta2"):
            st.session_state.show_auth = True
            st.session_state.auth_page = "register"
            st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # FOOTER
    st.markdown("""
    <div style="text-align:center;padding:2rem;font-size:0.875rem;opacity:0.5;border-top:1px solid rgba(128,128,128,0.2);">
        © 2025 RARAS Technology · Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def show_app():
    user     = st.session_state.user
    ai_model = load_ai_model()
    meta     = load_doc_meta()

    # SIDEBAR
    with st.sidebar:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**⚖️ Ethiopian Legal**")
        with col2:
            st.markdown('<div class="theme-btn">', unsafe_allow_html=True)
            if st.button("🌓", key="theme_toggle", help="Toggle dark/light theme"):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.caption(f"Signed in as **{user['username']}**")
        st.divider()

        # New chat button
        has_messages = bool(st.session_state.messages)
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("+ New Chat", use_container_width=True, disabled=not has_messages):
            _new_chat()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # Chat history
        chats = list_chats(user["id"])
        active_id = st.session_state.chat_id

        if chats:
            for chat in chats:
                is_active = chat["id"] == active_id
                label = _sidebar_label(chat["title"])

                col_chat, col_del = st.columns([5, 1])
                
                with col_chat:
                    if is_active:
                        st.markdown('<div class="active-chat">', unsafe_allow_html=True)
                    if st.button(label, key=f"c_{chat['id']}", use_container_width=True):
                        if not is_active:
                            _switch_chat(chat["id"])
                            st.rerun()
                    if is_active:
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with col_del:
                    st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                    if st.button("×", key=f"d_{chat['id']}", help="Delete"):
                        delete_chat(chat["id"])
                        if chat["id"] == active_id:
                            _new_chat()
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("No previous chats")

        st.divider()

        # DB info
        if st.button("Database Stats", use_container_width=True):
            st.session_state.chunk_count = count_chunks()
            st.session_state.cont_count = count_containers()
        
        if st.session_state.chunk_count:
            st.caption(f"📊 {st.session_state.chunk_count:,} articles • {st.session_state.cont_count:,} containers")

        st.divider()
        
        if st.button("Sign Out", use_container_width=True):
            for k in ("user", "chat_id", "messages", "embedding"):
                st.session_state[k] = None if k != "messages" else []
            st.rerun()
        
        st.caption("Powered by OpenRouter")

    # MAIN CHAT AREA
    st.markdown("## Ethiopian Legal Assistant")
    st.caption("Ask questions about Ethiopian law in plain language")
    st.divider()

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    prompt = st.chat_input("What are the legal requirements for marriage in Ethiopia?")

    if prompt:
        messages = st.session_state.messages
        is_first = len(messages) == 0
        auto_title = prompt.strip()[:60] if is_first else None

        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.embedding is None:
                with st.spinner("Loading embedding model..."):
                    st.session_state.embedding = Embedding()

            with st.status("Processing...", expanded=False) as status:
                # Embed
                qv = st.session_state.embedding._embed_one(prompt)
                
                # Rewrite
                status.update(label="Rewriting query...")
                rewritten = ai_model.rewrite_query(prompt)
                
                # Retrieve
                status.update(label="Searching knowledge base...")
                rw_vec = st.session_state.embedding._embed_one(prompt)
                context = retrieve(rw_vec, meta)
                n_chunks = context.count("--- [")
                
                # Generate
                status.update(label=f"Generating answer ({n_chunks} articles)...")
                if not context:
                    reply = "I could not find relevant legal provisions for your question."
                    status.update(label="Complete", state="complete")
                else:
                    full_prompt = ai_model.full_prompt_for_rag(
                        relevent_sections=context,
                        question_prompt=prompt,
                    )
                    def gen():
                        yield ai_model.ask_a_question(full_prompt)
                    reply = st.write_stream(gen())
                    status.update(label="Complete", state="complete")
            
            if not context:
                st.markdown(reply)

        messages.append({"role": "assistant", "content": reply})
        st.session_state.messages = messages

        # Create/save chat
        if not st.session_state.chat_id:
            st.session_state.chat_id = create_chat(user["id"], title=auto_title)
        
        save_chat_messages(st.session_state.chat_id, messages, title=auto_title)

        if is_first:
            st.rerun()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if st.session_state.user is None:
    if st.session_state.get("show_auth"):
        if st.session_state.auth_page == "register":
            show_register()
        else:
            show_login()
    else:
        show_landing()
else:
    show_app()
