"""
AI Personal Knowledge & Productivity Assistant — Second Brain
Built by a Top-Tier Engineer (1Cr+ mindset)
Architecture: Multi-agent LangGraph | Groq LLM | Memory | Tools | Streamlit UI
"""

import streamlit as st
import os
import json
import random
import datetime
import uuid
from dotenv import load_dotenv

# ── ENV ──────────────────────────────────────────────────────────────────────
load_dotenv()

# ── PAGE CONFIG (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Second Brain AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── IMPORTS ───────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Bebas+Neue&display=swap');

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   CSK THEME — WHISTLE PODU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
:root {
    --bg-primary:    #f8fafc;
    --bg-secondary:  #ffffff;
    --bg-card:       #ffffff;
    --csk-yellow:    #FDB913;
    --csk-yellow2:   #E3A30C;
    --csk-blue:      #004B8D;
    --csk-blue2:     #1A6BBA;
    --text-primary:  #0f172a;
    --text-secondary:#334155;
    --text-muted:    #64748b;
    --border-dim:    #e2e8f0;
    --border-bright: #cbd5e1;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* CSK BACKGROUND */
.stApp {
    background:
        radial-gradient(ellipse at 15% 50%, rgba(253,185,19,0.1) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 20%, rgba(0,75,141,0.08) 0%, transparent 55%),
        radial-gradient(ellipse at 55% 85%, rgba(253,185,19,0.05) 0%, transparent 45%),
        linear-gradient(180deg, #f8fafc 0%, #ffffff 50%, #f8fafc 100%) !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 60%, #ffffff 100%) !important;
    border-right: 2px solid rgba(253,185,19,0.3) !important;
    box-shadow: 4px 0 25px rgba(253,185,19,0.08) !important;
}

h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.02em !important;
    color: var(--csk-blue) !important;
}

/* CSK TITLE GRADIENT */
.main-title {
    background: linear-gradient(135deg, var(--csk-yellow) 0%, var(--csk-yellow2) 45%, var(--csk-blue) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -0.01em;
    filter: drop-shadow(0 0 10px rgba(253,185,19,0.2));
}

.main-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 0.25rem;
    letter-spacing: 0.04em;
}

/* MODULE CARDS */
.module-card {
    background: var(--bg-card);
    border: 1px solid var(--border-bright);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
}

.module-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--csk-yellow), var(--csk-blue), var(--csk-yellow));
    background-size: 200% 100%;
    animation: csk-scan 3s linear infinite;
}

.module-card:hover {
    border-color: rgba(253,185,19,0.55) !important;
    box-shadow: 0 0 30px rgba(253,185,19,0.15), 0 0 60px rgba(0,75,141,0.08) !important;
    transform: translateY(-2px);
}

@keyframes csk-scan {
    0%   { background-position: 0% 0%; }
    100% { background-position: 200% 0%; }
}

/* METRIC CARDS */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-bright);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}

.metric-value {
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--csk-yellow), var(--csk-yellow2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 8px rgba(253,185,19,0.2));
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
}

/* BUTTONS */
.stButton button {
    background: linear-gradient(135deg, var(--csk-yellow), var(--csk-yellow2)) !important;
    color: var(--text-primary) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 900 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(253,185,19,0.3) !important;
}

.stButton button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 20px rgba(253,185,19,0.4) !important;
    background: linear-gradient(135deg, var(--csk-yellow2), var(--csk-yellow)) !important;
}

.stButton button:active {
    transform: translateY(1px) scale(0.98) !important;
}

/* INPUT FIELDS */
.stTextInput input, .stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    caret-color: var(--csk-blue) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--csk-yellow) !important;
    box-shadow: 0 0 0 2px rgba(253,185,19,0.2) !important;
}

.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-muted) !important; }

/* BADGES */
.badge { display: inline-block; padding: 0.2rem 0.8rem; border-radius: 4px; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
.badge-green  { background: rgba(253,185,19,0.15);   color: var(--csk-yellow2); border: 1px solid rgba(253,185,19,0.4); }
.badge-blue   { background: rgba(0,75,141,0.12); color: var(--csk-blue); border: 1px solid rgba(0,75,141,0.3); }
.badge-purple { background: rgba(0,75,141,0.15); color: var(--csk-blue2); border: 1px solid rgba(0,75,141,0.4); }
.badge-orange { background: rgba(253,185,19,0.1);  color: var(--csk-yellow2); border: 1px solid rgba(253,185,19,0.25); }

/* NOTE CARDS */
.note-card {
    background: var(--bg-card);
    border: 1px solid var(--border-bright);
    border-left: 4px solid var(--csk-yellow);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-left-color 0.2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}
.note-card:hover { border-left-color: var(--csk-blue); }

/* HERO BANNER */
.hero-banner {
    background:
        radial-gradient(ellipse at 25% 50%, rgba(253,185,19,0.1) 0%, transparent 65%),
        radial-gradient(ellipse at 80% 30%, rgba(0,75,141,0.06) 0%, transparent 55%),
        linear-gradient(135deg, #ffffff, #f8fafc);
    border: 1px solid rgba(253,185,19,0.35);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(253,185,19,0.08);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--csk-yellow) 35%, var(--csk-blue) 65%, transparent);
    animation: csk-scan 4s linear infinite;
}

.hero-banner::after {
    content: '🦁';
    position: absolute;
    right: 1.5rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.15;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 10px !important;
    padding: 0.3rem !important;
    border: 1px solid var(--border-bright) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--csk-yellow), var(--csk-yellow2)) !important;
    color: var(--text-primary) !important;
    font-weight: 900 !important;
    box-shadow: 0 0 15px rgba(253,185,19,0.3) !important;
}

/* CHAT */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02) !important;
}

[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--csk-yellow) !important;
    box-shadow: 0 0 15px rgba(253,185,19,0.15) !important;
}

.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border-bright) !important;
    color: var(--text-primary) !important;
}

/* SCROLLBAR */
hr { border-color: var(--border-bright) !important; opacity: 0.6 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--csk-yellow); }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

.stRadio label    { color: var(--text-primary) !important; font-weight: 500 !important; }
.stCheckbox label { color: var(--text-primary) !important; }

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] .stRadio label { color: var(--text-primary) !important; font-weight: 600 !important; }

/* PROGRESS BAR */
.stProgress .st-bo {
    background: linear-gradient(90deg, var(--csk-yellow), var(--csk-blue)) !important;
    box-shadow: 0 0 8px rgba(253,185,19,0.3) !important;
}

/* EXPANDER */
.streamlit-expanderHeader {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 8px !important;
}

/* ALERTS */
.stInfo    { background: rgba(0,75,141,0.08) !important; border-color: rgba(0,75,141,0.3) !important; }
.stSuccess { background: rgba(253,185,19,0.1)  !important; border-color: rgba(253,185,19,0.4)  !important; }
.stWarning { background: rgba(253,185,19,0.05)  !important; border-color: rgba(253,185,19,0.25) !important; }

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 10px rgba(253,185,19,0.3); }
    50%       { box-shadow: 0 0 20px rgba(253,185,19,0.6); }
}
.pulsing {
    animation: pulse-glow 2.5s ease-in-out infinite !important;
    border-color: rgba(253,185,19,0.45) !important;
}

/* SELECTION */
::selection { background: rgba(253,185,19,0.25); color: var(--text-primary); }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def init_state():
    defaults = {
        "page": "🏠 Dashboard",
        "chat_history": [],
        "notes": [],
        "tasks": [],
        "agent_thread_id": str(uuid.uuid4()),
        "total_queries": 0,
        "focus_session": None,
        "flash_cards": [],
        "current_flashcard_idx": 0,
        "journal_entries": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM & AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource(show_spinner=False)
def build_agent():
    import wikipedia, random as rnd, requests as req
    _orig = req.get
    def _patched(url, **kw):
        if url.startswith("http://"):
            url = url.replace("http://", "https://")
        return _orig(url, **kw)
    req.get = _patched
    wikipedia.set_user_agent(f"SecondBrainApp_{rnd.randint(1000,9999)}/1.0")
    wikipedia.set_rate_limiting(True)

    llm = get_llm()
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
    weather_api = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=os.getenv("OPENWEATHER_API_KEY") or os.getenv("OPENWEATHERMAP_API_KEY", "")
    )

    @tool
    def get_weather(city: str) -> str:
        """Get real-time weather for any city."""
        return weather_api.run(city)

    @tool
    def get_current_datetime(_: str = "") -> str:
        """Return current date and time."""
        return datetime.datetime.now().strftime("Date: %A, %d %B %Y | Time: %I:%M %p IST")

    @tool
    def calculate(expression: str) -> str:
        """Safely evaluate a mathematical expression like '2**10 + 5*3'."""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    memory = MemorySaver()
    system_prompt = (
        "You are Second Brain — a world-class AI personal knowledge and productivity assistant. "
        "You have persistent memory, real-time tools, and deep expertise. "
        "Always respond in clean markdown with emojis. Be concise, insightful, and actionable. "
        "Use tools proactively. Format with headers and bullet points for readability."
    )
    agent = create_react_agent(llm, tools=[wiki_tool, get_weather, get_current_datetime, calculate],
                               checkpointer=memory, prompt=system_prompt)
    return agent, memory

def ask_agent(query: str) -> str:
    agent, _ = build_agent()
    config = {"configurable": {"thread_id": st.session_state.agent_thread_id}}
    resp = agent.invoke({"messages": [("user", query)]}, config=config)
    st.session_state.total_queries += 1
    return resp["messages"][-1].content

def ask_llm(prompt_text: str, system: str = "You are a helpful AI assistant.") -> str:
    llm = get_llm()
    chain = ChatPromptTemplate.from_messages([("system", system), ("human", "{q}")]) | llm | StrOutputParser()
    return chain.invoke({"q": prompt_text})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem;'>
        <div style='font-size:3rem;'>🌿</div>
        <div style='font-size:1.3rem;font-weight:800;background:linear-gradient(135deg,#10b981,#0d9488,#d97706);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Second Brain AI</div>
        <div style='font-size:0.75rem;color:#4d7c5f;'>Personal Knowledge OS</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div style='font-size:0.7rem;color:#4d7c5f;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>Navigation</div>", unsafe_allow_html=True)
    pages = ["🏠 Dashboard","🤖 AI Chat Agent","📝 Smart Notes","✅ Task Manager",
             "🔬 Research Assistant","💡 Idea Generator","🌤️ Weather & Info",
             "📚 Flashcards","📓 Daily Journal","⏱️ Focus Timer"]
    selected = st.radio("", pages, index=pages.index(st.session_state.page), label_visibility="collapsed")
    st.session_state.page = selected
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='metric-card' style='padding:0.6rem;'><div class='metric-value' style='font-size:1.4rem;'>{st.session_state.total_queries}</div><div class='metric-label' style='font-size:0.65rem;'>Queries</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card' style='padding:0.6rem;'><div class='metric-value' style='font-size:1.4rem;'>{len(st.session_state.notes)}</div><div class='metric-label' style='font-size:0.65rem;'>Notes</div></div>", unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.agent_thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("<div style='color:#4d7c5f;font-size:0.78rem;text-align:center;margin-top:1rem;'>Built with ❤️ | LangGraph · Groq · LLaMA 3.3</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if st.session_state.page == "🏠 Dashboard":
    now = datetime.datetime.now()
    greeting = 'Morning' if now.hour < 12 else 'Afternoon' if now.hour < 17 else 'Evening'

    st.markdown(f"""
    <div class='hero-banner'>
        <div style='font-size:0.8rem;color:#475569;text-transform:uppercase;letter-spacing:0.12em;'>{now.strftime("%A, %d %B %Y")}</div>
        <div class='main-title'>Good {greeting}! 👋</div>
        <div class='main-subtitle'>Your AI-powered Second Brain is ready. What will you create, learn, or solve today?</div>
        <div style='display:flex;gap:0.7rem;flex-wrap:wrap;margin-top:1rem;'>
            <span class='badge badge-green'>🟢 AI Agent Online</span>
            <span class='badge badge-blue'>⚡ Groq Ultra-Fast</span>
            <span class='badge badge-purple'>🧠 Memory Active</span>
            <span class='badge badge-orange'>🛠️ 4 Tools Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (str(st.session_state.total_queries), "AI Queries"),
        (str(len(st.session_state.notes)), "Smart Notes"),
        (str(len(st.session_state.tasks)), "Tasks"),
        (str(len(st.session_state.journal_entries)), "Journal Entries"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4], metrics):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{val}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🚀 Quick Actions")
        qa = st.columns(2)
        actions = [("🤖 Ask AI","🤖 AI Chat Agent"),("📝 New Note","📝 Smart Notes"),
                   ("🔬 Research","🔬 Research Assistant"),("💡 Brainstorm","💡 Idea Generator")]
        for i, (label, target) in enumerate(actions):
            with qa[i % 2]:
                if st.button(label, key=f"qa_{i}", use_container_width=True):
                    st.session_state.page = target; st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📝 Recent Notes")
        if not st.session_state.notes:
            st.markdown("<div style='color:#475569;padding:1rem;border:1px dashed #2d2d3f;border-radius:10px;text-align:center;'>No notes yet — start capturing your ideas! 💭</div>", unsafe_allow_html=True)
        else:
            for note in reversed(st.session_state.notes[-3:]):
                st.markdown(f"""
                <div class='note-card'>
                    <div style='font-weight:600;'>📌 {note['title']}</div>
                    <div style='font-size:0.78rem;color:#475569;margin-top:0.3rem;'>{note['created_at']} · <span class='badge badge-purple'>{note['tag']}</span></div>
                    <div style='font-size:0.88rem;color:#94a3b8;margin-top:0.5rem;'>{note['content'][:150]}{'...' if len(note['content'])>150 else ''}</div>
                </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("### ✅ Active Tasks")
        active_tasks = [t for t in st.session_state.tasks if not t.get("done")][:5]
        if not active_tasks:
            st.markdown("<div style='color:#475569;padding:1rem;border:1px dashed #2d2d3f;border-radius:10px;text-align:center;'>All clear! 🎯</div>", unsafe_allow_html=True)
        else:
            pcolors = {"🔴 High":"#ef4444","🟡 Medium":"#f59e0b","🟢 Low":"#10b981"}
            for task in active_tasks:
                color = pcolors.get(task.get("priority","🟢 Low"),"#10b981")
                st.markdown(f"""<div style='display:flex;align-items:center;gap:0.8rem;padding:0.7rem 1rem;background:#16161f;border:1px solid #2d2d3f;border-radius:10px;margin-bottom:0.4rem;'>
                    <div style='width:8px;height:8px;border-radius:50%;background:{color};'></div>
                    <div style='font-size:0.88rem;'>{task['title']}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        motivations = ["\"The expert in anything was once a beginner.\"","\"Your second brain never sleeps.\"",
                       "\"Build systems, not just goals.\"","\"Capture everything. Curate ruthlessly.\"",
                       "\"Knowledge compounds like interest.\"","\"One insight a day changes everything.\""]
        st.markdown(f"""<div class='module-card' style='padding:1.2rem;'><div style='font-size:1rem;font-style:italic;color:#94a3b8;line-height:1.6;'>{motivations[now.day % len(motivations)]}</div></div>""", unsafe_allow_html=True)

        done_count = sum(1 for t in st.session_state.tasks if t.get("done"))
        score = int(done_count / max(1, len(st.session_state.tasks)) * 100)
        st.markdown("### 📊 Productivity")
        st.progress(score / 100, text=f"Task completion: {score}%")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AI CHAT AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.page == "🤖 AI Chat Agent":
    st.markdown("""<div class='hero-banner' style='padding:1.5rem 2rem;'>
        <div class='main-title' style='font-size:2rem;'>🤖 AI Chat Agent</div>
        <div class='main-subtitle'>LangGraph ReAct · LLaMA 3.3 70B · 4 Live Tools · Persistent Memory</div></div>""", unsafe_allow_html=True)

    tool_cols = st.columns(4)
    for i, (icon, name, status) in enumerate([("🔍","Wikipedia","Live Search"),("🌤️","Weather","Real-time"),("🕐","DateTime","Current"),("🔢","Calculator","Math")]):
        with tool_cols[i]:
            st.markdown(f"<div class='metric-card' style='padding:0.8rem;'><div style='font-size:1.5rem;'>{icon}</div><div style='font-weight:600;font-size:0.85rem;margin-top:0.3rem;'>{name}</div><span class='badge badge-green' style='margin-top:0.4rem;font-size:0.65rem;'>{status}</span></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""<div style='text-align:center;padding:3rem 1rem;'>
            <div style='font-size:3rem;'>🧠</div>
            <div style='color:#94a3b8;font-size:1.1rem;font-weight:500;margin-top:1rem;'>Your AI agent is ready with memory and tools.</div>
            <div style='color:#475569;font-size:0.9rem;margin-top:0.5rem;'>Try: "Weather in Mumbai?" · "Summarize neural networks" · "What's 2^20?"</div></div>""", unsafe_allow_html=True)
        st.markdown("**💡 Quick starts:**")
        sug_cols = st.columns(3)
        suggestions = ["What is the weather in Chennai today?","Explain quantum computing in 5 points",
                       "Help me plan my day efficiently","Who invented the internet?",
                       "Calculate 15% of 85000","Give me 5 productivity tips"]
        for i, sug in enumerate(suggestions):
            with sug_cols[i % 3]:
                if st.button(f"💬 {sug[:35]}...", key=f"sug_{i}", use_container_width=True):
                    with st.spinner("🧠 Thinking..."):
                        response = ask_agent(sug)
                    st.session_state.chat_history.extend([{"role":"user","content":sug},{"role":"assistant","content":response}])
                    st.rerun()
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "🧠"):
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask your Second Brain anything...")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🔍 Thinking..."):
                response = ask_agent(user_input)
            st.markdown(response)
        st.session_state.chat_history.append({"role":"assistant","content":response})
        st.rerun()

    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"]=="assistant":
        if st.button("📌 Save Last Response as Note", use_container_width=True):
            last_q = next((m["content"] for m in reversed(st.session_state.chat_history) if m["role"]=="user"), "AI Response")
            st.session_state.notes.append({"id":str(uuid.uuid4()),"title":last_q[:60],"content":st.session_state.chat_history[-1]["content"],"tag":"🤖 AI Chat","created_at":datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")})
            st.success("✅ Saved to Notes!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMART NOTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.page == "📝 Smart Notes":
    st.markdown("""<div class='hero-banner' style='padding:1.5rem 2rem;'>
        <div class='main-title' style='font-size:2rem;'>📝 Smart Notes</div>
        <div class='main-subtitle'>Capture, organize, and AI-enhance your thoughts instantly</div></div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["✍️ Create Note", "📚 All Notes"])

    with tab1:
        note_title = st.text_input("📌 Note Title", placeholder="e.g., Meeting notes, Research on ML, Book insights...")
        note_content = st.text_area("📄 Content", placeholder="Write your thoughts, ideas, or paste content here...", height=180)
        note_tag = st.selectbox("🏷️ Tag", ["💡 Idea","📚 Learning","🏢 Work","🎯 Goal","🔬 Research","📖 Book","🧪 Experiment","🗺️ Planning"])

        col_save, col_ai = st.columns(2)
        with col_save:
            if st.button("💾 Save Note", use_container_width=True):
                if note_title and note_content:
                    st.session_state.notes.append({"id":str(uuid.uuid4()),"title":note_title,"content":note_content,"tag":note_tag,"created_at":datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")})
                    st.success("✅ Note saved!")
                    st.rerun()
                else:
                    st.warning("Add title and content.")
        with col_ai:
            if st.button("🧠 AI Enhance Note", use_container_width=True):
                if note_content:
                    with st.spinner("✨ Enhancing..."):
                        enhanced = ask_llm(f"Enhance this note with structure, key insights, and clarity:\n\n{note_content}",
                                           "You are a world-class note-taker and knowledge management expert.")
                    st.markdown("#### ✨ AI Enhanced")
                    st.markdown(f"<div class='module-card'>{enhanced}</div>", unsafe_allow_html=True)

    with tab2:
        if not st.session_state.notes:
            st.markdown("<div style='text-align:center;padding:3rem;color:#475569;'><div style='font-size:3rem;'>📝</div><div>No notes yet!</div></div>", unsafe_allow_html=True)
        else:
            search_q = st.text_input("🔍 Search notes...", placeholder="Type to filter...")
            filtered = [n for n in reversed(st.session_state.notes) if search_q.lower() in n['title'].lower() or search_q.lower() in n['content'].lower()]
            st.markdown(f"**{len(filtered)} notes**")
            for note in filtered:
                with st.expander(f"📌 {note['title']} · {note['tag']} · {note['created_at']}"):
                    st.markdown(note["content"])
                    c1, c2 = st.columns([3,1])
                    with c1:
                        if st.button("🧠 Summarize", key=f"sum_{note['id']}"):
                            with st.spinner("Summarizing..."):
                                summary = ask_llm(f"Summarize in 3 bullet points:\n\n{note['content']}")
                            st.info(summary)
                    with c2:
                        if st.button("🗑️ Delete", key=f"del_{note['id']}"):
                            st.session_state.notes = [n for n in st.session_state.notes if n["id"]!=note["id"]]
                            st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TASK MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.page == "✅ Task Manager":
    st.markdown("""<div class='hero-banner' style='padding:1.5rem 2rem;'>
        <div class='main-title' style='font-size:2rem;'>✅ Task Manager</div>
        <div class='main-subtitle'>AI-powered task management with priority intelligence</div></div>""", unsafe_allow_html=True)

    with st.expander("➕ Add New Task", expanded=True):
        tc1, tc2, tc3 = st.columns([3,1,1])
        with tc1: task_title = st.text_input("Task", placeholder="e.g., Review ML paper, Build feature X...", label_visibility="collapsed")
        with tc2: task_priority = st.selectbox("Priority", ["🔴 High","🟡 Medium","🟢 Low"], label_visibility="collapsed")
        with tc3:
            if st.button("➕ Add", use_container_width=True):
                if task_title:
                    st.session_state.tasks.append({"id":str(uuid.uuid4()),"title":task_title,"priority":task_priority,"done":False,"created_at":datetime.datetime.now().strftime("%d %b %Y")})
                    st.rerun()

    if st.button("🤖 AI: Suggest 5 Productivity Tasks", use_container_width=True):
        with st.spinner("Generating tasks..."):
            sugg = ask_llm("Give 5 high-impact productivity tasks for a knowledge worker today. Numbered list, specific and actionable.", "You are a world-class productivity coach.")
        st.markdown(f"<div class='module-card'>{sugg}</div>", unsafe_allow_html=True)

    st.markdown("---")
    active = [t for t in st.session_state.tasks if not t["done"]]
    done_list = [t for t in st.session_state.tasks if t["done"]]
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f"<span class='badge badge-orange'>⏳ Active: {len(active)}</span>", unsafe_allow_html=True)
    with c2: st.markdown(f"<span class='badge badge-green'>✅ Done: {len(done_list)}</span>", unsafe_allow_html=True)
    with c3:
        pct = int(len(done_list)/max(1,len(st.session_state.tasks))*100)
        st.markdown(f"<span class='badge badge-blue'>📊 {pct}% Complete</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    porder = {"🔴 High":0,"🟡 Medium":1,"🟢 Low":2}
    sorted_tasks = sorted(st.session_state.tasks, key=lambda x:(x["done"],porder.get(x.get("priority","🟢 Low"),2)))
    pcolors = {"🔴 High":"#ef4444","🟡 Medium":"#f59e0b","🟢 Low":"#10b981"}

    for task in sorted_tasks:
        color = pcolors.get(task.get("priority","🟢 Low"),"#10b981")
        done_style = "opacity:0.5;text-decoration:line-through;" if task["done"] else ""
        cc1, cc2, cc3, cc4 = st.columns([0.5,4,1,0.5])
        with cc1:
            checked = st.checkbox("", value=task["done"], key=f"chk_{task['id']}")
            if checked != task["done"]:
                task["done"] = checked; st.rerun()
        with cc2:
            st.markdown(f"<div style='{done_style}font-size:0.9rem;line-height:2.2;'>{task['title']}</div>", unsafe_allow_html=True)
        with cc3:
            st.markdown(f"<div style='margin-top:0.4rem;font-size:0.8rem;color:{color};font-weight:600;'>{task.get('priority','')}</div>", unsafe_allow_html=True)
        with cc4:
            if st.button("🗑", key=f"tdel_{task['id']}"):
                st.session_state.tasks = [t for t in st.session_state.tasks if t["id"]!=task["id"]]; st.rerun()

    if done_list and st.button("🗑️ Clear Completed"):
        st.session_state.tasks = [t for t in st.session_state.tasks if not t["done"]]; st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RESEARCH ASSISTANT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.page == "🔬 Research Assistant":
    st.markdown("""<div class='hero-banner' style='padding:1.5rem 2rem;'>
        <div class='main-title' style='font-size:2rem;'>🔬 Research Assistant</div>
        <div class='main-subtitle'>Deep research with Wikipedia integration + AI synthesis</div></div>""", unsafe_allow_html=True)

    r_tabs = st.tabs(["🔍 Research Topic","📊 Summarize Text","🆚 Compare Concepts","❓ Deep Q&A"])

    with r_tabs[0]:
        topic = st.text_input("🔬 Research Topic", placeholder="e.g., Quantum computing, CRISPR, Transformer architecture...")
        depth = st.select_slider("Depth", options=["Quick Overview","Standard","Deep Dive","Expert Analysis"], value="Standard")
        if st.button("🚀 Research Now", use_container_width=True):
            if topic:
                depth_map = {"Quick Overview":"Give a 3-bullet overview","Standard":"Structured summary with key concepts","Deep Dive":"Comprehensive deep-dive with history, applications, limitations","Expert Analysis":"Expert technical analysis with research frontiers and key contributors"}
                with st.spinner(f"🔬 Researching '{topic}'..."):
                    result = ask_agent(f"Research: {topic}. {depth_map[depth]}. Use Wikipedia. Format with markdown headers.")
                st.markdown(f"<div class='module-card'>{result}</div>", unsafe_allow_html=True)
                if st.button("📌 Save to Notes"):
                    st.session_state.notes.append({"id":str(uuid.uuid4()),"title":f"Research: {topic}","content":result,"tag":"🔬 Research","created_at":datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")})
                    st.success("Saved!")

    with r_tabs[1]:
        text_in = st.text_area("Paste text to summarize:", height=200)
        style = st.radio("Style", ["📋 Bullet Points","📄 Paragraph","🎯 Key Takeaways","👶 ELI5"], horizontal=True)
        if st.button("📊 Summarize", use_container_width=True):
            if text_in:
                smap = {"📋 Bullet Points":"5-7 bullet points","📄 Paragraph":"2-3 concise paragraphs","🎯 Key Takeaways":"5 numbered key takeaways","👶 ELI5":"Explain like I'm 5 with simple words"}
                with st.spinner("Summarizing..."):
                    out = ask_llm(f"Summarize as {smap[style]}:\n\n{text_in}", "You are an expert at understanding and synthesizing complex information.")
                st.markdown(f"<div class='module-card'>{out}</div>", unsafe_allow_html=True)

    with r_tabs[2]:
        ca, cb = st.columns(2)
        with ca: concept_a = st.text_input("Concept A", placeholder="e.g., Python")
        with cb: concept_b = st.text_input("Concept B", placeholder="e.g., Rust")
        if st.button("🆚 Compare", use_container_width=True):
            if concept_a and concept_b:
                with st.spinner("Analyzing..."):
                    comp = ask_llm(f"Compare '{concept_a}' vs '{concept_b}': Overview, Key Differences (table), Pros/Cons, Use Cases, When to choose which.", "You are an expert technical analyst. Use markdown tables.")
                st.markdown(f"<div class='module-card'>{comp}</div>", unsafe_allow_html=True)

    with r_tabs[3]:
        question = st.text_input("Your Question", placeholder="Ask anything deep and complex...")
        if st.button("🎯 Get Expert Answer", use_container_width=True):
            if question:
                with st.spinner("🧠 Generating expert answer..."):
                    ans = ask_agent(f"Answer with expert depth, using tools if needed: {question}")
                st.markdown(f"<div class='module-card'>{ans}</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IDEA GENERATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.page == "💡 Idea Generator":
    st.markdown("""<div class='hero-banner' style='padding:1.5rem 2rem;'>
        <div class='main-title' style='font-size:2rem;'>💡 Idea Generator</div>
        <div class='main-subtitle'>AI-powered brainstorming, creativity amplification & innovation engine</div></div>""", unsafe_allow_html=True)

    i_tabs = st.tabs(["🧠 Brainstorm","🚀 Startup Ideas","📖 Content Ideas","🔧 Problem Solver"])

    with i_tabs[0]:
        topic_idea = st.text_input("Topic / Domain", placeholder="e.g., EdTech, fitness app for busy professionals...")
        num_ideas = st.slider("Number of ideas", 5, 20, 10)
        creativity = st.select_slider("Creativity", ["Practical","Balanced","Wild","Moonshot"], value="Balanced")
        if st.button("⚡ Generate Ideas", use_container_width=True):
            if topic_idea:
                cmap = {"Practical":"practical, immediately implementable","Balanced":"mix of practical and innovative","Wild":"bold, unconventional, disruptive","Moonshot":"audacious moonshot ideas that could change the world"}
                with st.spinner("💡 Generating..."):
                    ideas = ask_llm(f"Generate {num_ideas} {cmap[creativity]} ideas about: {topic_idea}. Numbered list with 1-line description each.", "You are a world-class innovation strategist.")
                st.markdown(f"<div class='module-card'>{ideas}</div>", unsafe_allow_html=True)

    with i_tabs[1]:
        domain = st.text_input("Domain / Industry", placeholder="e.g., Healthcare AI, Climate Tech...")
        audience = st.text_input("Target Audience", placeholder="e.g., College students, Remote workers...")
        budget = st.selectbox("Budget", ["Bootstrapped (₹0-1L)","Seed (₹1L-10L)","Funded (₹10L+)"])
        if st.button("🚀 Generate Startup Ideas", use_container_width=True):
            if domain:
                with st.spinner("Generating startup ideas..."):
                    ideas = ask_llm(f"Generate 5 startup ideas for {domain} targeting {audience or 'general users'} with {budget} budget. Include: Name, Pitch, Problem, Revenue model, Unique advantage.", "You are a top VC advisor and entrepreneur.")
                st.markdown(f"<div class='module-card'>{ideas}</div>", unsafe_allow_html=True)

    with i_tabs[2]:
        niche = st.text_input("Your Niche", placeholder="e.g., Python programming, fitness, personal finance...")
        platform = st.selectbox("Platform", ["YouTube","LinkedIn","Twitter/X","Blog","Instagram","Podcast","Newsletter"])
        if st.button("📝 Generate Content Ideas", use_container_width=True):
            if niche:
                with st.spinner("Generating content ideas..."):
                    ideas = ask_llm(f"Generate 10 viral {platform} content ideas for {niche}. Include title, hook, and format for each.", "You are a top content strategist who grows creators to millions of followers.")
                st.markdown(f"<div class='module-card'>{ideas}</div>", unsafe_allow_html=True)

    with i_tabs[3]:
        problem = st.text_area("Describe your problem:", height=120, placeholder="Explain your challenge in detail...")
        approach = st.radio("Approach", ["First Principles","SCAMPER Method","Design Thinking","Six Thinking Hats"], horizontal=True)
        if st.button("🔧 Solve Problem", use_container_width=True):
            if problem:
                amap = {"First Principles":"Use first principles — break to fundamental truths, rebuild from scratch","SCAMPER Method":"Apply SCAMPER (Substitute,Combine,Adapt,Modify,Put to other uses,Eliminate,Reverse)","Design Thinking":"Apply Design Thinking (Empathize,Define,Ideate,Prototype,Test)","Six Thinking Hats":"Apply Six Thinking Hats (White/Data,Red/Emotion,Black/Caution,Yellow/Optimism,Green/Creativity,Blue/Process)"}
                with st.spinner("🔧 Solving..."):
                    sol = ask_llm(f"Problem: {problem}\n\n{amap[approach]}. Provide actionable, specific solutions.", "You are a world-class problem-solver and systems thinker.")
                st.markdown(f"<div class='module-card'>{sol}</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEATHER & INFO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.page == "🌤️ Weather & Info":
    st.markdown("""<div class='hero-banner' style='padding:1.5rem 2rem;'>
        <div class='main-title' style='font-size:2rem;'>🌤️ Weather & Info</div>
        <div class='main-subtitle'>Real-time weather + Wikipedia knowledge at your fingertips</div></div>""", unsafe_allow_html=True)

    w_tab, wiki_tab = st.tabs(["🌤️ Live Weather","📖 Wikipedia"])

    with w_tab:
        city_input = st.text_input("🏙️ City Name", placeholder="e.g., Chennai, Mumbai, Delhi, New York...")
        quick_cities = ["Chennai","Mumbai","Delhi","Bangalore","Hyderabad","London","New York","Tokyo"]
        city_cols = st.columns(4)
        for i, city in enumerate(quick_cities):
            with city_cols[i%4]:
                if st.button(city, key=f"city_{i}", use_container_width=True):
                    with st.spinner(f"Fetching weather for {city}..."):
                        wr = ask_agent(f"What is the current weather in {city}? Include temp in Celsius, humidity, wind speed, condition, and activity/clothing advice.")
                    st.markdown(f"<div class='module-card'>{wr}</div>", unsafe_allow_html=True)

        if st.button("🌤️ Get Weather", use_container_width=True):
            if city_input:
                with st.spinner(f"Fetching weather for {city_input}..."):
                    wr = ask_agent(f"What is the current weather in {city_input}? Include temp in Celsius, humidity, wind speed, condition, and activity/clothing advice.")
                st.markdown(f"<div class='module-card'>{wr}</div>", unsafe_allow_html=True)

    with wiki_tab:
        wiki_query = st.text_input("🔍 Search Wikipedia", placeholder="e.g., Artificial Intelligence, Black holes...")
        if st.button("🔍 Search", use_container_width=True):
            if wiki_query:
                with st.spinner(f"Searching Wikipedia for '{wiki_query}'..."):
                    wr = ask_agent(f"Search Wikipedia for: {wiki_query}. Provide comprehensive summary with key facts, history, and significance. Use clear sections.")
                st.markdown(f"<div class='module-card'>{wr}</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FLASHCARDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.page == "📚 Flashcards":
    st.markdown("""<div class='hero-banner' style='padding:1.5rem 2rem;'>
        <div class='main-title' style='font-size:2rem;'>📚 AI Flashcards</div>
        <div class='main-subtitle'>AI-generated flashcards for accelerated learning</div></div>""", unsafe_allow_html=True)

    fc_tab1, fc_tab2 = st.tabs(["⚡ Generate Flashcards","🃏 Study Mode"])

    with fc_tab1:
        fc_topic = st.text_input("Topic to Learn", placeholder="e.g., Machine Learning, Python OOP, Indian History...")
        num_cards = st.slider("Number of Flashcards", 3, 15, 6)
        if st.button("⚡ Generate Flashcards", use_container_width=True):
            if fc_topic:
                with st.spinner("Generating flashcards..."):
                    fc_resp = ask_llm(
                        f"Create {num_cards} flashcards on: {fc_topic}. Return ONLY a valid JSON array like: [{{\"question\":\"...\",\"answer\":\"...\"}}]. No other text.",
                        "You are an expert educator. Create clear flashcards."
                    )
                import re
                json_match = re.search(r'\[.*?\]', fc_resp, re.DOTALL)
                if json_match:
                    try:
                        cards = json.loads(json_match.group())
                        st.session_state.flash_cards.extend(cards)
                        st.success(f"✅ {len(cards)} flashcards generated!")
                        st.rerun()
                    except:
                        st.error("Could not parse flashcards. Try again.")
                else:
                    st.error("No valid JSON found. Try again.")

        if st.session_state.flash_cards:
            st.markdown(f"**Total flashcards: {len(st.session_state.flash_cards)}**")

    with fc_tab2:
        if not st.session_state.flash_cards:
            st.markdown("<div style='text-align:center;padding:3rem;color:#475569;'><div style='font-size:3rem;'>🃏</div><div>Generate flashcards first!</div></div>", unsafe_allow_html=True)
        else:
            total = len(st.session_state.flash_cards)
            idx = st.session_state.current_flashcard_idx % total
            card = st.session_state.flash_cards[idx]
            st.markdown(f"**Card {idx+1} of {total}**")
            st.progress((idx+1)/total)
            st.markdown(f"""<div class='module-card pulsing' style='text-align:center;padding:2.5rem;'>
                <div style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:1rem;'>❓ Question</div>
                <div style='font-size:1.3rem;font-weight:600;color:#f8fafc;line-height:1.5;'>{card['question']}</div></div>""", unsafe_allow_html=True)

            with st.expander("👁️ Reveal Answer"):
                st.markdown(f"<div style='padding:1rem;background:rgba(16,185,129,0.05);border-radius:10px;border:1px solid rgba(16,185,129,0.2);'><div style='font-size:0.75rem;color:#10b981;margin-bottom:0.5rem;'>✅ Answer</div><div style='font-size:1rem;color:#f8fafc;line-height:1.6;'>{card['answer']}</div></div>", unsafe_allow_html=True)

            n1,n2,n3 = st.columns([1,2,1])
            with n1:
                if st.button("◀ Prev", use_container_width=True):
                    st.session_state.current_flashcard_idx=(idx-1)%total; st.rerun()
            with n2:
                if st.button("🔀 Random", use_container_width=True):
                    st.session_state.current_flashcard_idx=random.randint(0,total-1); st.rerun()
            with n3:
                if st.button("Next ▶", use_container_width=True):
                    st.session_state.current_flashcard_idx=(idx+1)%total; st.rerun()

            if st.button("🗑️ Clear All Flashcards"):
                st.session_state.flash_cards=[]; st.session_state.current_flashcard_idx=0; st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DAILY JOURNAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.page == "📓 Daily Journal":
    st.markdown("""<div class='hero-banner' style='padding:1.5rem 2rem;'>
        <div class='main-title' style='font-size:2rem;'>📓 Daily Journal</div>
        <div class='main-subtitle'>Reflect, grow, and get AI insights on your thoughts</div></div>""", unsafe_allow_html=True)

    jt1, jt2 = st.tabs(["✍️ Write Entry","📖 Past Entries"])

    with jt1:
        today = datetime.datetime.now().strftime("%A, %d %B %Y")
        st.markdown(f"### ✍️ Journal — {today}")
        mood = st.select_slider("😊 Mood", ["😔 Low","😐 Neutral","🙂 Good","😊 Happy","🤩 Excellent"], value="🙂 Good")
        gratitude = st.text_area("🙏 3 Things I'm Grateful For", placeholder="1. \n2. \n3. ", height=100)
        highlights = st.text_area("⭐ Today's Highlights", placeholder="What went well?", height=100)
        learnings = st.text_area("📚 What I Learned Today", placeholder="Insights, lessons...", height=100)
        tomorrow = st.text_area("🎯 Tomorrow's #1 Intention", placeholder="What's the ONE thing?", height=80)

        jc1, jc2 = st.columns(2)
        with jc1:
            if st.button("💾 Save Entry", use_container_width=True):
                st.session_state.journal_entries.append({"id":str(uuid.uuid4()),"date":today,"mood":mood,"gratitude":gratitude,"highlights":highlights,"learnings":learnings,"tomorrow":tomorrow})
                st.success("✅ Journal entry saved!")
                st.balloons()
        with jc2:
            if st.button("🧠 AI Reflection", use_container_width=True):
                content = f"Mood: {mood}\nGratitude: {gratitude}\nHighlights: {highlights}\nLearnings: {learnings}\nTomorrow: {tomorrow}"
                if any([gratitude, highlights, learnings]):
                    with st.spinner("🧠 Reflecting..."):
                        reflection = ask_llm(f"Based on this journal:\n\n{content}\n\nProvide: 1) Positive reinforcement 2) Pattern recognition 3) Growth insight 4) One powerful question to ponder", "You are a compassionate life coach and psychologist.")
                    st.markdown(f"<div class='module-card'>{reflection}</div>", unsafe_allow_html=True)

    with jt2:
        if not st.session_state.journal_entries:
            st.markdown("<div style='text-align:center;padding:3rem;color:#475569;'><div style='font-size:3rem;'>📓</div><div>No entries yet!</div></div>", unsafe_allow_html=True)
        else:
            for entry in reversed(st.session_state.journal_entries):
                with st.expander(f"📓 {entry['date']} · {entry['mood']}"):
                    if entry.get("gratitude"): st.markdown(f"**🙏 Gratitude:** {entry['gratitude']}")
                    if entry.get("highlights"): st.markdown(f"**⭐ Highlights:** {entry['highlights']}")
                    if entry.get("learnings"): st.markdown(f"**📚 Learnings:** {entry['learnings']}")
                    if entry.get("tomorrow"): st.markdown(f"**🎯 Tomorrow:** {entry['tomorrow']}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOCUS TIMER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.page == "⏱️ Focus Timer":
    st.markdown("""<div class='hero-banner' style='padding:1.5rem 2rem;'>
        <div class='main-title' style='font-size:2rem;'>⏱️ Focus Timer</div>
        <div class='main-subtitle'>Deep work sessions with AI-powered productivity coaching</div></div>""", unsafe_allow_html=True)

    focus_task = st.text_input("What will you work on?", placeholder="e.g., Study ML, Write blog post, Code feature X...")
    fc1, fc2 = st.columns(2)
    with fc1: duration = st.selectbox("Session Duration", ["25 min (Pomodoro)","45 min (Flow)","60 min (Deep Work)","90 min (Ultra Focus)"])
    with fc2: break_time = st.selectbox("Break Time", ["5 min","10 min","15 min","20 min"])

    if focus_task and st.button("🧠 Get AI Focus Tips", use_container_width=True):
        with st.spinner("Generating tips..."):
            tips = ask_llm(f"Give 5 specific focus/productivity tips for: '{focus_task}'. Mental and environmental.", "You are a world-class productivity coach specializing in deep work.")
        st.markdown(f"<div class='module-card'>{tips}</div>", unsafe_allow_html=True)

    dur_map = {"25 min (Pomodoro)":25,"45 min (Flow)":45,"60 min (Deep Work)":60,"90 min (Ultra Focus)":90}
    selected_min = dur_map[duration]

    if st.button("▶️ Start Focus Session", use_container_width=True):
        st.session_state.focus_session = {"task":focus_task or "Deep Work","start_time":datetime.datetime.now(),"duration_min":selected_min}
        st.success(f"🎯 {selected_min}-minute session started! Work on: **{focus_task or 'Deep Work'}**")

    if st.session_state.focus_session:
        sess = st.session_state.focus_session
        elapsed = min(sess["duration_min"], (datetime.datetime.now()-sess["start_time"]).seconds//60)
        remaining = max(0, sess["duration_min"]-elapsed)
        st.markdown(f"""<div class='module-card' style='text-align:center;'>
            <div style='font-size:0.8rem;color:#475569;'>Active Session: {sess['task']}</div>
            <div style='font-size:3rem;font-weight:800;color:#7c3aed;margin:0.5rem 0;'>{remaining:02d}:00</div>
            <div style='font-size:0.85rem;color:#94a3b8;'>Minutes remaining of {sess['duration_min']} min session</div></div>""", unsafe_allow_html=True)
        st.progress(min(1.0, elapsed/sess["duration_min"]))
        if remaining == 0:
            st.success("🎉 Session complete! Take a break!")
            st.balloons()

    st.markdown("---")
    st.markdown("### 🧘 Focus Techniques")
    for icon, name, desc in [("🍅","Pomodoro","25 min work + 5 min break. 4 sessions → 30 min long break."),
                               ("🌊","Flow State","45-90 min deep sessions. Eliminate all notifications."),
                               ("📵","Digital Minimalism","Phone away, blockers on, headphones in."),
                               ("🎵","Binaural Beats","40Hz gamma or brown noise for deep focus.")]:
        st.markdown(f"<div class='module-card' style='padding:1rem 1.2rem;margin-bottom:0.5rem;'><div style='display:flex;align-items:center;gap:0.8rem;'><div style='font-size:1.5rem;'>{icon}</div><div><div style='font-weight:600;'>{name}</div><div style='font-size:0.85rem;color:#94a3b8;'>{desc}</div></div></div></div>", unsafe_allow_html=True)
