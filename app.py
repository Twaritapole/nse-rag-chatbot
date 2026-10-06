import html
import logging
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import streamlit as st


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))

import rag


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("sagealpha.ui")

st.set_page_config(
    page_title="SageAlpha | India Equity Universe",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --navy: #0b1b2b; --navy-2: #11283b; --ink: #172b3d; --muted: #647587;
        --teal: #087f83; --gold: #c6a15b; --line: #dce4e9; --paper: #f5f7f8;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 1.1rem; }
    [data-testid="stSidebar"] { background: #102235; border-right: 1px solid #20384c; }
    [data-testid="stSidebar"] * { color: #e7eef3; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: white; }
    .brandline { display:flex; align-items:center; justify-content:space-between; gap:1rem;
        padding:.45rem 0 1.2rem; border-bottom:1px solid var(--line); margin-bottom:1.15rem; }
    .brand { color:var(--navy); font-size:1.12rem; font-weight:800; letter-spacing:.16em; }
    .brand-sub { display:block; margin-top:.18rem; color:var(--muted); font-size:.64rem;
        font-weight:700; letter-spacing:.2em; }
    .header-status { color:#35676a; font-size:.7rem; font-weight:750; letter-spacing:.13em; }
    .hero { position:relative; overflow:hidden; background:linear-gradient(110deg,#0b1b2b 0%,#122d42 76%,#17434b 100%);
        color:white; padding:2rem 2.2rem; border-radius:9px; border:1px solid #244256;
        box-shadow:0 10px 28px rgba(18,40,59,.10); }
    .hero:after { content:""; position:absolute; right:5%; top:-52px; width:210px; height:210px;
        border:1px solid rgba(174,215,212,.18); border-radius:50%; box-shadow:0 0 0 27px rgba(174,215,212,.035),0 0 0 55px rgba(174,215,212,.025); }
    .hero-kicker { color:#a8d5d1; font-size:.68rem; font-weight:800; letter-spacing:.22em; }
    .hero h1 { color:white; font-size:clamp(1.8rem,4vw,2.7rem); line-height:1.12; margin:.55rem 0 .35rem; }
    .hero h2 { color:#e6edf1; font-size:1.12rem; font-weight:500; margin:0 0 .8rem; }
    .hero p { color:#c4d1d9; max-width:700px; font-size:.96rem; margin:0; }
    .hero-badge { display:inline-block; margin-top:1.15rem; padding:.34rem .6rem; color:#c8e7e3;
        border:1px solid rgba(168,213,209,.4); border-radius:3px; font-size:.62rem; font-weight:750; letter-spacing:.16em; }
    .section-kicker { color:var(--teal); font-size:.67rem; font-weight:800; letter-spacing:.18em; margin:1.75rem 0 .28rem; }
    .section-title { color:var(--navy); font-size:1.34rem; font-weight:760; margin:0; }
    .section-copy { color:var(--muted); font-size:.9rem; margin:.25rem 0 1rem; }
    .chat-panel { background:white; border:1px solid var(--line); border-top:3px solid var(--teal);
        border-radius:7px; padding:1.25rem 1.35rem; box-shadow:0 5px 16px rgba(24,45,63,.045); }
    .ai-label { color:#47717a; font-size:.65rem; letter-spacing:.17em; font-weight:800; margin-bottom:.6rem; }
    .source-card { min-height:110px; background:#fff; border:1px solid var(--line); border-left:3px solid var(--gold);
        border-radius:5px; padding:.8rem .9rem; margin:.2rem 0 .6rem; }
    .source-company { color:var(--navy); font-size:.91rem; font-weight:750; }
    .source-meta { color:var(--muted); font-size:.78rem; line-height:1.5; margin-top:.3rem; overflow-wrap:anywhere; }
    .trust-note { color:#597477; background:#edf5f3; border:1px solid #d3e6e2; padding:.7rem .9rem;
        border-radius:4px; font-size:.78rem; margin-top:.8rem; }
    .company-tile { background:white; border:1px solid var(--line); border-radius:5px; padding:.85rem 1rem; }
    .company-name { color:var(--navy); font-weight:750; }
    .company-years { color:var(--muted); font-size:.75rem; margin-top:.2rem; }
    .footer { border-top:1px solid var(--line); margin-top:2.2rem; padding:1rem 0 .5rem; color:var(--muted); font-size:.75rem; }
    div.stButton > button[kind="primary"] { background:#087f83; border:1px solid #087f83; color:white;
        font-weight:750; letter-spacing:.04em; min-height:2.8rem; border-radius:4px; }
    div.stButton > button[kind="primary"]:hover { background:#066c70; border-color:#066c70; }
    div[data-testid="stTextArea"] textarea { border-color:#cbd7df; background:#fbfcfd; }
    div[data-testid="stTextArea"] textarea:focus { border-color:var(--teal); box-shadow:0 0 0 1px var(--teal); }
    @media (max-width:700px) {
        [data-testid="stMainBlockContainer"] { padding-left:1rem; padding-right:1rem; }
        .hero { padding:1.4rem; } .hero:after { right:-100px; } .brandline { align-items:flex-start; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_rag_resources():
    """Load the existing RAG components once per Streamlit process."""
    return rag.load_rag_resources()


@st.cache_data(show_spinner=False)
def get_dataset_coverage():
    """Discover company folders and FY labels from the existing chunk filenames."""
    companies = []
    all_years = set()
    if not rag.CHUNKS_FOLDER.exists():
        return companies, []

    year_pattern = re.compile(r"FY\s*(20\d{2})[-–](20\d{2}|\d{2})", re.I)
    folders = sorted(path for path in rag.CHUNKS_FOLDER.iterdir() if path.is_dir())
    for folder in folders:
        chunk_files = list(folder.glob("*.json"))
        if not chunk_files:
            continue
        record = {"name": folder.name.replace("_", " "), "years": set()}
        for chunk_file in chunk_files:
            match = year_pattern.search(chunk_file.name)
            if match:
                start, end = match.groups()
                year = f"FY{start}-{end[-2:]}"
                record["years"].add(year)
                all_years.add(year)
        record["years"] = sorted(record["years"], key=rag.year_end)
        companies.append(record)
    return companies, sorted(all_years, key=rag.year_end)


@st.cache_data(ttl=15, show_spinner=False)
def check_local_ollama():
    """Check the configured local Ollama service; this is not a market-data check."""
    tags_url = rag.OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=0.4) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def queue_example(question):
    st.session_state["question_input"] = question


def nice_question_type(value):
    labels = {
        "single-fact": "SINGLE FACT", "comparison": "COMPARISON", "trend": "TREND",
        "calculation": "CALCULATION", "multi-year": "MULTI-YEAR", "source": "SOURCE LOOKUP",
        "highest-lowest": "HIGHEST / LOWEST", "managerial": "FINANCIAL ANALYSIS",
        "summary": "SUMMARY", "future": "FUTURE REQUEST", "investment": "INVESTMENT REQUEST",
        "out-of-scope": "OUT OF SCOPE",
    }
    return labels.get(value, str(value or "FINANCIAL QUESTION").replace("_", " ").upper())


def friendly_error_message(error):
    """Give the user a useful next step without exposing a Python traceback."""
    message = str(error).casefold()
    if isinstance(error, ImportError):
        return "A required RAG package is missing. Check the project's virtual environment and installed dependencies."
    if isinstance(error, FileNotFoundError):
        return "The chatbot could not find its report chunks or saved FAISS index. Check the data and faiss_index folders."
    if isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError)) or (
        "ollama" in message
        and any(part in message for part in ("connection", "refused", "unavailable", "timed out"))
    ):
        return f"The local AI model is not responding. Start Ollama and run `ollama run {rag.OLLAMA_MODEL}`, then ask again."
    if "model" in message and ("not found" in message or "404" in message):
        return f"The configured local model is unavailable. Run `ollama run {rag.OLLAMA_MODEL}` and retry."
    if isinstance(error, MemoryError) or "allocate" in message or "out of memory" in message:
        return "Your computer ran out of memory while processing this question. Close other large applications and try again."
    return "Unable to process this question right now. Please try again or rephrase it. The technical details were written to the terminal."


def render_sources(sources):
    if not sources:
        return
    st.markdown('<div class="section-kicker">SOURCES</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Report evidence used for this response</div>', unsafe_allow_html=True)
    columns = st.columns(min(3, len(sources)))
    for index, source in enumerate(sources):
        details = []
        for field in ("financial_year", "source_pdf"):
            if source.get(field):
                details.append(str(source[field]))
        if source.get("page_number") is not None:
            details.append(f"PDF page {source['page_number']}")
        if source.get("basis"):
            details.append(str(source["basis"]).title())
        elif source.get("section"):
            details.append(str(source["section"]))
        company = html.escape(str(source.get("company") or "Annual report source"))
        meta = html.escape(" · ".join(details))
        with columns[index % len(columns)]:
            st.markdown(
                f'<div class="source-card"><div class="source-company">{company}</div>'
                f'<div class="source-meta">{meta}</div></div>',
                unsafe_allow_html=True,
            )


def render_turn(turn):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(
            f'<div class="ai-label">SAGEALPHA AI &nbsp; / &nbsp; '
            f'{html.escape(nice_question_type(turn.get("question_type")))}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(turn["answer"])
        render_sources(turn.get("sources", []))
        st.markdown(
            '<div class="trust-note"><b>GROUNDED IN FINANCIAL REPORT DATA</b><br>'
            'Responses are generated from annual-report evidence available in this dataset.</div>',
            unsafe_allow_html=True,
        )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "previous_analysis" not in st.session_state:
    st.session_state.previous_analysis = None
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

companies, financial_years = get_dataset_coverage()
ollama_online = check_local_ollama()

with st.sidebar:
    st.markdown('<div class="brand">SAGEALPHA</div><div class="brand-sub">INDIA EQUITY UNIVERSE</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**ABOUT**")
    st.caption("AI-powered financial research using annual-report data loaded in this project.")
    st.markdown("**DATA COVERAGE**")
    st.metric("Companies", len(companies))
    year_label = f"{financial_years[0]} — {financial_years[-1]}" if financial_years else "Not available"
    st.caption(f"Financial years: {year_label}")
    st.caption("Statements: Balance Sheet · Profit & Loss · Cash Flow")
    st.markdown("**AI STACK**")
    st.caption("FAISS · BM25 · Sentence Transformers")
    st.caption(f"Ollama · {rag.OLLAMA_MODEL}")
    state = "LOCAL AI ONLINE" if ollama_online else "LOCAL AI OFFLINE"
    color = "#8dd6b0" if ollama_online else "#e5b58e"
    st.markdown(
        f'<div style="margin-top:.8rem;color:{color};font-size:.7rem;font-weight:800;letter-spacing:.12em">'
        f'● {state}</div><div style="color:#aebdc8;font-size:.72rem;margin-top:.25rem">Annual-report research · no live market feed</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("CLEAR CHAT HISTORY", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.previous_analysis = None
        st.rerun()

st.markdown(
    '<div class="brandline"><div><span class="brand">SAGEALPHA</span>'
    '<span class="brand-sub">INDIA EQUITY UNIVERSE</span></div>'
    f'<div class="header-status">AI FINANCIAL RESEARCH &nbsp; · &nbsp; '
    f'● {"AI ONLINE" if ollama_online else "AI OFFLINE"} &nbsp; · &nbsp; REPORT DATA</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<section class="hero"><div class="hero-kicker">INDIA EQUITY UNIVERSE</div>'
    '<h1>Financial Intelligence, Powered by AI</h1>'
    '<p>Research Indian companies using evidence-grounded AI and historical financial-report data.</p>'
    '<span class="hero-badge">FINANCIAL AI</span></section>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">ASK FINANCIAL AI</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Your research desk</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">Ask about companies, financial years, metrics, comparisons and historical trends.</div>', unsafe_allow_html=True)

with st.form("sagealpha_question_form", clear_on_submit=True):
    st.markdown('<div class="ai-label">SAGEALPHA FINANCIAL AI</div>', unsafe_allow_html=True)
    question_value = st.text_area(
        "Financial research question",
        placeholder="Ask about a company, financial year, metric, comparison or trend...",
        key="question_input", label_visibility="collapsed", height=105,
    )
    st.caption("Example: How did the total assets of TCS change from FY2020-21 to FY2024-25?")
    submitted = st.form_submit_button("ASK SAGEALPHA AI", type="primary", use_container_width=True)

if submitted:
    question = question_value.strip()
    if not question:
        st.warning("Enter a financial question to continue.")
    else:
        try:
            with st.status("ANALYZING QUESTION", expanded=True) as status:
                st.write("Loading the existing financial research system if needed.")
                resources = get_rag_resources()
                st.write("Retrieving annual-report evidence and preparing the response.")
                response = rag.answer_question(question, resources, st.session_state.previous_analysis)
                if not isinstance(response, dict) or not str(response.get("answer", "")).strip():
                    raise RuntimeError("The RAG backend returned no answer text.")
                status.update(label="ANALYSIS COMPLETE", state="complete", expanded=False)
            st.session_state.chat_history.insert(0, {
                "question": question,
                "answer": response["answer"],
                "question_type": response.get("question_type"),
                "sources": response.get("sources", []),
            })
            st.session_state.previous_analysis = response.get("analysis")
        except Exception as error:
            LOGGER.exception("SageAlpha request failed")
            st.session_state.chat_history.append({
                "question": question,
                "answer": friendly_error_message(error),
                "question_type": "error", "sources": [],
            })
            st.session_state.previous_analysis = None

if st.session_state.chat_history:
    st.markdown('<div class="section-kicker">RESEARCH SESSION</div>', unsafe_allow_html=True)
    for turn in st.session_state.chat_history:
        render_turn(turn)

st.markdown('<div class="section-kicker">EXPLORE FINANCIAL QUESTIONS</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Start with a focused question</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">Select an example to place it in the question box.</div>', unsafe_allow_html=True)
example_questions = [
    "What were the total assets of TCS in FY2024-25?",
    "Compare the total assets of TCS and Reliance Industries in FY2024-25.",
    "How did TCS revenue change from FY2020-21 to FY2024-25?",
    "Compare Infosys and TCS revenue in FY2024-25.",
    "What was HDFC Bank's profit in FY2024-25?",
    "How did Reliance Industries' total assets change over the available years?",
    "Give me the source and page for TCS total assets in FY2024-25.",
]
for start in range(0, len(example_questions), 2):
    columns = st.columns(2)
    for offset, question in enumerate(example_questions[start:start + 2]):
        with columns[offset]:
            st.button(
                question, key=f"example_{start + offset}", on_click=queue_example,
                args=(question,), use_container_width=True,
            )

st.markdown('<div class="section-kicker">INDIA EQUITY UNIVERSE</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Companies in the research dataset</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">Company coverage is discovered from the project’s chunked annual-report data.</div>', unsafe_allow_html=True)
if companies:
    columns = st.columns(min(5, len(companies)))
    for index, company in enumerate(companies):
        years = company["years"]
        period = f"{years[0]} — {years[-1]}" if years else "Years not identified"
        with columns[index % len(columns)]:
            st.markdown(
                f'<div class="company-tile"><div class="company-name">{html.escape(company["name"])}</div>'
                f'<div class="company-years">{html.escape(period)}</div></div>',
                unsafe_allow_html=True,
            )
else:
    st.info("No company folders with chunked report data were found. Add chunked annual reports and reload this page.")

st.markdown(
    '<div class="footer"><b>SAGEALPHA</b> &nbsp; · &nbsp; INDIA EQUITY UNIVERSE<br>'
    '<span>AI-powered financial research from annual-report data.</span></div>',
    unsafe_allow_html=True,
)
