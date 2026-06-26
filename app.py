# ======================================
# LANGCHAIN IMPORTS
# ======================================
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ======================================
# BASIC SETUP
# ======================================
import os
import time
import streamlit as st
from dotenv import load_dotenv
load_dotenv(override=True)

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Flowers Knowledge RAG App",
    page_icon="🌸",
    layout="centered"
)

# ======================================
# CUSTOM CSS
# ======================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lora:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">

<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #FDF6EC !important;
    font-family: 'Lora', serif;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse at 10% 20%, rgba(255,182,193,0.45) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 10%, rgba(216,191,216,0.4) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(255,218,185,0.4) 0%, transparent 55%),
        radial-gradient(ellipse at 90% 70%, rgba(255,213,79,0.25) 0%, transparent 45%),
        radial-gradient(ellipse at 20% 90%, rgba(200,230,201,0.3) 0%, transparent 45%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"] {
    position: relative;
    z-index: 1;
}

/* ── Floating Petals ── */
.petals-container {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.petal {
    position: absolute;
    top: -60px;
    font-size: 1.4rem;
    animation: floatDown linear infinite;
    opacity: 0.75;
}
.petal:nth-child(1)  { left: 5%;  animation-duration: 9s;  animation-delay: 0s;   font-size: 1.2rem; }
.petal:nth-child(2)  { left: 15%; animation-duration: 12s; animation-delay: 2s;   font-size: 1rem;   }
.petal:nth-child(3)  { left: 25%; animation-duration: 10s; animation-delay: 4s;   font-size: 1.5rem; }
.petal:nth-child(4)  { left: 38%; animation-duration: 14s; animation-delay: 1s;   font-size: 1.1rem; }
.petal:nth-child(5)  { left: 50%; animation-duration: 11s; animation-delay: 3s;   font-size: 1.3rem; }
.petal:nth-child(6)  { left: 62%; animation-duration: 13s; animation-delay: 5s;   font-size: 1rem;   }
.petal:nth-child(7)  { left: 73%; animation-duration: 9s;  animation-delay: 0.5s; font-size: 1.4rem; }
.petal:nth-child(8)  { left: 83%; animation-duration: 12s; animation-delay: 2.5s; font-size: 1.2rem; }
.petal:nth-child(9)  { left: 92%; animation-duration: 10s; animation-delay: 6s;   font-size: 1rem;   }
.petal:nth-child(10) { left: 45%; animation-duration: 15s; animation-delay: 7s;   font-size: 1.5rem; }

@keyframes floatDown {
    0%   { transform: translateY(-60px) rotate(0deg);   opacity: 0;    }
    10%  { opacity: 0.75; }
    90%  { opacity: 0.6;  }
    100% { transform: translateY(105vh) rotate(360deg); opacity: 0;    }
}

/* ── Header ── */
.header-container {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    position: relative;
    z-index: 2;
}
.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #C2185B;
    text-shadow: 0 0 20px rgba(194,24,91,0.35), 0 2px 8px rgba(194,24,91,0.2);
    margin-bottom: 0.4rem;
    line-height: 1.2;
}
.header-subtitle {
    font-family: 'Lora', serif;
    font-style: italic;
    font-size: 1.1rem;
    color: #7B5EA7;
    margin-top: 0;
}

/* ── Chat Messages ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    padding: 1rem 0.5rem;
    position: relative;
    z-index: 2;
}

/* User message */
.user-row {
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
    gap: 0.7rem;
    animation: fadeSlideIn 0.4s ease-out;
}
.user-bubble {
    background: linear-gradient(135deg, #C2185B, #F48FB1, #FFCCBC);
    color: #fff;
    padding: 0.85rem 1.2rem;
    border-radius: 20px 20px 4px 20px;
    max-width: 70%;
    font-family: 'Lora', serif;
    font-size: 0.97rem;
    box-shadow:
        0 4px 15px rgba(194,24,91,0.3),
        inset 0 1px 0 rgba(255,255,255,0.35);
    transform: perspective(800px) rotateX(1deg);
    line-height: 1.6;
}
.user-avatar {
    width: 42px; height: 42px;
    border-radius: 50%;
    background: linear-gradient(135deg, #E91E63, #F48FB1);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    border: 2px solid #fff;
    box-shadow: 0 0 10px rgba(194,24,91,0.4);
    flex-shrink: 0;
}

/* Bot message */
.bot-row {
    display: flex;
    justify-content: flex-start;
    align-items: flex-end;
    gap: 0.7rem;
    animation: fadeSlideIn 0.4s ease-out;
}
.bot-bubble {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(156,39,176,0.2);
    color: #3E2723;
    padding: 0.85rem 1.2rem;
    border-radius: 20px 20px 20px 4px;
    max-width: 70%;
    font-family: 'Lora', serif;
    font-size: 0.97rem;
    box-shadow:
        0 4px 20px rgba(156,39,176,0.15),
        0 1px 4px rgba(46,125,50,0.1);
    transform: perspective(800px) rotateX(1deg);
    line-height: 1.6;
}
.bot-avatar {
    width: 42px; height: 42px;
    border-radius: 50%;
    background: linear-gradient(135deg, #2E7D32, #26C6DA);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    border: 2px solid #fff;
    box-shadow: 0 0 0 3px rgba(46,125,50,0.2);
    animation: pulseGlow 2.5s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 3px rgba(46,125,50,0.2); }
    50%       { box-shadow: 0 0 0 6px rgba(46,125,50,0.4); }
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0);    }
}

/* ── Hide default Streamlit chat UI ── */
[data-testid="stChatMessage"] { display: none !important; }

/* ── Input Form ── */
[data-testid="stForm"] {
    background: rgba(194,24,91,0.75) !important;
    border: 2px solid #F48FB1 !important;
    border-radius: 30px !important;
    padding: 0.3rem 0.8rem !important;
    box-shadow: 0 4px 15px rgba(194,24,91,0.15) !important;
}
[data-testid="stTextInput"] input {
    font-family: 'Lora', serif !important;
    background: transparent !important;
    border: none !important;
    color: #FFFFFF !important;
    font-size: 0.97rem !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: rgba(255,255,255,0.6) !important;
}
[data-testid="stTextInput"] input:focus {
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #C2185B, #9C27B0) !important;
    border-radius: 50% !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(194,24,91,0.4) !important;
    color: white !important;
    font-size: 1.2rem !important;
    transition: transform 0.2s ease !important;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(194,24,91,0.5) !important;
}

/* ── Suggestion Chips ── */
.sugg-row {
    margin-top: 0.7rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    font-size: 0.82rem;
    color: #7B5EA7;
}
.sugg-chip {
    background: rgba(194,24,91,0.08);
    border: 1px solid rgba(194,24,91,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.7rem;
    color: #C2185B;
    font-family: 'Lora', serif;
    font-style: italic;
    cursor: default;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #C2185B !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>

<div class="petals-container">
    <div class="petal">🌸</div>
    <div class="petal">🌺</div>
    <div class="petal">🌼</div>
    <div class="petal">🌸</div>
    <div class="petal">🌺</div>
    <div class="petal">🌼</div>
    <div class="petal">🌸</div>
    <div class="petal">🌺</div>
    <div class="petal">🌼</div>
    <div class="petal">🌸</div>
</div>
""", unsafe_allow_html=True)

# ======================================
# HEADER
# ======================================
st.markdown("""
<div class="header-container">
    <div class="header-title">🌸 Flowers Knowledge RAG App</div>
    <div class="header-subtitle">RAG Model for Flowers</div>
</div>
""", unsafe_allow_html=True)

# ======================================
# LOAD FLOWERS.TXT
# ======================================
@st.cache_data(show_spinner=False)
def load_data():
    loader = TextLoader("flowers.txt", encoding="utf-8")
    return loader.load()

documents = load_data()

# ======================================
# SPLIT TEXT INTO CHUNKS
# ======================================
@st.cache_data(show_spinner=False)
def split_documents(_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(_docs)

docs = split_documents(documents)

# ======================================
# API KEY (GROQ)
# ======================================
groq_api_key = None
for key in ["GROQ_API_KEY"]:
    try:
        groq_api_key = st.secrets[key]
        break
    except:
        groq_api_key = os.environ.get(key)
        if groq_api_key:
            break

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# ======================================
# VECTOR DATABASE (HuggingFace Embeddings - FREE, runs locally)
# ======================================
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

@st.cache_resource(show_spinner="🌱 Building flower knowledge base...")
def create_vectorstore(_docs):
    """Create or load a persistent ChromaDB vector store.
    Uses HuggingFace embeddings that run locally - no API calls or quotas needed."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    # If persistent DB already exists, just load it
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        try:
            return Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings
            )
        except Exception:
            pass  # Fall through to rebuild

    # Build from scratch (runs locally, no API needed)
    try:
        return Chroma.from_documents(
            _docs,
            embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
    except Exception as e:
        st.error(f"❌ Vectorstore error: {str(e)}")
        st.stop()

vectorstore = create_vectorstore(docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ======================================
# LLM (GROQ - Fast & Free Tier)
# ======================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    groq_api_key=groq_api_key,
    max_retries=3
)

# ======================================
# RAG FUNCTION
# ======================================
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_rag(question):
    retrieved_docs = retriever.invoke(question)
    context = combine_docs(retrieved_docs)
    prompt = f"""
You are a factual question-answering assistant.
Use ONLY the information provided in the context.
If the answer is not present, say: "I don't know".
At the end of your answer, add a new line and write exactly:
SUGGESTED: <question1> | <question2> | <question3>
Suggested questions must be related to flowers and based on the context.

Context:
{context}

Question: {question}

Answer:
"""
    response = llm.invoke(prompt)

    content = response.content
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content if isinstance(block, dict))

    if "SUGGESTED:" in content:
        parts = content.split("SUGGESTED:")
        answer = parts[0].strip()
        suggestions = [s.strip() for s in parts[1].split("|")]
    else:
        answer = content.strip()
        suggestions = []

    return answer, suggestions

# ======================================
# CHAT INTERFACE
# ======================================
if "history" not in st.session_state:
    st.session_state.history = []

# Render chat history
for entry in st.session_state.history:
    q, a, suggs = entry
    st.markdown(f"""
    <div class="user-row">
        <div class="user-bubble">{q}</div>
        <div class="user-avatar">🌷</div>
    </div>""", unsafe_allow_html=True)
    sugg_html = ""
    if suggs:
        sugg_buttons = "".join(f'<span class="sugg-chip">{s}</span>' for s in suggs)
        sugg_html = f'<div class="sugg-row">💡 {sugg_buttons}</div>'
    st.markdown(f"""
    <div class="bot-row">
        <div class="bot-avatar">🤖</div>
        <div class="bot-bubble">{a}{sugg_html}</div>
    </div>""", unsafe_allow_html=True)

# Chat input form always at bottom
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        user_question = st.text_input("", placeholder="🌺 Ask a question about flowers...", label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("🌸")

if submitted and user_question.strip():
    with st.spinner("🌱 Searching the garden of knowledge..."):
        try:
            answer, suggestions = ask_rag(user_question.strip())
            st.session_state.history.append((user_question.strip(), answer, suggestions))
            st.rerun()
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate_limit" in err_str or "quota" in err_str:
                st.error("🚫 **Rate Limit Hit**")
                st.warning(
                    "Too many requests to Groq API. Please wait a moment and try again.\n\n"
                    "Groq free tier allows ~30 requests/minute. Just wait a few seconds!\n\n"
                    "For more info: [Groq Rate Limits](https://console.groq.com/docs/rate-limits)"
                )
            else:
                st.error(f"❌ Error: {str(e)}")
