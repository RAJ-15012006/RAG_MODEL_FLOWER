# ======================================
# LANGCHAIN IMPORTS
# ======================================
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ======================================
# BASIC SETUP
# ======================================
import os
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
[data-testid="stChatInput"] textarea {
    font-family: 'Lora', serif !important;
    background: rgba(253,246,236,0.95) !important;
    border: 2px solid #F48FB1 !important;
    border-radius: 30px !important;
    padding: 0.8rem 1.2rem !important;
    color: #3E2723 !important;
    font-size: 0.97rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #C2185B !important;
    box-shadow: 0 0 12px rgba(194,24,91,0.25) !important;
    outline: none !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #C2185B, #9C27B0) !important;
    border-radius: 50% !important;
    box-shadow: 0 4px 12px rgba(194,24,91,0.4) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stChatInput"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(194,24,91,0.5) !important;
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
    <div class="header-subtitle">Ask anything about flowers — powered by Google Gemini & LangChain</div>
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
# API KEY
# ======================================
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found.")
    st.stop()

# ======================================
# VECTOR DATABASE
# ======================================
@st.cache_resource(show_spinner=False, ttl=3600)
def create_vectorstore(_docs, api_key):
    return Chroma.from_documents(
        _docs,
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
    )

vectorstore = create_vectorstore(docs, api_key)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ======================================
# LLM
# ======================================
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3,
    google_api_key=api_key
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

Context:
{context}

Question: {question}

Answer:
"""
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content

# ======================================
# CHAT INTERFACE
# ======================================
if "history" not in st.session_state:
    st.session_state.history = []

# Render chat history
chat_html = '<div class="chat-wrapper">'
for q, a in st.session_state.history:
    chat_html += f"""
    <div class="user-row">
        <div class="user-bubble">{q}</div>
        <div class="user-avatar">🌷</div>
    </div>
    <div class="bot-row">
        <div class="bot-avatar">🤖</div>
        <div class="bot-bubble">{a}</div>
    </div>
    """
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# Chat input
user_question = st.chat_input("🌺 Ask a question about flowers...")

if user_question:
    with st.spinner("🌱 Searching the garden of knowledge..."):
        answer = ask_rag(user_question)
    st.session_state.history.append((user_question, answer))
    st.rerun()
