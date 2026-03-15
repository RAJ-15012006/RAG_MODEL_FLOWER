# ======================================
# 2️⃣ LANGCHAIN IMPORTS
# ======================================
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ======================================
# 1️⃣ BASIC SETUP
# ======================================
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv(override=True)

# ======================================
# 3️⃣ STREAMLIT PAGE SETUP
# ======================================
st.set_page_config(
    page_title="Flowers RAG App",
    page_icon="🌸"
)
st.title("🌸 Flowers Knowledge RAG App")
st.write(
    "Ask questions based on the **flowers.txt** document. "
    "The answers are generated using Retrieval-Augmented Generation (RAG)."
)

# ======================================
# 4️⃣ LOAD FLOWERS.TXT FILE
# ======================================
@st.cache_data(show_spinner=True)
def load_data():
    loader = TextLoader("flowers.txt", encoding="utf-8")
    return loader.load()

documents = load_data()

# ======================================
# 5️⃣ SPLIT TEXT INTO CHUNKS
# ======================================
@st.cache_data(show_spinner=True)
def split_documents(_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(_docs)

docs = split_documents(documents)

# ======================================
# 6️⃣ CREATE VECTOR DATABASE
# ======================================
@st.cache_resource(show_spinner=True)
def create_vectorstore(_docs, api_key):
    return Chroma.from_documents(
        _docs,
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
    )

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found.")
    st.stop()
vectorstore = create_vectorstore(docs, api_key)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ======================================
# 7️⃣ INITIALIZE LLM
# ======================================
gemini_key = api_key
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3,
    google_api_key=gemini_key
)

# ======================================
# 8️⃣ HELPER FUNCTIONS
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
# 9️⃣ STREAMLIT CHAT INTERFACE
# ======================================
if "history" not in st.session_state:
    st.session_state.history = []

for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

user_question = st.chat_input("Ask a question about flowers...")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)
    with st.spinner("🌱 Searching knowledge..."):
        answer = ask_rag(user_question)
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.history.append((user_question, answer))