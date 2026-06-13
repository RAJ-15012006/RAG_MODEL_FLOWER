# 🌸 Flowers Knowledge RAG App

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about flowers using Google Gemini AI and LangChain.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange?logo=google)

---

## 🚀 Live Demo

> Deploy on [Streamlit Community Cloud](https://share.streamlit.io) to get a public URL.

---

## 🧠 How It Works

```
User Question
     ↓
Retrieve relevant chunks from flowers.txt (ChromaDB)
     ↓
Pass context + question to Gemini LLM
     ↓
Get accurate, grounded answer
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🦜 LangChain | RAG pipeline framework |
| 🌐 Google Gemini | LLM + Embeddings |
| 🗄️ ChromaDB | Vector database |
| 🎈 Streamlit | Web UI |
| 📄 flowers.txt | Knowledge base |

---

## ⚙️ Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/RAJ-15012006/RAG_MODEL_FLOWER.git
cd RAG_MODEL_FLOWER
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_google_api_key_here
```
Get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 5. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `GOOGLE_API_KEY` in **Advanced Settings → Secrets**
5. Click **Deploy** 🚀

---

## 📁 Project Structure

```
RAG_MODEL_FLOWER/
├── app.py              # Main Streamlit app
├── flowers.txt         # Knowledge base document
├── requirements.txt    # Python dependencies
├── .env                # API key (not pushed to GitHub)
├── .gitignore          # Ignores .env and venv
└── README.md           # You are here!
```

---

## 💡 Features

- 🔍 Semantic search over flowers.txt using vector embeddings
- 🤖 Powered by Google Gemini (`gemini-flash-latest`)
- 💬 Clean chat interface with Streamlit
- ⚡ Cached vectorstore for fast responses
- 🔒 Secure API key handling via `.env` / Streamlit secrets

---

## 👨‍💻 Author

**RAJ** — Built with ❤️ using LangChain + Gemini + Streamlit
