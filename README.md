## 📝 Short Report – Unified Hotel Booking Insights & QA App
🛠️ Project Overview
This project presents a unified web application that offers:

📊 Hotel Booking Analytics Dashboard – for understanding trends, cancellations, and customer behavior.

🤖 AI-Powered Q&A Assistant – to answer natural language questions using a Retrieval-Augmented Generation (RAG) pipeline powered by LLMs.

We integrated FastAPI for backend intelligence and Streamlit for an interactive, visual frontend.

📌 Implementation Choices
1. Technology Stack
FastAPI (app2.py): A standalone FastAPI app that handles document embedding, vector search via FAISS, and integrates with Groq’s LLaMA3 for responses.

Streamlit (app.py): Initially used for analytics visualizations like cancellation rates, booking trends, and comparisons.

Notebook (main.ipynb): Served as a scratchpad for testing embeddings, LangChain chains, model selection (like ChatGroq), and debugging.

Streamlit Integration (sa.py): Final production Streamlit app that combines both the analytics dashboard and the AI-powered Q&A assistant into a single user interface.

LangChain + HuggingFace Embeddings: For document chunking, semantic search, and RAG-based responses.

FAISS: Chosen over ChromaDB for faster indexing, better retrieval, and seamless LangChain compatibility.

Groq + LLaMA3 70B: Delivers fast, accurate responses to user questions.

2. Frontend Navigation
Streamlit does not support direct URL paths like /ask or /analytics.

✅ Solution: Sidebar radio buttons used to switch views inside sa.py, simulating page navigation while keeping a unified experience.

Outputs:
/analytics
![image](https://github.com/user-attachments/assets/b6c5e43e-a540-4921-9743-0aca8a528d1e)
![image](https://github.com/user-attachments/assets/8c95bfd0-ae63-45d7-8b7f-8b6d79dc1338)
![image](https://github.com/user-attachments/assets/e3034018-f074-49b2-9246-3b8a729101a5)


/ask
![image](https://github.com/user-attachments/assets/125987a7-83af-4c89-809e-c8d7d523a52c)
![image](https://github.com/user-attachments/assets/1870b0eb-814a-47e7-9626-5e3a272184d1)


⚠️ Challenges Faced
1. Routing Issues
Native page routing (like /ask, /analytics) isn’t supported in Streamlit.

✅ Solution: Handled with sidebar-based navigation in sa.py.

2. ChromaDB Limitations
Initially used ChromaDB, but it didn’t meet expectations:

Slower query times.

Less control over retrieval quality.

✅ Solution: Replaced with FAISS, which worked well with LangChain and supported scalable vector search.

3. Backend–Frontend Integration
Coordinating FastAPI (app2.py) with Streamlit (sa.py) needed careful endpoint design and testing.

✅ Solution: Used requests in Streamlit to call FastAPI endpoints for answering questions.

4. Embedding Overhead
On-the-fly embeddings during each question posed performance issues.

✅ Solution: Cached vectorstore indexes and loaded them at runtime to improve speed.

5. LLM Response Accuracy
Generic/hallucinated responses from LLMs when used without context.

✅ Solution: Implemented RAG – embedding dataset into chunks and retrieving relevant context to send to Groq LLaMA3.

6. Data Quality Issues
Columns like children and babies had nulls or outliers.

✅ Solution: Data was preprocessed (null handling, type corrections) before analysis.

