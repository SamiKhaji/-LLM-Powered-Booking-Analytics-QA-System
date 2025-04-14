import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware
import pickle

# Constants
FAISS_INDEX_PATH = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
CSV_FILE = "data/hotel_bookings.csv"
GROQ_API_KEY = "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"

# Load data
df = pd.read_csv(CSV_FILE)

# Embed text data
def create_docs():
    docs = []
    for _, row in df.iterrows():
        try:
            text = f"""
Hotel Booking:
- Hotel: {row['hotel']}
- ADR: {row['adr']}
- Country: {row['country']}
- Adults: {row['adults']}, Children: {row['children']}, Babies: {row['babies']}
- Lead Time: {row['lead_time']} days
- Booking Status: {'Canceled' if row['is_canceled'] == 1 else 'Confirmed'}
- Month: {row['arrival_date_month']}, Year: {row['arrival_date_year']}
"""
            docs.append(Document(page_content=text))
        except:
            continue
    return docs

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Create or load FAISS
if os.path.exists(FAISS_INDEX_PATH):
    print("✅ FAISS index found. Loading...")
    with open(os.path.join(FAISS_INDEX_PATH, "faiss_store.pkl"), "rb") as f:
        vectordb = pickle.load(f)
else:
    print("🚀 No FAISS index found. Creating...")
    docs = create_docs()
    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(os.path.join(FAISS_INDEX_PATH, "faiss_store.pkl"), "wb") as f:
        pickle.dump(vectordb, f)

# RAG Chain
retriever = vectordb.as_retriever()
llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# FastAPI setup
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Hotel Booking RAG API running ✅"}

@app.post("/ask")
def ask(req: QueryRequest):
    response = qa_chain.run(req.query)
    return {"response": response}
