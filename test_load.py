from vectorstore_store import load_vectorstore
try:
    db = load_vectorstore()
    print("Vectorstore loaded successfully:", db.index.ntotal)
    
    from core.retrieval import make_retriever
    retriever = make_retriever(db)
    print("Retriever made")

    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=400, groq_api_key="test")
    print("ChatGroq loaded")
except Exception as e:
    import traceback
    traceback.print_exc()
