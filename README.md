# Checking different vector data base for RAG system

In this code we tested different vector data base like chromaDB,
langchain and faiss. The embedding model is all-MiniLM-L6-v2. The data are simple
txt-files. The vector database will be created and saved locally in /RAG/chroma_db ,
 /RAG/langchain_db or faiss_index.index. We measured the time for calling the
3 different vector database for figuring out, which one gives the fastest response.
