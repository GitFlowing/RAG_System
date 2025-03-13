# from langchain_community.vectorstores import Chroma
# from sentence_transformers import SentenceTransformer

# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & efficient model
# vectorstore = Chroma("langchain_store", embedding_model, persist_directory='./langchain_db')
import os
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

# 1️⃣ Load Text Files from "data/" Directory
DATA_PATH = "./data"  # Folder containing .txt files
documents = []

for file_name in os.listdir(DATA_PATH):
    if file_name.endswith(".txt"):  # Only process .txt files
        file_path = os.path.join(DATA_PATH, file_name)
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())

# 2️⃣ Split Large Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Adjust as needed
    chunk_overlap=50,
    length_function=len
)

chunks = text_splitter.split_documents(documents)

# 3️⃣ Convert Text Chunks to Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4️⃣ Store in ChromaDB
vectorstore = Chroma(
    persist_directory="./langchain_db",  # Local DB storage
    embedding_function=embedding_model
)

# Add chunks to vector database
vectorstore.add_documents(chunks)

# 5️⃣ Persist the Database
vectorstore.persist()

print("✅ Text files successfully added to ChromaDB 🚀")
