import os
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
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
        documents.extend(loader.load())  # Add documents to the list

# 2️⃣ Split Large Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Adjust chunk size as needed
    chunk_overlap=50,  # Adjust overlap for better context
    length_function=len
)

chunks = text_splitter.split_documents(documents)  # Split documents into chunks

# 3️⃣ Convert Text Chunks to Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert documents into embeddings
embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])

# Convert embeddings into a numpy array (FAISS requires numpy arrays)
embedding_matrix = np.array(embeddings).astype("float32")  # FAISS requires float32 dtype

# 4️⃣ Use FAISS for storing and querying
# Create a FAISS index for similarity search
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance is used here

# Add the embeddings to the index
index.add(embedding_matrix)

# Save the FAISS index to disk (persistently)
faiss.write_index(index, "./faiss_index.index")

print("✅ FAISS index successfully created and stored!")

# To load the index again in the future:
# loaded_index = faiss.read_index("./faiss_index.index")
