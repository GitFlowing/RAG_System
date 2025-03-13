import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# TODO use timestamp Data from mark
# NOW with data with
# Directory containing text files and vector database
DATA_FOLDER = "./data"
CHROMA_PATH = "./chroma_db"


# Initialize ChromaDB (local persistent storage)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="text_collection")

# Load the embedding model (free & local)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & efficient model

# Read all text files and store contents
documents = []
ids = []
for idx, filename in enumerate(os.listdir(DATA_FOLDER)):
    if filename.endswith(".txt"):  # Process only .txt files
        file_path = os.path.join(DATA_FOLDER, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            # Split in paragraphs
            paragraphs = text.split("\n\n")
            documents.extend(paragraphs)

            # Create String list
            string_list = [f"{filename[:-4]}_{num}" for num in range(len(paragraphs))]
            ids.extend(string_list)
    print(f'read file {idx}')

# Generate embeddings
embeddings = embedding_model.encode(documents).tolist()

# Store in ChromaDB
collection.add(documents=documents, ids=ids, embeddings=embeddings)

print("✅ Vector database created and saved locally!")



## ====== Data Base with chunking ==========================
# import os
# import chromadb
# import nltk
# from nltk.tokenize import sent_tokenize
# from sentence_transformers import SentenceTransformer

# # Download NLTK tokenizer data
# nltk.download("punkt")

# # Initialize ChromaDB (local persistent storage)
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(name="text_collection")

# # Load the embedding model (free & local)
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Directory containing text files
# DATA_FOLDER = "./data"

# # Function to split text into overlapping chunks
# def chunk_text(text, chunk_size=300, overlap=50):
#     sentences = sent_tokenize(text)  # Split text into sentences
#     chunks = []
#     current_chunk = []

#     for sentence in sentences:
#         current_chunk.append(sentence)
#         if sum(len(s) for s in current_chunk) > chunk_size:
#             chunks.append(" ".join(current_chunk))  # Add chunk to list
#             current_chunk = current_chunk[-overlap:]  # Keep overlap
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))  # Add remaining text

#     return chunks

# # Read all text files, chunk them, and store in ChromaDB
# ids = []
# documents = []
# for idx, filename in enumerate(os.listdir(DATA_FOLDER)):
#     if filename.endswith(".txt"):
#         file_path = os.path.join(DATA_FOLDER, filename)
#         with open(file_path, "r", encoding="utf-8") as file:
#             text = file.read()
#             text_chunks = chunk_text(text)  # Apply text chunking

#             # Store chunks
#             for chunk_idx, chunk in enumerate(text_chunks):
#                 documents.append(chunk)
#                 ids.append(f"{filename}_chunk_{chunk_idx}")

# # Generate embeddings for all chunks
# embeddings = embedding_model.encode(documents).tolist()

# # Store in ChromaDB
# collection.add(documents=documents, ids=ids, embeddings=embeddings)

# print(f"✅ Stored {len(documents)} text chunks in the vector database!")
