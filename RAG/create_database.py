import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
            text = text.strip('\n')
            text = text.strip()
            # Split in paragraphs
            paragraphs = text.split("\n\n")

            # Create String list
            string_list = [f"{filename[:-4]}_{paragraphs[num][:8]}" for num in range(len(paragraphs))]
            ids.extend(string_list)

            # Cut Time from paragraphs
            corrected_paragraphs = [paragraph[8:].strip() for paragraph in paragraphs]
            documents.extend(corrected_paragraphs)


    print(f'read file {idx}')

# Generate embeddings
embeddings = embedding_model.encode(documents).tolist()

# Store in ChromaDB
collection.add(documents=documents, ids=ids, embeddings=embeddings)

print("✅ Vector database created and saved locally!")
