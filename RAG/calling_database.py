import chromadb
from sentence_transformers import SentenceTransformer

# Load Chroma-Datenbank
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="text_collection")

# Load the same Embedding modell like during the creation of the Vector Database
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define your query
query_text = "Wer ist Albert Einstein?"
query_embedding = embedding_model.encode([query_text]).tolist()

# Execute query in database (search for the best 2 results)
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

# Print the best results
print("🔍 Beste Treffer:")
for idx, doc in enumerate(results["documents"][0]):
    print(f"{idx+1}. {doc}\n")
