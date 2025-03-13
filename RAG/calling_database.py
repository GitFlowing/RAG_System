import chromadb
from sentence_transformers import SentenceTransformer
import time

# Load Chroma-Datenbank
start_time = time.time()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="text_collection")
end_time = time.time()

print(f'Load the Chroma DB in {end_time - start_time} seconds')

# Load the same Embedding modell like during the creation of the Vector Database
start_time = time.time()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
end_time = time.time()
print(f'Load Sentence Transformer in {end_time - start_time} seconds')

# Define your query
start_time = time.time()
query_text = "What is the definition of AI?"
query_embedding = embedding_model.encode([query_text]).tolist()
end_time = time.time()
print(f'Load Embedding of query {end_time - start_time} seconds')

# Execute query in database (search for the best 2 results)
start_time = time.time()
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)
end_time = time.time()
print(f'Load Query Search in {end_time - start_time} seconds')

# Print the best results
print("🔍 Beste Treffer:")
for doc_id, doc in zip(results["ids"][0], results["documents"][0]):
    print(f"ID: {doc_id} → Document: {doc}\n")
