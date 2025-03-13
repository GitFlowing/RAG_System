import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1️⃣ Load the FAISS index
index = faiss.read_index("./faiss_index.index")

# 2️⃣ Load the HuggingFace model (the same one used for embedding)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3️⃣ Define the query text
query_text = "What is the theory of relativity?"

# 4️⃣ Convert the query to an embedding
query_embedding = embedding_model.embed_documents([query_text])
query_embedding = np.array(query_embedding).astype("float32")

# 5️⃣ Perform the similarity search in FAISS
k = 2  # Number of closest documents to return
distances, indices = index.search(query_embedding, k)

# 6️⃣ Display the results
print("🔍 Search Results:")
for i, idx in enumerate(indices[0]):
    print(f"Document {i+1}: Index {idx}, Distance {distances[0][i]}")
