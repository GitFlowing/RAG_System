from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

# Load the same embedding model used during indexing
start_time = time.time()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
end_time = time.time()
print(f'Load Sentence Transformer in {end_time - start_time} seconds')

# Load the existing ChromaDB
start_time = time.time()
vectorstore = Chroma(persist_directory="./langchain_db", embedding_function=embedding_model)
end_time = time.time()
print(f'Load the Chroma DB in {end_time - start_time} seconds')

# Define your query
query_text = "Who is Albert Einstein?"

# Perform a similarity search (return top 2 results)
start_time = time.time()
results = vectorstore.similarity_search(query_text, k=2)
end_time = time.time()
print(f'Load Query Search in {end_time - start_time} seconds')

# Print the best matches
print("🔍 Best matches:")
for idx, doc in enumerate(results):
    print(f"{idx+1}. {doc.page_content}\n")
