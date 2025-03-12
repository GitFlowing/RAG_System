import chromadb

# Use Chroma DB
chroma_client = chromadb.Client()

# Create Collection
collection = chroma_client.create_collection(name="my_texts")

collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

# Create Query
results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)
