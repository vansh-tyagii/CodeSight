# backend/database.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Initialize Qdrant locally (No Docker needed)
# It will create a folder called 'qdrant_local_db' in your project
client = QdrantClient(path="./qdrant_local_db")

COLLECTION_NAME = "codesight_hybrid_db"

def setup_database():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "syntax_vector": VectorParams(size=768, distance=Distance.COSINE),
                "intent_vector": VectorParams(size=768, distance=Distance.COSINE),
            }
        )
        print("Database Collection Created with Named Vectors.")
    else:
        print("Database already exists. Ready to connect.")
    client.close()

if __name__ == "__main__":
    setup_database()