from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


# --------------------------------------------------
# 1. Load embedding model
# --------------------------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Create ChromaDB client
# --------------------------------------------------

CHROMA_DIR = "data/chroma"

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)


# --------------------------------------------------
# 3. Create ChromaDB collection
# --------------------------------------------------

COLLECTION_NAME = "zepto_policies"

try:
    chroma_client.delete_collection(COLLECTION_NAME)

except Exception:
    pass


collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)


# --------------------------------------------------
# 4. Documents folder
# --------------------------------------------------

DOCS_DIR = Path("docs")


# --------------------------------------------------
# 5. Create empty lists
# --------------------------------------------------

documents = []
embeddings = []
ids = []
metadatas = []


# --------------------------------------------------
# 6. Read all documents
# --------------------------------------------------

for file_path in sorted(DOCS_DIR.glob("doc_*.txt")):

    text = file_path.read_text(
        encoding="utf-8"
    ).strip()
    
    documents.append(text)

    ids.append(file_path.stem)

    metadatas.append({
        "source": file_path.name,
        "title": file_path.stem
    })


# --------------------------------------------------
# 7. Function to create embeddings
# --------------------------------------------------

def embed_text(text: str):

    return model.encode(text).tolist()


# --------------------------------------------------
# 8. Create document embeddings
# --------------------------------------------------

for document in documents:

    embedding = embed_text(document)

    embeddings.append(embedding)


# --------------------------------------------------
# 9. Store everything in ChromaDB
# --------------------------------------------------

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)
