import json
from pathlib import Path

import faiss
import numpy as np


INDEX_FOLDER = Path("faiss_index")
EMBEDDINGS_PATH = INDEX_FOLDER / "embeddings.npy"
ROWS_PATH = INDEX_FOLDER / "embedding_rows.json"
INDEX_PATH = INDEX_FOLDER / "index.faiss"


def main():
    if not EMBEDDINGS_PATH.exists():
        print(f"Embeddings file not found: {EMBEDDINGS_PATH}")
        return

    if not ROWS_PATH.exists():
        print(f"Embedding row mapping not found: {ROWS_PATH}")
        return

    embeddings = np.load(EMBEDDINGS_PATH)

    with ROWS_PATH.open("r", encoding="utf-8") as input_file:
        row_data = json.load(input_file)

    chunk_ids = row_data.get("chunk_ids", [])

    if embeddings.ndim != 2:
        print("The embeddings file must contain a 2D array.")
        return

    if len(embeddings) != len(chunk_ids):
        print("Vector count does not match the chunk ID count.")
        return

    if len(embeddings) == 0:
        print("No embeddings found.")
        return

    embeddings = np.asarray(embeddings, dtype=np.float32)
    vector_dimension = embeddings.shape[1]

    # Embeddings were normalized, so inner-product search gives cosine similarity.
    index = faiss.IndexFlatIP(vector_dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    print(f"Vectors added to FAISS: {index.ntotal}")
    print(f"Vector dimensions: {vector_dimension}")
    print(f"FAISS index saved: {INDEX_PATH}")


if __name__ == "__main__":
    main()