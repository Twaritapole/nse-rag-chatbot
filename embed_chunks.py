import json
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_FOLDER = Path("chunked_data")
OUTPUT_FOLDER = Path("faiss_index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32

REQUIRED_FIELDS = {
    "chunk_id",
    "company",
    "financial_year",
    "page_number",
    "source_pdf",
    "text",
}


def load_chunks():
    chunks = []
    seen_chunk_ids = set()
    seen_texts = set()
    skipped_empty = 0
    skipped_duplicates = 0

    json_files = sorted(CHUNKS_FOLDER.rglob("*.json"))

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as input_file:
            report_chunks = json.load(input_file)

        if not isinstance(report_chunks, list):
            raise ValueError(f"Expected a list of chunks in {json_path}")

        for chunk_number, chunk in enumerate(report_chunks, start=1):
            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Invalid chunk {chunk_number} in {json_path}"
                )

            missing_fields = REQUIRED_FIELDS - chunk.keys()
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(
                    f"Chunk {chunk_number} in {json_path} "
                    f"is missing required fields: {missing}"
                )

            text = chunk["text"]
            if not isinstance(text, str):
                raise ValueError(
                    f"Chunk {chunk_number} in {json_path} has non-text content"
                )

            text = text.strip()
            if not text:
                skipped_empty += 1
                continue

            chunk_id = str(chunk["chunk_id"])
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk ID found: {chunk_id}")
            seen_chunk_ids.add(chunk_id)

            normalized_text = " ".join(text.split()).casefold()

            # Keep matching text from different PDFs or pages as separate sources.
            duplicate_key = (
                chunk["company"],
                chunk["financial_year"],
                chunk["source_pdf"],
                chunk["page_number"],
                normalized_text,
            )

            if duplicate_key in seen_texts:
                skipped_duplicates += 1
                continue

            seen_texts.add(duplicate_key)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )

    return chunks, skipped_empty, skipped_duplicates


def main():
    if not CHUNKS_FOLDER.exists():
        print(f"Chunk folder not found: {CHUNKS_FOLDER}")
        return

    try:
        chunks, skipped_empty, skipped_duplicates = load_chunks()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Could not load chunks: {error}")
        return

    print(f"Empty chunks skipped: {skipped_empty}")
    print(f"Duplicate chunks skipped on the same page: {skipped_duplicates}")

    if not chunks:
        print(f"No usable chunks found under {CHUNKS_FOLDER}.")
        return

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    batch_count = math.ceil(len(texts) / BATCH_SIZE)

    print(f"Chunks to embed: {len(texts)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Approximate batches: {batch_count}")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or len(embeddings) != len(chunk_ids):
        print("Embedding count does not match the chunk count. Files not saved.")
        return

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    embeddings_path = OUTPUT_FOLDER / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    row_map_path = OUTPUT_FOLDER / "embedding_rows.json"
    with row_map_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "vector_count": len(chunk_ids),
                "chunk_ids": chunk_ids,
            },
            output_file,
            indent=2,
        )

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved vectors: {embeddings_path}")
    print(f"Saved row mapping: {row_map_path}")


if __name__ == "__main__":
    main()
    