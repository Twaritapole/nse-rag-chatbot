import json
from pathlib import Path

import faiss


CHUNKS_FOLDER = Path("chunked_data")
INDEX_FOLDER = Path("faiss_index")

ROWS_PATH = INDEX_FOLDER / "embedding_rows.json"
FAISS_INDEX_PATH = INDEX_FOLDER / "index.faiss"
OUTPUT_PATH = INDEX_FOLDER / "chunk_metadata.json"

REQUIRED_FIELDS = {
    "chunk_id",
    "company",
    "financial_year",
    "page_number",
    "source_pdf",
    "text",
}


def main():
    for required_path in (ROWS_PATH, FAISS_INDEX_PATH):
        if not required_path.exists():
            print(f"Required file not found: {required_path}")
            return

    try:
        with ROWS_PATH.open("r", encoding="utf-8") as input_file:
            row_data = json.load(input_file)

        chunk_ids = row_data.get("chunk_ids")
        if not isinstance(chunk_ids, list):
            print("The row mapping does not contain a chunk_ids list.")
            return

        if len(chunk_ids) != len(set(chunk_ids)):
            print("The row mapping contains duplicate chunk IDs.")
            return

        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        if faiss_index.ntotal != len(chunk_ids):
            print(
                "FAISS vector count does not match the row mapping: "
                f"{faiss_index.ntotal} vectors and {len(chunk_ids)} IDs."
            )
            return

        chunks_by_id = {}

        for json_path in sorted(CHUNKS_FOLDER.rglob("*.json")):
            with json_path.open("r", encoding="utf-8") as input_file:
                chunks = json.load(input_file)

            if not isinstance(chunks, list):
                raise ValueError(f"Expected a list of chunks in {json_path}")

            for chunk in chunks:
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    raise ValueError(f"Chunk without an ID in {json_path}")

                if chunk_id in chunks_by_id:
                    raise ValueError(f"Duplicate chunk ID found: {chunk_id}")

                chunks_by_id[chunk_id] = chunk

        metadata_rows = []
        missing_ids = []

        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)

            if chunk is None:
                missing_ids.append(chunk_id)
                continue

            missing_fields = REQUIRED_FIELDS - chunk.keys()
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(
                    f"Chunk {chunk_id} is missing fields: {missing}"
                )

            metadata_rows.append(
                {field: chunk[field] for field in REQUIRED_FIELDS}
            )

        if missing_ids:
            preview = ", ".join(missing_ids[:10])
            print(
                f"Could not find {len(missing_ids)} chunk IDs. "
                f"First missing IDs: {preview}"
            )
            return

        metadata_store = {
            "vector_count": len(metadata_rows),
            "rows": metadata_rows,
        }

        with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
            json.dump(
                metadata_store,
                output_file,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        print(f"Metadata rows saved: {len(metadata_rows)}")
        print(f"Metadata store: {OUTPUT_PATH}")

    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Could not build the metadata store: {error}")


if __name__ == "__main__":
    main()