import json
import re
from pathlib import Path

EXTRACTED_FOLDER = Path("extracted_text")
OUTPUT_FOLDER = Path("chunked_data")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SENTENCE_LOOKBACK = 250


def split_into_chunks(text):
    """Split one page into overlapping chunks without cutting words."""
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        target_end = min(start + CHUNK_SIZE, len(text))
        end = target_end

        if target_end < len(text):
            # Prefer ending after a sentence in the last part of the chunk.
            sentence_start = max(start + CHUNK_SIZE // 2, target_end - SENTENCE_LOOKBACK)
            sentence_matches = list(
                re.finditer(r"[.!?][\"'”’)]?(?=\s)", text[sentence_start:target_end])
            )
            if sentence_matches:
                end = sentence_start + sentence_matches[-1].end()
            else:
                # Fall back to a whitespace boundary, so no word is cut.
                end = target_end
                while end > start and not text[end - 1].isspace():
                    end -= 1

                if end <= start:
                    end = target_end
                    while end < len(text) and not text[end].isspace():
                        end += 1

                if end >= len(text):
                    end = len(text)

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(text):
            break

        next_start = max(end - CHUNK_OVERLAP, start + 1)

        # Move to the first character after a whitespace boundary.
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1

        while next_start < end and text[next_start].isspace():
            next_start += 1

        start = next_start

    return chunks


def chunk_report(json_path):
    """Chunk one extracted report and save its metadata."""
    with json_path.open("r", encoding="utf-8") as input_file:
        report = json.load(input_file)

    company = report["company"]
    financial_year = report["financial_year"]
    source_pdf = report["source_pdf"]
    source_name = Path(source_pdf).stem

    report_chunks = []

    for page in report["pages"]:
        page_number = page["page_number"]
        text = page.get("text", "")

        page_chunks = split_into_chunks(text)

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            report_chunks.append(
                {
                    "chunk_id": (
                        f"{source_name}_p{page_number}_c{chunk_number}"
                    ),
                    "company": company,
                    "financial_year": financial_year,
                    "page_number": page_number,
                    "source_pdf": source_pdf,
                    "text": chunk_text,
                }
            )

    output_folder = OUTPUT_FOLDER / json_path.parent.name
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{json_path.stem}_chunks.json"

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            report_chunks,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path, len(report_chunks)


def main():
    json_files = sorted(EXTRACTED_FOLDER.rglob("*.json"))

    if not json_files:
        print(f"No extracted JSON files found under {EXTRACTED_FOLDER}.")
        return

    total_chunks = 0
    failures = []

    for file_number, json_path in enumerate(json_files, start=1):
        print(f"\n[{file_number}/{len(json_files)}] {json_path}")

        try:
            output_path, chunk_count = chunk_report(json_path)
            total_chunks += chunk_count
            print(f"Chunks created: {chunk_count}")
            print(f"Saved: {output_path}")
        except Exception as error:
            failures.append(f"{json_path}: {error}")
            print(f"Could not process this report: {error}")

    print(f"\nReports processed: {len(json_files) - len(failures)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Reports with errors: {len(failures)}")

    for failure in failures:
        print(f"- {failure}")


if __name__ == "__main__":
    main()
