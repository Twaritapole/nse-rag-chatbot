import json
import re
import unicodedata
from pathlib import Path

import pymupdf


DATA_FOLDER = Path("data")
OUTPUT_FOLDER = Path("extracted_text")

COMPANY_NAMES = {
    "TCS": "TCS",
    "Infosys": "Infosys",
    "Reliance_Industries": "Reliance Industries",
    "HDFC_Bank": "HDFC Bank",
    "ICICI_Bank": "ICICI Bank",
}

FINANCIAL_YEAR_PATTERN = re.compile(r"FY(\d{4})-(\d{2})", re.IGNORECASE)


def clean_page_text(text):
    """Normalize common PDF text artifacts while keeping line breaks."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")

    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_report(pdf_path):
    """Extract one PDF part while retaining its source and page numbers."""
    relative_pdf_path = pdf_path.relative_to(DATA_FOLDER)
    if len(relative_pdf_path.parts) < 2:
        raise ValueError("PDF must be inside a company folder under data")

    company_folder = relative_pdf_path.parts[0]
    company = COMPANY_NAMES.get(company_folder, company_folder)

    year_match = FINANCIAL_YEAR_PATTERN.search(relative_pdf_path.as_posix())
    if not year_match:
        raise ValueError("Filename does not contain a year like FY2025-26")

    financial_year = (
        f"FY{year_match.group(1)}-{year_match.group(2)}"
    )
    source_pdf = relative_pdf_path.as_posix()
    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = clean_page_text(page.get_text("text"))

            pages.append(
                {
                    "company": company,
                    "financial_year": financial_year,
                    "page_number": page_number,
                    "source_pdf": source_pdf,
                    "text": text,
                }
            )

    return company, financial_year, source_pdf, pages


def main():
    pdf_files = sorted(
        path
        for path in DATA_FOLDER.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print(f"No PDF files found under {DATA_FOLDER}.")
        return

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    failures = []

    for file_number, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n[{file_number}/{len(pdf_files)}] {pdf_path}")

        try:
            company, financial_year, source_pdf, pages = extract_report(
                pdf_path
            )

            company_folder = pdf_path.relative_to(DATA_FOLDER).parts[0]
            output_folder = OUTPUT_FOLDER / company_folder
            output_folder.mkdir(parents=True, exist_ok=True)
            output_path = output_folder / f"{pdf_path.stem}.json"

            report_data = {
                "company": company,
                "financial_year": financial_year,
                "source_pdf": source_pdf,
                "page_count": len(pages),
                "pages": pages,
            }

            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(
                    report_data,
                    output_file,
                    ensure_ascii=False,
                    indent=2,
                )

            total_characters = sum(len(page["text"]) for page in pages)
            empty_pages = sum(not page["text"] for page in pages)

            print(f"Pages extracted: {len(pages)}")
            print(f"Characters extracted: {total_characters}")
            print(f"Pages with no text: {empty_pages}")
            print(f"Saved: {output_path}")

        except Exception as error:
            failures.append(f"{pdf_path}: {error}")
            print(f"Could not process this report: {error}")

    print(f"\nReports found: {len(pdf_files)}")
    print(f"Reports with errors: {len(failures)}")

    for failure in failures:
        print(f"- {failure}")


if __name__ == "__main__":
    main()
