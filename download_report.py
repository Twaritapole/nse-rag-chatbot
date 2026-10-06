from io import BytesIO
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import quote, urljoin
import zipfile

import requests


NSE_HOME_URL = "https://www.nseindia.com/"
NSE_API_URL = "https://www.nseindia.com/api/annual-reports"
NSE_ARCHIVE_URL = "https://nsearchives.nseindia.com/annual_reports/"
NSE_REPORTS_PAGE = (
    "https://www.nseindia.com/companies-listing/"
    "corporate-filings-annual-reports"
)

COMPANIES = {
    "TCS": "TCS",
    "Infosys": "INFY",
    "Reliance_Industries": "RELIANCE",
    "HDFC_Bank": "HDFCBANK",
    "ICICI_Bank": "ICICIBANK",
}

FIRST_FINANCIAL_YEAR = 2015
LAST_FINANCIAL_YEAR = 2025
DATA_FOLDER = Path("data")
RECHECK_EXISTING_REPORTS = True


def find_report_rows(value):
    """Find annual report records in NSE's JSON response."""
    rows = []

    if isinstance(value, dict):
        if value.get("fileName") or value.get("file_name"):
            rows.append(value)

        for child in value.values():
            rows.extend(find_report_rows(child))

    elif isinstance(value, list):
        for child in value:
            rows.extend(find_report_rows(child))

    return rows


def extract_year(value):
    """Get a four-digit year from a value such as '2025'."""
    match = re.search(r"20\d{2}", str(value))
    return int(match.group()) if match else None


def make_report_url(file_name):
    """Convert the filename or path from NSE into a full URL."""
    file_name = str(file_name).strip()

    if file_name.startswith(("http://", "https://")):
        return file_name

    if "nsearchives.nseindia.com/" in file_name:
        return "https://" + file_name.lstrip("/")

    if file_name.startswith("/"):
        return urljoin("https://nsearchives.nseindia.com", file_name)

    if "/" in file_name:
        return urljoin(
            "https://nsearchives.nseindia.com/",
            quote(file_name, safe="/:%?=&"),
        )

    return urljoin(
        NSE_ARCHIVE_URL,
        quote(file_name, safe="/:%?=&"),
    )


def safe_archive_member_path(member_name):
    """Return a safe relative path for a ZIP member, or None if unsafe."""
    normalized_name = member_name.replace("\\", "/")
    member_path = PurePosixPath(normalized_name)

    if (
        member_path.is_absolute()
        or ".." in member_path.parts
        or (member_path.parts and ":" in member_path.parts[0])
    ):
        return None

    return Path(*member_path.parts)


def save_pdf_response(response, destination, company_folder, symbol, financial_year):
    """Save a PDF response or every file found inside a ZIP response."""
    response.raise_for_status()
    downloaded_bytes = response.content

    if downloaded_bytes.startswith(b"%PDF-"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(downloaded_bytes)
        return [destination], 0

    elif zipfile.is_zipfile(BytesIO(downloaded_bytes)):
        saved_pdfs = []
        saved_other_files = 0
        pdf_members = []
        other_members = []

        with zipfile.ZipFile(BytesIO(downloaded_bytes)) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue

                member_path = safe_archive_member_path(item.filename)
                if member_path is None:
                    raise ValueError(
                        f"Unsafe path in ZIP; not extracted: {item.filename}"
                    )

                with archive.open(item) as member_file:
                    member_bytes = member_file.read()

                if member_bytes.startswith(b"%PDF-"):
                    pdf_members.append((member_path, member_bytes))
                else:
                    other_members.append((member_path, member_bytes))

        output_folder = DATA_FOLDER / company_folder
        output_folder.mkdir(parents=True, exist_ok=True)

        for index, (member_path, pdf_bytes) in enumerate(pdf_members, start=1):
            if index == 1:
                pdf_destination = destination
            else:
                safe_stem = re.sub(
                    r"[^A-Za-z0-9_-]+", "_", member_path.stem
                ).strip("_") or "report_part"
                pdf_destination = output_folder / (
                    f"{symbol}_{financial_year}_part{index:02d}_"
                    f"{safe_stem}.pdf"
                )

            pdf_destination.write_bytes(pdf_bytes)
            saved_pdfs.append(pdf_destination)

        extras_folder = (
            output_folder / "archive_files" / f"{symbol}_{financial_year}"
        )
        for member_path, member_bytes in other_members:
            extra_destination = extras_folder / member_path
            extra_destination.parent.mkdir(parents=True, exist_ok=True)
            extra_destination.write_bytes(member_bytes)
            saved_other_files += 1

        if not saved_pdfs:
            raise ValueError(
                "ZIP contained no PDF files; "
                f"saved {saved_other_files} other archive file(s)"
            )

        return saved_pdfs, saved_other_files

    else:
        content_type = response.headers.get("Content-Type", "unknown")
        raise ValueError(
            f"Response is not a PDF or ZIP. "
            f"HTTP {response.status_code}; "
            f"Content-Type: {content_type}; "
            f"URL: {response.url}; "
            f"first bytes: {downloaded_bytes[:12]!r}"
        )

    raise ValueError("The response did not contain a valid PDF or ZIP file.")


def get_report_rows(session, symbol):
    response = session.get(
        NSE_API_URL,
        params={"index": "equities", "symbol": symbol},
        timeout=45,
    )
    response.raise_for_status()
    return find_report_rows(response.json())


def main():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": NSE_REPORTS_PAGE,
        }
    )

    # Start a session with NSE before requesting its report listing.
    try:
        home_response = session.get(NSE_HOME_URL, timeout=45)
        home_response.raise_for_status()
    except Exception as error:
        print(f"Could not connect to NSE: {error}")
        return

    missing_reports = []
    failed_downloads = []
    downloaded_count = 0
    archived_other_file_count = 0

    for company_folder, symbol in COMPANIES.items():
        print(f"\nChecking {company_folder} ({symbol})...")

        try:
            report_rows = get_report_rows(session, symbol)
        except Exception as error:
            print(f"Could not retrieve NSE's listing: {error}")
            continue

        for start_year in range(
            FIRST_FINANCIAL_YEAR,
            LAST_FINANCIAL_YEAR + 1,
        ):
            end_year = start_year + 1
            financial_year = f"FY{start_year}-{str(end_year)[-2:]}"

            matching_rows = []
            for row in report_rows:
                row_start = extract_year(
                    row.get("fromYr") or row.get("fromYear")
                )
                row_end = extract_year(
                    row.get("toYr") or row.get("toYear")
                )

                if row_start == start_year and row_end == end_year:
                    matching_rows.append(row)

            if not matching_rows:
                missing_reports.append(
                    f"{company_folder} {financial_year}"
                )
                print(f"Not listed by NSE: {financial_year}")
                continue

            # If NSE lists duplicate filings for a year, use its first entry.
            row = matching_rows[0]
            file_name = row.get("fileName") or row.get("file_name")

            if not file_name:
                failed_downloads.append(
                    f"{company_folder} {financial_year}: no file link"
                )
                print(f"No file link for {financial_year}")
                continue

            output_folder = DATA_FOLDER / company_folder
            output_file = (
                output_folder / f"{symbol}_{financial_year}.pdf"
            )

            if output_file.exists() and not RECHECK_EXISTING_REPORTS:
                with output_file.open("rb") as saved_file:
                    is_valid_pdf = saved_file.read(5) == b"%PDF-"

                if is_valid_pdf:
                    print(f"Already downloaded: {output_file}")
                    continue

                failed_downloads.append(
                    f"{company_folder} {financial_year}: "
                    f"existing file is not a valid PDF: {output_file}"
                )
                print(f"Existing file is not a valid PDF: {output_file}")
                continue

            url = make_report_url(file_name)

            try:
                response = session.get(url, timeout=90)
                saved_pdfs, saved_other_files = save_pdf_response(
                    response,
                    output_file,
                    company_folder,
                    symbol,
                    financial_year,
                )

                downloaded_count += len(saved_pdfs)
                archived_other_file_count += saved_other_files
                for saved_pdf in saved_pdfs:
                    print(f"Downloaded PDF: {saved_pdf}")
                if saved_other_files:
                    print(
                        "Saved non-PDF ZIP files under: "
                        f"{DATA_FOLDER / company_folder / 'archive_files'} "
                        f"({saved_other_files} file(s))"
                    )
                time.sleep(1)

            except Exception as error:
                failed_downloads.append(
                    f"{company_folder} {financial_year}: {error}"
                )
                print(f"Download failed for {financial_year}: {error}")

    print(f"\nPDF files saved or updated: {downloaded_count}")
    print(
        "Non-PDF files saved from ZIP archives: "
        f"{archived_other_file_count}"
    )

    print("\nReports not listed by NSE:")
    if missing_reports:
        for report in missing_reports:
            print(f"- {report}")
    else:
        print("- None")

    print("\nDownload failures:")
    if failed_downloads:
        for failure in failed_downloads:
            print(f"- {failure}")
    else:
        print("- None")


if __name__ == "__main__":
    main()