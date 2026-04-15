from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import pandas as pd
import pdfplumber


BOI_RAW_COLUMNS = [
    "Sl No",
    "Txn Date",
    "Description",
    "Cheque No",
    "Withdrawal (in Rs.)",
    "Deposits (in Rs.)",
    "Balance (in Rs.)",
]

STANDARD_COLUMNS = ["Date", "Description", "Type", "Amount"]


@dataclass(slots=True)
class ParsedStatement:
    """Structured parser result returned to the Streamlit app."""

    bank_key: str
    bank_name: str
    dataframe: pd.DataFrame


class BaseStatementParser(ABC):
    """Base contract used to register and execute bank-specific PDF parsers."""

    bank_key: str
    bank_name: str

    @abstractmethod
    def parse(self, pdf_bytes: bytes) -> ParsedStatement:
        """Read raw PDF bytes and return a normalized statement DataFrame."""


class BOIOverdraftParser(BaseStatementParser):
    """
    Parser for Bank of India overdraft account statements.

    The statement layout is table-based and usually repeats its column header on
    each page. The parser extracts all table fragments, removes non-transaction
    lines, merges wrapped description rows, and normalizes the result to the
    shared transaction schema.
    """

    bank_key = "boi_od"
    bank_name = "Bank of India (BOI) Overdraft"

    TABLE_SETTINGS = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 10,
        "snap_tolerance": 3,
        "join_tolerance": 3,
    }

    def parse(self, pdf_bytes: bytes) -> ParsedStatement:
        """Extract the BOI overdraft transaction table and normalize it."""

        extracted_rows: list[list[str]] = []

        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables(table_settings=self.TABLE_SETTINGS) or []
                for table in page_tables:
                    extracted_rows.extend(self._extract_table_rows(table))

        merged_rows = self._merge_continuation_rows(extracted_rows)
        if not merged_rows:
            raise ValueError(
                "No BOI transaction table could be extracted from the PDF. Verify that the statement format matches the BOI overdraft layout."
            )

        raw_df = pd.DataFrame(merged_rows, columns=BOI_RAW_COLUMNS)
        normalized_df = self._normalize_dataframe(raw_df)

        if normalized_df.empty:
            raise ValueError(
                "The PDF was read, but no valid transaction rows remained after cleaning."
            )

        return ParsedStatement(
            bank_key=self.bank_key,
            bank_name=self.bank_name,
            dataframe=normalized_df,
        )

    def _extract_table_rows(self, table: list[list[str | None]]) -> list[list[str]]:
        """Return only the meaningful transaction rows from one extracted table."""

        cleaned_rows: list[list[str]] = []
        header_seen = False

        for raw_row in table:
            normalized_row = self._normalize_row(raw_row)
            if not any(normalized_row):
                continue

            if self._is_header_row(normalized_row):
                header_seen = True
                continue

            if self._is_transaction_row(normalized_row) or self._is_continuation_row(normalized_row):
                cleaned_rows.append(normalized_row)
            elif header_seen:
                # Rows below a valid header that do not look like transaction
                # data are treated as page noise and skipped.
                continue

        return cleaned_rows

    def _merge_continuation_rows(self, rows: Iterable[list[str]]) -> list[list[str]]:
        """
        Merge wrapped description rows back into the previous transaction row.

        BOI PDFs occasionally split long descriptions across multiple lines. The
        continuation line typically has a blank date and blank amount columns,
        so it can be stitched onto the preceding transaction description.
        """

        merged_rows: list[list[str]] = []

        for row in rows:
            if self._is_continuation_row(row) and merged_rows:
                previous_row = merged_rows[-1]
                previous_row[2] = f"{previous_row[2]} {row[2]}".strip()
                if not previous_row[3] and row[3]:
                    previous_row[3] = row[3]
                continue

            merged_rows.append(list(row))

        return merged_rows

    def _normalize_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Convert BOI raw columns into the app's standard transaction schema."""

        frame = dataframe.copy()

        frame["Txn Date"] = frame["Txn Date"].map(self._clean_cell)
        frame["Description"] = frame["Description"].map(self._clean_cell)
        frame["Withdrawal (in Rs.)"] = frame["Withdrawal (in Rs.)"].map(self._parse_amount)
        frame["Deposits (in Rs.)"] = frame["Deposits (in Rs.)"].map(self._parse_amount)

        frame["Date"] = pd.to_datetime(
            frame["Txn Date"],
            dayfirst=True,
            errors="coerce",
        ).dt.date
        frame["Type"] = frame.apply(self._determine_type, axis=1)
        frame["Amount"] = frame.apply(self._determine_amount, axis=1)

        frame["Description"] = frame["Description"].str.replace(r"\s+", " ", regex=True).str.strip()
        frame = frame.dropna(subset=["Date", "Amount"])
        frame = frame[frame["Description"].astype(bool)]
        frame = frame.loc[:, STANDARD_COLUMNS].reset_index(drop=True)

        return frame

    def _normalize_row(self, row: list[str | None] | None) -> list[str]:
        """Pad or trim extracted table rows to the exact BOI column count."""

        raw_cells = row or []
        cleaned_cells = [self._clean_cell(cell) for cell in raw_cells]

        if len(cleaned_cells) < len(BOI_RAW_COLUMNS):
            cleaned_cells.extend([""] * (len(BOI_RAW_COLUMNS) - len(cleaned_cells)))

        return cleaned_cells[: len(BOI_RAW_COLUMNS)]

    def _is_header_row(self, row: list[str]) -> bool:
        """Match BOI statement header rows using the exact expected column names."""

        normalized_expected = [value.lower() for value in BOI_RAW_COLUMNS]
        normalized_row = [value.lower() for value in row]
        return normalized_row == normalized_expected

    def _is_transaction_row(self, row: list[str]) -> bool:
        """Identify standard transaction rows using the BOI date pattern."""

        txn_date = row[1]
        return bool(re.fullmatch(r"\d{2}[/-]\d{2}[/-]\d{2,4}", txn_date))

    def _is_continuation_row(self, row: list[str]) -> bool:
        """Identify wrapped description lines that belong to the prior transaction."""

        has_blank_date = not row[1]
        has_description = bool(row[2])
        has_no_amounts = not row[4] and not row[5]
        has_no_balance = not row[6]

        return has_blank_date and has_description and has_no_amounts and has_no_balance

    def _determine_type(self, row: pd.Series) -> str | None:
        """Resolve transaction type using withdrawal and deposit columns."""

        withdrawal = row["Withdrawal (in Rs.)"]
        deposit = row["Deposits (in Rs.)"]

        if pd.notna(withdrawal) and float(withdrawal) > 0:
            return "Debit"
        if pd.notna(deposit) and float(deposit) > 0:
            return "Credit"
        return None

    def _determine_amount(self, row: pd.Series) -> float | None:
        """Pick the correct amount column based on the resolved transaction type."""

        if row["Type"] == "Debit":
            return row["Withdrawal (in Rs.)"]
        if row["Type"] == "Credit":
            return row["Deposits (in Rs.)"]
        return None

    def _clean_cell(self, value: str | None) -> str:
        """Collapse PDF whitespace noise into a clean single-line string."""

        if value is None:
            return ""

        cleaned_value = str(value).replace("\n", " ").replace("\r", " ")
        cleaned_value = re.sub(r"\s+", " ", cleaned_value)
        return cleaned_value.strip()

    def _parse_amount(self, value: str | None) -> float | None:
        """Parse BOI amount text into a numeric float."""

        cleaned_value = self._clean_cell(value)
        if not cleaned_value:
            return None

        normalized = cleaned_value.replace(",", "")
        normalized = normalized.replace("CR", "").replace("DR", "").strip()
        normalized = normalized.replace("(", "-").replace(")", "")

        try:
            return float(normalized)
        except ValueError:
            return None


PARSERS: dict[str, BaseStatementParser] = {
    BOIOverdraftParser.bank_key: BOIOverdraftParser(),
}


def get_supported_parsers() -> dict[str, str]:
    """Return a map of available parser keys to their display names."""

    return {bank_key: parser.bank_name for bank_key, parser in PARSERS.items()}


def parse_statement(pdf_bytes: bytes, bank_key: str = "boi_od") -> ParsedStatement:
    """Dispatch the uploaded statement to the correct bank parser."""

    parser = PARSERS.get(bank_key)
    if parser is None:
        supported_keys = ", ".join(sorted(PARSERS))
        raise ValueError(
            f"Unsupported bank parser '{bank_key}'. Supported parser keys: {supported_keys}"
        )

    return parser.parse(pdf_bytes)
