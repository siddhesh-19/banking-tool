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
DATE_PATTERN = re.compile(r"\d{2}[/-]\d{2}[/-]\d{2,4}")


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
        return bool(DATE_PATTERN.fullmatch(txn_date))

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


class PNBStatementParser(BaseStatementParser):
    """
    Parser for Punjab National Bank statements.

    PNB PDF extraction is less stable than BOI. Depending on the statement
    layout, pdfplumber may either preserve separate debit/credit columns or
    collapse them into a single merged amount column. This parser handles both
    layouts and normalizes every transaction to the same schema used by the app.
    """

    bank_key = "pnb"
    bank_name = "Punjab National Bank (PNB)"

    TABLE_SETTINGS = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 10,
        "snap_tolerance": 3,
        "join_tolerance": 3,
    }

    def parse(self, pdf_bytes: bytes) -> ParsedStatement:
        extracted_entries: list[dict[str, str]] = []
        header_map: dict[str, int] | None = None

        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables(table_settings=self.TABLE_SETTINGS) or []
                for table in page_tables:
                    for raw_row in table:
                        row = self._normalize_dynamic_row(raw_row)
                        if not any(row):
                            continue

                        detected_header = self._detect_header_map(row)
                        if detected_header is not None:
                            header_map = detected_header
                            continue

                        if header_map is None:
                            continue

                        if self._is_transaction_row(row, header_map):
                            extracted_entries.append(self._row_to_entry(row, header_map))
                            continue

                        if self._is_continuation_row(row, header_map) and extracted_entries:
                            continuation = self._get_cell(row, header_map.get("description_index"))
                            if continuation:
                                extracted_entries[-1]["description_raw"] = (
                                    f"{extracted_entries[-1]['description_raw']} {continuation}".strip()
                                )

        if not extracted_entries:
            raise ValueError(
                "No PNB transaction table could be extracted from the PDF. Verify that the statement format matches the expected Punjab National Bank layout."
            )

        normalized_df = self._normalize_entries(extracted_entries)
        if normalized_df.empty:
            raise ValueError(
                "The PNB PDF was read, but no valid transaction rows remained after cleaning."
            )

        return ParsedStatement(
            bank_key=self.bank_key,
            bank_name=self.bank_name,
            dataframe=normalized_df,
        )

    def _normalize_entries(self, entries: list[dict[str, str]]) -> pd.DataFrame:
        """Convert extracted PNB rows into the shared statement schema."""

        frame = pd.DataFrame(entries)
        frame["Date"] = pd.to_datetime(
            frame["date_raw"],
            dayfirst=True,
            errors="coerce",
        ).dt.date
        frame["Description"] = frame["description_raw"].map(self._clean_cell)
        frame["Debit Amount"] = frame["debit_raw"].map(self._parse_amount)
        frame["Credit Amount"] = frame["credit_raw"].map(self._parse_amount)
        frame["Merged Amount"] = frame["merged_amount_raw"].map(self._parse_amount)
        frame["Balance Value"] = frame["balance_raw"].map(self._parse_signed_balance)

        frame["Type"] = None
        frame["Amount"] = None

        for row_index in frame.index:
            txn_type, amount = self._resolve_type_and_amount(frame, row_index)
            frame.at[row_index, "Type"] = txn_type
            frame.at[row_index, "Amount"] = amount

        frame["Description"] = frame["Description"].str.replace(r"\s+", " ", regex=True).str.strip()
        frame = frame.dropna(subset=["Date", "Amount"])
        frame = frame[frame["Description"].astype(bool)]
        frame = frame.loc[:, STANDARD_COLUMNS].reset_index(drop=True)

        return frame

    def _detect_header_map(self, row: list[str]) -> dict[str, int] | None:
        """Detect a PNB header row and return column positions for the table."""

        normalized_cells = [self._normalize_header_cell(cell) for cell in row]
        joined_row = " | ".join(normalized_cells)

        if "txn date" not in joined_row or "description" not in joined_row:
            return None

        header_map: dict[str, int] = {}

        for index, cell in enumerate(normalized_cells):
            if "txn no" in cell:
                header_map["txn_no_index"] = index
            elif "txn date" in cell:
                header_map["date_index"] = index
            elif "description" in cell:
                header_map["description_index"] = index
            elif "branch" in cell:
                header_map["branch_index"] = index
            elif "cheque" in cell:
                header_map["cheque_index"] = index
            elif "remarks" in cell:
                header_map["remarks_index"] = index
            elif "balance" in cell:
                header_map["balance_index"] = index
            elif "dr amount" in cell and "cr amount" in cell:
                header_map["merged_amount_index"] = index
            elif "dr amount" in cell or cell == "dr":
                header_map["debit_index"] = index
            elif "cr amount" in cell or cell == "cr":
                header_map["credit_index"] = index
            elif cell == "amount" and "merged_amount_index" not in header_map:
                header_map["merged_amount_index"] = index

        required_keys = {"date_index", "description_index"}
        if not required_keys.issubset(header_map):
            return None

        return header_map

    def _is_transaction_row(self, row: list[str], header_map: dict[str, int]) -> bool:
        """Identify transaction rows using the detected PNB date column."""

        txn_date = self._get_cell(row, header_map.get("date_index"))
        return bool(DATE_PATTERN.fullmatch(txn_date))

    def _is_continuation_row(self, row: list[str], header_map: dict[str, int]) -> bool:
        """Identify wrapped description rows that belong to the previous PNB transaction."""

        if any(DATE_PATTERN.fullmatch(cell) for cell in row if cell):
            return False

        description = self._get_cell(row, header_map.get("description_index"))
        if not description:
            return False

        amount_candidates = [
            self._get_cell(row, header_map.get("debit_index")),
            self._get_cell(row, header_map.get("credit_index")),
            self._get_cell(row, header_map.get("merged_amount_index")),
            self._get_cell(row, header_map.get("balance_index")),
        ]

        return not any(value for value in amount_candidates if value)

    def _row_to_entry(self, row: list[str], header_map: dict[str, int]) -> dict[str, str]:
        """Project a raw PNB table row into the fields used for normalization."""

        return {
            "date_raw": self._get_cell(row, header_map.get("date_index")),
            "description_raw": self._get_cell(row, header_map.get("description_index")),
            "debit_raw": self._get_cell(row, header_map.get("debit_index")),
            "credit_raw": self._get_cell(row, header_map.get("credit_index")),
            "merged_amount_raw": self._get_cell(row, header_map.get("merged_amount_index")),
            "balance_raw": self._get_cell(row, header_map.get("balance_index")),
            "remarks_raw": self._get_cell(row, header_map.get("remarks_index")),
        }

    def _resolve_type_and_amount(
        self, frame: pd.DataFrame, row_index: int
    ) -> tuple[str | None, float | None]:
        """Resolve transaction type and amount from the raw PNB fields."""

        row = frame.loc[row_index]
        debit_amount = row["Debit Amount"]
        credit_amount = row["Credit Amount"]
        merged_amount = row["Merged Amount"]

        if self._has_amount(debit_amount) and not self._has_amount(credit_amount):
            return "Debit", float(debit_amount)

        if self._has_amount(credit_amount) and not self._has_amount(debit_amount):
            return "Credit", float(credit_amount)

        if self._has_amount(debit_amount) and self._has_amount(credit_amount):
            if float(debit_amount) >= float(credit_amount):
                return "Debit", float(debit_amount)
            return "Credit", float(credit_amount)

        if self._has_amount(merged_amount):
            inferred_type = self._infer_type_from_balance(frame, row_index, float(merged_amount))
            if inferred_type is None:
                inferred_type = self._infer_type_from_text(
                    description=str(row["Description"]),
                    remarks=str(row["remarks_raw"]),
                    amount_cell=str(row["merged_amount_raw"]),
                )

            if inferred_type is not None:
                return inferred_type, float(merged_amount)

            return None, None

        return None, None

    def _infer_type_from_balance(
        self, frame: pd.DataFrame, row_index: int, amount: float
    ) -> str | None:
        """
        Infer debit/credit direction from adjacent running balances.

        This handles layouts where pdfplumber collapses PNB debit and credit
        columns into a single amount field.
        """

        current_balance = frame.at[row_index, "Balance Value"]
        if pd.isna(current_balance):
            return None

        inferred_types: set[str] = set()
        for neighbor_index in (row_index - 1, row_index + 1):
            if neighbor_index not in frame.index:
                continue

            neighbor_balance = frame.at[neighbor_index, "Balance Value"]
            if pd.isna(neighbor_balance):
                continue

            delta = float(current_balance) - float(neighbor_balance)
            if self._amounts_close(delta, amount):
                inferred_types.add("Credit")
            elif self._amounts_close(delta, -amount):
                inferred_types.add("Debit")

        if len(inferred_types) == 1:
            return inferred_types.pop()

        return None

    def _infer_type_from_text(
        self, description: str, remarks: str, amount_cell: str
    ) -> str | None:
        """Fallback heuristic for merged PNB amount columns when balance inference is unavailable."""

        combined_text = " ".join([description, remarks, amount_cell]).lower()

        if re.search(r"\bcr\b|\bcredit\b", combined_text):
            return "Credit"
        if re.search(r"\bdr\b|\bdebit\b", combined_text):
            return "Debit"

        credit_keywords = [
            "nrtgs",
            "interest",
            "refund",
            "reversal",
            "salary",
            "cash deposit",
            "cash dep",
            "by clg",
        ]
        debit_keywords = [
            "withdrawal",
            "atm",
            "charges",
            "charge",
            "emi",
            "ecs",
            "ach",
            "debit card",
            "purchase",
        ]

        if any(keyword in combined_text for keyword in credit_keywords):
            return "Credit"
        if any(keyword in combined_text for keyword in debit_keywords):
            return "Debit"

        return None

    def _normalize_dynamic_row(self, row: list[str | None] | None) -> list[str]:
        """Clean a variable-width pdfplumber row without forcing a fixed column count."""

        return [self._clean_cell(cell) for cell in (row or [])]

    def _normalize_header_cell(self, value: str | None) -> str:
        """Normalize header cell text so header detection is resilient to whitespace noise."""

        cell = self._clean_cell(value).lower()
        cell = cell.replace(".", "")
        cell = re.sub(r"\s+", " ", cell)
        return cell.strip()

    def _get_cell(self, row: list[str], index: int | None) -> str:
        """Safely read a value from a variable-width row."""

        if index is None or index >= len(row):
            return ""
        return self._clean_cell(row[index])

    def _clean_cell(self, value: str | None) -> str:
        """Collapse whitespace and embedded newlines into a single clean string."""

        if value is None:
            return ""

        cleaned_value = str(value).replace("\n", " ").replace("\r", " ")
        cleaned_value = re.sub(r"\s+", " ", cleaned_value)
        return cleaned_value.strip()

    def _parse_amount(self, value: str | None) -> float | None:
        """Parse a numeric amount from a PNB amount field."""

        cleaned_value = self._clean_cell(value)
        if not cleaned_value:
            return None

        numeric_match = re.search(r"-?\d[\d,]*\.\d{2}", cleaned_value)
        if numeric_match is None:
            return None

        numeric_text = numeric_match.group(0).replace(",", "")
        try:
            return abs(float(numeric_text))
        except ValueError:
            return None

    def _parse_signed_balance(self, value: str | None) -> float | None:
        """Parse balance values such as '14,738.01 Cr.' or '2,500.00 Dr.' into signed floats."""

        cleaned_value = self._clean_cell(value)
        if not cleaned_value:
            return None

        amount = self._parse_amount(cleaned_value)
        if amount is None:
            return None

        if re.search(r"\bdr\b", cleaned_value.lower()):
            return -amount

        return amount

    def _has_amount(self, value: float | None) -> bool:
        """Return True when the parsed value is a positive numeric amount."""

        return pd.notna(value) and float(value) > 0

    def _amounts_close(self, left: float, right: float, tolerance: float = 0.05) -> bool:
        """Compare two monetary values using a small tolerance for PDF parsing noise."""

        return abs(float(left) - float(right)) <= tolerance


PARSERS: dict[str, BaseStatementParser] = {
    BOIOverdraftParser.bank_key: BOIOverdraftParser(),
    PNBStatementParser.bank_key: PNBStatementParser(),
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
