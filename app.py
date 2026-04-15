from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from pdf_parser import ParsedStatement, parse_statement
from tally_api import fetch_tally_ledgers, push_voucher_to_tally


APP_TITLE = "Bank Statement to Tally"
DEFAULT_BANK_KEY = "boi_od"
DEFAULT_TALLY_PORT = 9000
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "bank_memory.db"


def init_database(db_path: Path = DB_PATH) -> None:
    """Create the SQLite memory database used for historical ledger tagging."""

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL UNIQUE,
                ledger_name TEXT NOT NULL
            )
            """
        )
        connection.commit()


def load_saved_mappings(db_path: Path = DB_PATH) -> list[tuple[str, str]]:
    """
    Return historical keyword-to-ledger mappings.

    Longer keywords are loaded first so more specific matches win when multiple
    keywords appear in a bank description.
    """

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT keyword, ledger_name
            FROM ledger_mapping
            ORDER BY LENGTH(keyword) DESC, keyword ASC
            """
        ).fetchall()

    return [(str(keyword).strip().lower(), str(ledger_name).strip()) for keyword, ledger_name in rows]


def save_mapping_records(
    records: Iterable[tuple[str, str]], db_path: Path = DB_PATH
) -> int:
    """Persist approved description keywords and their selected ledger names."""

    record_list = list(records)
    if not record_list:
        return 0

    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO ledger_mapping (keyword, ledger_name)
            VALUES (?, ?)
            """,
            record_list,
        )
        connection.commit()

    return len(record_list)


def init_session_state() -> None:
    """Initialize predictable Streamlit state values used across reruns."""

    st.session_state.setdefault("tally_port", DEFAULT_TALLY_PORT)
    st.session_state.setdefault("tally_ledgers", [])
    st.session_state.setdefault("tally_error", "")
    st.session_state.setdefault("primary_bank_ledger", "")
    st.session_state.setdefault("uploaded_file_hash", "")
    st.session_state.setdefault("parsed_statement", None)
    st.session_state.setdefault("review_df", None)
    st.session_state.setdefault("sync_results", [])


def compute_file_hash(file_bytes: bytes) -> str:
    """Generate a stable signature so files are only reparsed when they change."""

    return hashlib.md5(file_bytes).hexdigest()


def extract_keyword(description: str, word_count: int = 3) -> str:
    """Use the first three description tokens as the reusable memory keyword."""

    tokens = re.findall(r"[A-Za-z0-9/-]+", str(description))
    return " ".join(tokens[:word_count]).lower().strip()


def default_voucher_type(transaction_type: str) -> str:
    """Map parsed debit/credit rows to the default voucher type."""

    return "Receipt" if str(transaction_type).strip().lower() == "credit" else "Payment"


def suggest_transaction(
    description: str,
    transaction_type: str,
    mappings: list[tuple[str, str]],
) -> tuple[str, str]:
    """
    Suggest the target ledger and voucher type for a bank row.

    Priority:
    1. Hardcoded CASH contra rules.
    2. Historical keyword matches from SQLite memory.
    3. Blank ledger with default voucher type.
    """

    normalized_description = str(description).strip()
    upper_description = normalized_description.upper()
    lower_description = normalized_description.lower()

    if "CASH" in upper_description and transaction_type in {"Credit", "Debit"}:
        return "Cash", "Contra"

    for keyword, ledger_name in mappings:
        if keyword and keyword in lower_description:
            return ledger_name, default_voucher_type(transaction_type)

    return "", default_voucher_type(transaction_type)


def apply_auto_tagging(
    dataframe: pd.DataFrame, mappings: list[tuple[str, str]]
) -> pd.DataFrame:
    """Add review and approval columns on top of the normalized parser output."""

    review_df = dataframe.copy()
    review_df["Bank Description"] = review_df["Description"]

    suggestions = review_df.apply(
        lambda row: suggest_transaction(
            description=row["Bank Description"],
            transaction_type=row["Type"],
            mappings=mappings,
        ),
        axis=1,
    )

    review_df["Suggested Ledger"] = [ledger for ledger, _ in suggestions]
    review_df["Voucher Type"] = [voucher_type for _, voucher_type in suggestions]
    review_df["Custom Narration"] = review_df["Bank Description"]
    review_df["Approve"] = False

    review_df = review_df[
        [
            "Date",
            "Bank Description",
            "Type",
            "Amount",
            "Suggested Ledger",
            "Voucher Type",
            "Custom Narration",
            "Approve",
        ]
    ].reset_index(drop=True)

    return review_df


def parse_uploaded_file(file_bytes: bytes) -> ParsedStatement:
    """Parse the uploaded statement using the configured bank parser."""

    return parse_statement(pdf_bytes=file_bytes, bank_key=DEFAULT_BANK_KEY)


def refresh_ledgers_from_tally(port: int) -> None:
    """Fetch ledger names from Tally and store the result in session state."""

    try:
        ledgers = fetch_tally_ledgers(port=port)
    except Exception as exc:
        st.session_state["tally_ledgers"] = []
        st.session_state["tally_error"] = str(exc)
        if st.session_state.get("primary_bank_ledger"):
            st.session_state["primary_bank_ledger"] = ""
        return

    st.session_state["tally_ledgers"] = ledgers
    st.session_state["tally_error"] = ""

    current_bank_ledger = st.session_state.get("primary_bank_ledger", "")
    if current_bank_ledger not in ledgers:
        st.session_state["primary_bank_ledger"] = ""


def build_voucher_payload(row: pd.Series, primary_bank_ledger: str) -> dict[str, str | float]:
    """
    Convert an approved review row into the payload required by the Tally API.

    Debit/Credit rules:
    - Bank withdrawal => Bank ledger credited, selected ledger debited.
    - Bank deposit => Bank ledger debited, selected ledger credited.
    """

    row_type = str(row["Type"]).strip()
    suggested_ledger = str(row["Suggested Ledger"]).strip()
    voucher_type = str(row["Voucher Type"]).strip() or default_voucher_type(row_type)

    if row_type == "Debit":
        debit_ledger = suggested_ledger
        credit_ledger = primary_bank_ledger
    else:
        debit_ledger = primary_bank_ledger
        credit_ledger = suggested_ledger

    return {
        "voucher_type": voucher_type,
        "date": row["Date"],
        "amount": float(row["Amount"]),
        "narration": str(row["Custom Narration"]).strip(),
        "debit_ledger": debit_ledger,
        "credit_ledger": credit_ledger,
    }


def persist_approved_mappings(review_df: pd.DataFrame) -> int:
    """Save approved row mappings to SQLite using the first three description words."""

    mapping_records: list[tuple[str, str]] = []

    approved_rows = review_df[review_df["Approve"] == True]
    for _, row in approved_rows.iterrows():
        ledger_name = str(row["Suggested Ledger"]).strip()
        if not ledger_name:
            continue

        keyword = extract_keyword(str(row["Bank Description"]))
        if not keyword:
            continue

        mapping_records.append((keyword, ledger_name))

    return save_mapping_records(mapping_records)


def render_sidebar() -> str:
    """Render Tally settings and return the selected primary bank ledger."""

    st.sidebar.header("Tally Settings")
    tally_port = st.sidebar.number_input(
        "Tally Port",
        min_value=1,
        max_value=65535,
        value=int(st.session_state["tally_port"]),
        step=1,
        help="Default TallyPrime XML HTTP port is 9000.",
    )
    st.session_state["tally_port"] = int(tally_port)

    if st.sidebar.button("Refresh Ledgers from Tally", use_container_width=True):
        refresh_ledgers_from_tally(port=int(tally_port))

    if st.session_state["tally_error"]:
        st.sidebar.error(st.session_state["tally_error"])
    elif st.session_state["tally_ledgers"]:
        st.sidebar.success(
            f"Loaded {len(st.session_state['tally_ledgers'])} ledger(s) from Tally."
        )
    else:
        st.sidebar.info("Refresh ledgers to load active Tally ledgers.")

    ledgers = st.session_state["tally_ledgers"]
    current_value = st.session_state.get("primary_bank_ledger", "")

    if ledgers:
        default_index = ledgers.index(current_value) if current_value in ledgers else None
        selected_ledger = st.sidebar.selectbox(
            "Primary Bank Ledger",
            options=ledgers,
            index=default_index,
            placeholder="Select the bank ledger for this statement",
            help="This is the Tally ledger representing the uploaded bank account.",
        )
    else:
        st.sidebar.selectbox(
            "Primary Bank Ledger",
            options=["Refresh ledgers from Tally first"],
            index=0,
            disabled=True,
        )
        selected_ledger = ""

    st.session_state["primary_bank_ledger"] = selected_ledger
    return selected_ledger


def render_data_editor(review_df: pd.DataFrame, ledgers: list[str]) -> pd.DataFrame:
    """Display the editable transaction approval grid."""

    column_config = {
        "Date": st.column_config.DateColumn("Date", disabled=True),
        "Bank Description": st.column_config.TextColumn(
            "Bank Description",
            disabled=True,
            width="large",
        ),
        "Type": st.column_config.TextColumn("Type", disabled=True),
        "Amount": st.column_config.NumberColumn(
            "Amount",
            disabled=True,
            format="%.2f",
        ),
        "Suggested Ledger": st.column_config.SelectboxColumn(
            "Suggested Ledger",
            options=ledgers,
            required=False,
            width="medium",
        ),
        "Voucher Type": st.column_config.TextColumn("Voucher Type", disabled=True),
        "Custom Narration": st.column_config.TextColumn(
            "Custom Narration",
            width="large",
        ),
        "Approve": st.column_config.CheckboxColumn("Approve"),
    }

    return st.data_editor(
        review_df,
        key="transaction_editor",
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config=column_config,
    )


def render_sync_results(sync_results: list[dict[str, str | bool]]) -> None:
    """Render the success/failure log for voucher sync attempts."""

    if not sync_results:
        return

    st.subheader("Tally Sync Log")
    for result in sync_results:
        message = str(result["message"])
        if result["success"]:
            st.success(message)
        else:
            st.error(message)


def main() -> None:
    """Streamlit application entrypoint."""

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Upload a bank statement PDF, review the extracted rows, and push approved vouchers directly into TallyPrime."
    )

    try:
        init_database()
    except Exception as exc:
        st.error(f"Unable to initialize the local SQLite database: {exc}")
        return

    init_session_state()
    primary_bank_ledger = render_sidebar()

    uploaded_file = st.file_uploader(
        "Upload Bank Statement PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Currently configured for Bank of India overdraft statements.",
    )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            file_hash = compute_file_hash(file_bytes)
        except Exception as exc:
            st.error(f"Unable to read the uploaded file: {exc}")
            return

        if file_hash != st.session_state["uploaded_file_hash"]:
            try:
                parsed_statement = parse_uploaded_file(file_bytes)
                mappings = load_saved_mappings()
                review_df = apply_auto_tagging(parsed_statement.dataframe, mappings)
            except Exception as exc:
                st.session_state["uploaded_file_hash"] = ""
                st.session_state["parsed_statement"] = None
                st.session_state["review_df"] = None
                st.error(f"Unable to parse the uploaded PDF: {exc}")
                return

            st.session_state["uploaded_file_hash"] = file_hash
            st.session_state["parsed_statement"] = parsed_statement
            st.session_state["review_df"] = review_df
            st.session_state["sync_results"] = []

    review_df = st.session_state.get("review_df")
    parsed_statement = st.session_state.get("parsed_statement")

    if review_df is None or parsed_statement is None:
        st.info("Upload a BOI overdraft statement PDF to begin.")
        return

    st.success(
        f"Parsed {len(parsed_statement.dataframe)} transaction row(s) from {parsed_statement.bank_name}."
    )

    approved_count = int(review_df["Approve"].sum())
    auto_tagged_count = int(review_df["Suggested Ledger"].astype(str).str.strip().ne("").sum())

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Transactions", len(review_df))
    metric_col_2.metric("Auto-Tagged", auto_tagged_count)
    metric_col_3.metric("Approved", approved_count)

    edited_df = render_data_editor(review_df, st.session_state["tally_ledgers"])
    st.session_state["review_df"] = edited_df

    action_col_1, action_col_2 = st.columns(2)

    with action_col_1:
        if st.button("Save Mappings", use_container_width=True):
            try:
                saved_count = persist_approved_mappings(edited_df)
            except Exception as exc:
                st.error(f"Unable to save mappings to SQLite: {exc}")
            else:
                if saved_count == 0:
                    st.info("No approved rows with a selected ledger were available to save.")
                else:
                    st.toast(f"Saved {saved_count} mapping(s) to local memory.")
                    st.success(f"Saved {saved_count} mapping(s) to bank_memory.db.")

    with action_col_2:
        if st.button("Sync Approved to Tally", use_container_width=True):
            approved_rows = edited_df[edited_df["Approve"] == True]

            if approved_rows.empty:
                st.session_state["sync_results"] = []
                st.info("No approved rows were selected for sync.")
            elif not primary_bank_ledger:
                st.session_state["sync_results"] = []
                st.error("Select the Primary Bank Ledger before syncing.")
            else:
                sync_results: list[dict[str, str | bool]] = []
                port = int(st.session_state["tally_port"])

                for _, row in approved_rows.iterrows():
                    description = str(row["Bank Description"]).strip()
                    suggested_ledger = str(row["Suggested Ledger"]).strip()

                    if not suggested_ledger:
                        sync_results.append(
                            {
                                "success": False,
                                "message": (
                                    f"{row['Date']} | {description} | Skipped because Suggested Ledger is blank."
                                ),
                            }
                        )
                        continue

                    try:
                        voucher_payload = build_voucher_payload(
                            row=row,
                            primary_bank_ledger=primary_bank_ledger,
                        )
                        result = push_voucher_to_tally(
                            voucher_data=voucher_payload,
                            port=port,
                        )
                    except Exception as exc:
                        sync_results.append(
                            {
                                "success": False,
                                "message": f"{row['Date']} | {description} | {exc}",
                            }
                        )
                        continue

                    sync_results.append(
                        {
                            "success": bool(result["success"]),
                            "message": f"{row['Date']} | {description} | {result['message']}",
                        }
                    )

                st.session_state["sync_results"] = sync_results

    render_sync_results(st.session_state["sync_results"])


if __name__ == "__main__":
    main()
