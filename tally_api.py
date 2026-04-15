from __future__ import annotations

from datetime import date, datetime
from typing import Any
import xml.etree.ElementTree as ET

import requests


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9000
DEFAULT_TIMEOUT = 30
SUPPORTED_VOUCHER_TYPES = {"Receipt", "Payment", "Contra"}


def build_tally_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    """Return the HTTP endpoint exposed by TallyPrime's XML server."""

    return f"http://{host}:{int(port)}"


def escape_xml(value: str) -> str:
    """Escape XML-sensitive characters in narration and ledger names."""

    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_tally_date(value: Any) -> str:
    """Convert supported Python date values into Tally's YYYYMMDD format."""

    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")

    if isinstance(value, date):
        return value.strftime("%Y%m%d")

    parsed_date = datetime.fromisoformat(str(value))
    return parsed_date.strftime("%Y%m%d")


def extract_line_errors(root: ET.Element) -> list[str]:
    """Collect any LINEERROR nodes returned by Tally in its XML response."""

    line_errors: list[str] = []

    for element in root.iter():
        if element.tag.upper().endswith("LINEERROR"):
            text = (element.text or "").strip()
            if text:
                line_errors.append(text)

    return line_errors


def parse_import_summary(response_xml: str) -> dict[str, Any]:
    """Parse Tally's import response into a structured summary dictionary."""

    try:
        root = ET.fromstring(response_xml)
    except ET.ParseError:
        return {
            "created": 0,
            "altered": 0,
            "errors": 1,
            "line_errors": ["Tally returned non-XML content."],
        }

    def read_first_numeric(tag_suffix: str) -> int:
        for element in root.iter():
            if element.tag.upper().endswith(tag_suffix):
                try:
                    return int((element.text or "0").strip())
                except ValueError:
                    return 0
        return 0

    return {
        "created": read_first_numeric("CREATED"),
        "altered": read_first_numeric("ALTERED"),
        "errors": read_first_numeric("ERRORS"),
        "line_errors": extract_line_errors(root),
    }


def fetch_tally_ledgers(
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[str]:
    """
    Fetch all active ledger names from the company currently open in Tally.

    Raises:
        RuntimeError: If Tally is unreachable or the XML response is invalid.
    """

    request_xml = """
    <ENVELOPE>
        <HEADER>
            <VERSION>1</VERSION>
            <TALLYREQUEST>Export</TALLYREQUEST>
            <TYPE>Collection</TYPE>
            <ID>LedgerCollection</ID>
        </HEADER>
        <BODY>
            <DESC>
                <STATICVARIABLES>
                    <SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>
                </STATICVARIABLES>
                <TDL>
                    <TDLMESSAGE>
                        <COLLECTION NAME="LedgerCollection" ISMODIFY="No">
                            <TYPE>Ledger</TYPE>
                            <NATIVEMETHOD>Name</NATIVEMETHOD>
                        </COLLECTION>
                    </TDLMESSAGE>
                </TDL>
            </DESC>
        </BODY>
    </ENVELOPE>
    """.strip()

    url = build_tally_url(host=host, port=port)

    try:
        response = requests.post(
            url,
            data=request_xml.encode("utf-8"),
            headers={"Content-Type": "application/xml"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Unable to connect to Tally at {url}. Ensure TallyPrime is running and its XML server is enabled. Details: {exc}"
        ) from exc

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        raise RuntimeError(f"Tally returned invalid XML while fetching ledgers: {exc}") from exc

    line_errors = extract_line_errors(root)
    if line_errors:
        raise RuntimeError("Tally returned an error while fetching ledgers: " + " | ".join(line_errors))

    ledgers: set[str] = set()

    for ledger_element in root.iter():
        if not ledger_element.tag.upper().endswith("LEDGER"):
            continue

        attribute_name = (ledger_element.attrib.get("NAME") or "").strip()
        if attribute_name:
            ledgers.add(attribute_name)

        for child in ledger_element:
            if not child.tag.upper().endswith("NAME"):
                continue

            child_name = (child.text or "").strip()
            if child_name:
                ledgers.add(child_name)

    cleaned_ledgers = sorted(
        ledger for ledger in ledgers if ledger and ledger.lower() not in {"name", "ledger"}
    )

    if not cleaned_ledgers:
        raise RuntimeError(
            "Tally responded successfully, but no active ledgers were returned from the currently open company."
        )

    return cleaned_ledgers


def build_voucher_xml(voucher_data: dict[str, Any]) -> str:
    """
    Construct standard Tally voucher import XML from a normalized payload.

    Required keys:
        voucher_type, date, amount, narration, debit_ledger, credit_ledger
    """

    required_keys = {
        "voucher_type",
        "date",
        "amount",
        "narration",
        "debit_ledger",
        "credit_ledger",
    }
    missing_keys = required_keys.difference(voucher_data)
    if missing_keys:
        raise ValueError(
            "Voucher data is missing required fields: " + ", ".join(sorted(missing_keys))
        )

    voucher_type = str(voucher_data["voucher_type"]).strip().title()
    if voucher_type not in SUPPORTED_VOUCHER_TYPES:
        raise ValueError(
            f"Unsupported voucher type '{voucher_type}'. Supported types: Receipt, Payment, Contra."
        )

    debit_ledger = str(voucher_data["debit_ledger"]).strip()
    credit_ledger = str(voucher_data["credit_ledger"]).strip()
    narration = str(voucher_data["narration"]).strip()
    amount = abs(float(voucher_data["amount"]))
    tally_date = format_tally_date(voucher_data["date"])

    if not debit_ledger or not credit_ledger:
        raise ValueError("Both debit_ledger and credit_ledger must be provided.")

    if amount <= 0:
        raise ValueError("Voucher amount must be greater than zero.")

    return f"""
    <ENVELOPE>
        <HEADER>
            <TALLYREQUEST>Import Data</TALLYREQUEST>
        </HEADER>
        <BODY>
            <IMPORTDATA>
                <REQUESTDESC>
                    <REPORTNAME>Vouchers</REPORTNAME>
                </REQUESTDESC>
                <REQUESTDATA>
                    <TALLYMESSAGE xmlns:UDF="TallyUDF">
                        <VOUCHER VCHTYPE="{escape_xml(voucher_type)}" ACTION="Create" OBJVIEW="Accounting Voucher View">
                            <DATE>{tally_date}</DATE>
                            <EFFECTIVEDATE>{tally_date}</EFFECTIVEDATE>
                            <VOUCHERTYPENAME>{escape_xml(voucher_type)}</VOUCHERTYPENAME>
                            <PERSISTEDVIEW>Accounting Voucher View</PERSISTEDVIEW>
                            <NARRATION>{escape_xml(narration)}</NARRATION>
                            <ALLLEDGERENTRIES.LIST>
                                <LEDGERNAME>{escape_xml(debit_ledger)}</LEDGERNAME>
                                <ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>
                                <AMOUNT>{amount:.2f}</AMOUNT>
                            </ALLLEDGERENTRIES.LIST>
                            <ALLLEDGERENTRIES.LIST>
                                <LEDGERNAME>{escape_xml(credit_ledger)}</LEDGERNAME>
                                <ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE>
                                <AMOUNT>-{amount:.2f}</AMOUNT>
                            </ALLLEDGERENTRIES.LIST>
                        </VOUCHER>
                    </TALLYMESSAGE>
                </REQUESTDATA>
            </IMPORTDATA>
        </BODY>
    </ENVELOPE>
    """.strip()


def push_voucher_to_tally(
    voucher_data: dict[str, Any],
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Push one voucher to Tally and return a structured success/failure result.

    The returned dictionary always includes:
        success, message, request_xml, response_text, summary
    """

    try:
        request_xml = build_voucher_xml(voucher_data)
    except Exception as exc:
        return {
            "success": False,
            "message": f"Voucher payload is invalid: {exc}",
            "request_xml": "",
            "response_text": "",
            "summary": {},
        }

    url = build_tally_url(host=host, port=port)

    try:
        response = requests.post(
            url,
            data=request_xml.encode("utf-8"),
            headers={"Content-Type": "application/xml"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return {
            "success": False,
            "message": (
                f"Unable to connect to Tally at {url}. Ensure TallyPrime is running and its XML server is enabled. Details: {exc}"
            ),
            "request_xml": request_xml,
            "response_text": "",
            "summary": {},
        }

    response_text = response.text.strip()
    summary = parse_import_summary(response_text)

    if summary["errors"] > 0 or summary["line_errors"]:
        error_message = " | ".join(summary["line_errors"]) if summary["line_errors"] else "Tally reported an import error."
        return {
            "success": False,
            "message": error_message,
            "request_xml": request_xml,
            "response_text": response_text,
            "summary": summary,
        }

    if summary["created"] > 0 or summary["altered"] > 0:
        return {
            "success": True,
            "message": f"Voucher synced successfully. Created={summary['created']}, Altered={summary['altered']}.",
            "request_xml": request_xml,
            "response_text": response_text,
            "summary": summary,
        }

    return {
        "success": True,
        "message": "Tally accepted the request.",
        "request_xml": request_xml,
        "response_text": response_text,
        "summary": summary,
    }
