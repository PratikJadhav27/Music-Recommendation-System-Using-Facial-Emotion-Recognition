"""
Optional Google Sheets backend for user feedback (persists on Streamlit Cloud).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

HEADERS = ["timestamp", "emotion", "confidence_scores", "song_name", "song_url", "rating"]

_sheet_client = None
_worksheet = None


def _load_secrets_dict() -> Optional[dict[str, Any]]:
    """Read service account + sheet id from Streamlit secrets or env."""
    sheet_id = os.environ.get("FEEDBACK_SHEET_ID", "").strip()
    sa_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "").strip()

    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            if "feedback" in st.secrets and "sheet_id" in st.secrets["feedback"]:
                sheet_id = sheet_id or str(st.secrets["feedback"]["sheet_id"]).strip()
            if "gcp_service_account" in st.secrets:
                return {
                    "sheet_id": sheet_id,
                    "service_account": dict(st.secrets["gcp_service_account"]),
                }
    except Exception:
        pass

    if sa_json and sheet_id:
        try:
            return {"sheet_id": sheet_id, "service_account": json.loads(sa_json)}
        except json.JSONDecodeError:
            logger.warning("GCP_SERVICE_ACCOUNT_JSON is not valid JSON")
    return None


def is_sheets_configured() -> bool:
    cfg = _load_secrets_dict()
    return bool(cfg and cfg.get("sheet_id") and cfg.get("service_account"))


def _get_worksheet():
    global _sheet_client, _worksheet
    if _worksheet is not None:
        return _worksheet

    cfg = _load_secrets_dict()
    if not cfg:
        return None

    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_info(cfg["service_account"], scopes=SCOPES)
    _sheet_client = gspread.authorize(creds)
    spreadsheet = _sheet_client.open_by_key(cfg["sheet_id"])
    _worksheet = spreadsheet.sheet1
    return _worksheet


def append_feedback_row(row: List[Any]) -> bool:
    """Append one feedback row to Google Sheets. Returns False if not configured or on error."""
    try:
        ws = _get_worksheet()
        if ws is None:
            return False

        if ws.row_count == 0 or (ws.cell(1, 1).value or "").strip().lower() != "timestamp":
            ws.insert_row(HEADERS, index=1)

        ws.append_row([str(v) for v in row], value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        logger.warning("Google Sheets feedback append failed: %s", e)
        return False
