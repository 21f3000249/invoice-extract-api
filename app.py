import os
import re
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dateutil import parser as dateparser
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Add it as an environment variable in your deployment platform."
    )

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"  # free-tier eligible, strong at structured extraction

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_no": {"type": "string", "nullable": True},
        "date": {"type": "string", "nullable": True},
        "vendor": {"type": "string", "nullable": True},
        "amount": {"type": "number", "nullable": True},
        "tax": {"type": "number", "nullable": True},
        "currency": {"type": "string", "nullable": True},
    },
    "required": ["invoice_no", "date", "vendor", "amount", "tax", "currency"],
}

model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": RESPONSE_SCHEMA,
    },
)

REQUIRED_KEYS = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]

PROMPT_TEMPLATE = """You are an invoice data extraction system. Read the raw invoice
text below and extract exactly these 6 fields:

- invoice_no: the invoice/reference number (string), or null if not present
- date: the invoice/issue date, converted to ISO format YYYY-MM-DD, or null if not present
- vendor: the name of the vendor/seller issuing the invoice (not the client/bill-to), or null
- amount: the SUBTOTAL before tax, as a plain number (no currency symbols, no commas), or null
- tax: the tax amount only (e.g. GST/IGST/VAT amount), as a plain number, or null
- currency: the currency code, e.g. INR, USD, EUR, or null if not determinable

Rules:
- amount must be the pre-tax subtotal, NOT the grand total.
- tax must be just the tax line amount, not included in amount.
- If the invoice text uses "Rs." or "₹" treat currency as INR unless stated otherwise.
- Dates may appear in many formats (e.g. "15 March 2026", "2026-01-22", "22/01/2026") -
  always normalize to YYYY-MM-DD.
- Return null for any field you cannot confidently determine. Never guess.

Invoice text:
---
{invoice_text}
---

Return ONLY a JSON object with keys: invoice_no, date, vendor, amount, tax, currency.
"""

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Invoice Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    invoice_text: str


class ExtractResponse(BaseModel):
    invoice_no: str | None = None
    date: str | None = None
    vendor: str | None = None
    amount: float | None = None
    tax: float | None = None
    currency: str | None = None


def _normalize_date(value):
    """Best-effort conversion of a date string to YYYY-MM-DD. Returns None on failure."""
    if not value:
        return None
    try:
        dt = dateparser.parse(str(value), dayfirst=False, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError, TypeError):
        return None


def _to_number(value):
    """Coerce a value to a float, stripping common currency symbols/codes
    and thousands separators before parsing."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    # Strip currency words/symbols first so a trailing "Rs." period doesn't
    # get mistaken for a decimal point.
    text = re.sub(r"(rs\.?|inr|usd|eur|gbp|₹|\$|€|£)", "", text, flags=re.I)
    match = re.search(r"-?\d[\d,]*\.?\d*", text)
    if not match:
        return None
    num_str = match.group(0).replace(",", "")
    try:
        return float(num_str)
    except ValueError:
        return None


def _extract_json_object(raw_text: str) -> dict:
    """Parse the model's raw text as JSON, falling back to extracting the
    first {...} block if the model wrapped it in extra text/markdown."""
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise HTTPException(status_code=502, detail="Model did not return valid JSON")


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    prompt = PROMPT_TEMPLATE.format(invoice_text=req.invoice_text)

    try:
        response = model.generate_content(prompt)
        raw_answer = response.text or "{}"
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}")

    parsed = _extract_json_object(raw_answer)

    result = {key: parsed.get(key) for key in REQUIRED_KEYS}
    result["date"] = _normalize_date(result.get("date"))
    result["amount"] = _to_number(result.get("amount"))
    result["tax"] = _to_number(result.get("tax"))

    if result.get("currency"):
        result["currency"] = str(result["currency"]).strip().upper()

    return ExtractResponse(**result)
