# Invoice Extraction API

`POST /extract` — accepts `{"invoice_text": "..."}` and returns:

```json
{
  "invoice_no": "...",
  "date": "YYYY-MM-DD",
  "vendor": "...",
  "amount": 0.0,
  "tax": 0.0,
  "currency": "..."
}
```

Uses Google Gemini (`gemini-2.5-flash`) with a strict JSON response schema to
extract the 6 fields, then normalizes dates to ISO format and numbers to
plain floats as a safety net.

Set the `GEMINI_API_KEY` environment variable before running.
