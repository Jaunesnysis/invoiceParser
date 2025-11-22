import streamlit as st
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import zipfile
import tempfile
import os
import json
import pandas as pd
from openai import OpenAI
import re

# ================== OCR FUNCTION ==================
def extract_text(path: str) -> str:
    """
    Extract text from a PDF or image file using Tesseract OCR.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        pages = convert_from_path(path, dpi=300)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page)
        return text
    else:
        image = Image.open(path)
        return pytesseract.image_to_string(image)


# ================== LOCAL PARSING (NO AI) ==================
MONEY_REGEX = r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})"


def detect_currency(text: str) -> str:
    """
    Very simple currency detection from the whole invoice text.
    """
    if "$" in text:
        return "USD"
    if "€" in text:
        return "EUR"
    # Very rough guesses for some common ones:
    if " NOK" in text or "kr" in text:
        return "NOK"
    if " PLN" in text or "zł" in text.lower():
        return "PLN"
    if " GBP" in text or "£" in text:
        return "GBP"
    return ""  # unknown / not detected


def extract_invoice_meta_basic(ocr_text: str):
    """
    Very rough invoice metadata extractor (number, date, client).
    Works across many layouts, but not perfect.
    """
    invoice_number = ""
    invoice_date = ""
    client_name = ""

    # Invoice number (e.g. "Invoice #1058", "Faktura 7001884827")
    m = re.search(
        r"(Invoice|Faktura)\s*#?\s*[:\-]?\s*([A-Za-z0-9\-\/]+)",
        ocr_text,
        re.IGNORECASE,
    )
    if m:
        invoice_number = m.group(2).strip()

    # Try a few common date formats
    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",      # 2024-02-19
        r"\d{2}\.\d{2}\.\d{4}",    # 19.02.2024
        r"\d{1,2}/\d{1,2}/\d{2,4}" # 7/19/24 or 07/19/2024
    ]
    for p in date_patterns:
        d = re.search(p, ocr_text)
        if d:
            invoice_date = d.group(0)
            break

    # Very rough client name: after "Bill to" or "Recipient"
    c = re.search(r"(Bill to|Recipient|RECIPIENT)[:\s]+(.+)", ocr_text, re.IGNORECASE)
    if c:
        client_name = c.group(2).strip()

    return invoice_number, invoice_date, client_name


# ---------- SPECIALISED PARSERS FOR YOUR INVOICES ----------

def _parse_bestdrive_items(ocr_text: str,
                           invoice_number: str,
                           invoice_date: str,
                           client_name: str,
                           currency: str):
    """
    Bestdrive / Continental invoices.

    We parse single lines that already contain:
    NAME  QTY  EA  UNIT_PRICE  [DISCOUNT]  TOTAL

    Examples:
      255/50R19*Y BRAVURIS 5HM 107Y FR XL 2,00 EA 1 842,00 3 684,00
      MILJU@AVGIFT GRUPPE 1 2,00 EA 20,00 40,00
    """
    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
    items = []

    # Regex:
    #   name      -> everything up to quantity
    #   qty       -> 2,00
    #   unit      -> 1 842,00
    #   total(opt)-> 3 684,00
    line_re = re.compile(
        r"^(?P<name>.+?)\s+"
        r"(?P<qty>[\d,]+)\s+EA\s+"
        r"(?P<unit>[\d\s]+,[\d]+)"
        r"(?:\s+(?P<total>[\d\s]+,[\d]+))?$"
    )

    for line in lines:
        m = line_re.match(line)
        if not m:
            continue

        name = m.group("name")
        qty_str = m.group("qty")
        total_str = m.group("total") or m.group("unit")  # fallback if no total

        # Normalise numbers: remove spaces, change comma to dot
        quantity = qty_str.replace(" ", "").replace(".", "").replace(",", ".")
        product_price = total_str.replace(" ", "").replace(".", "", 0).replace(",", ".")

        items.append({
            "invoice_number": invoice_number,
            "invoice_date": invoice_date,
            "client_name": client_name,
            "product_name": name,
            "quantity": quantity,
            "product_price": product_price,
            "currency": currency or "NOK",
        })

    return items



def _parse_ndi_items(ocr_text: str,
                     invoice_number: str,
                     invoice_date: str,
                     client_name: str,
                     currency: str):
    """
    NDI Norge invoices.

    Example product line:
      3965185651541006 185/65R15 92T XL Nexen N'blue 4 stk 881,00 30,0 2.466,80

    Example fee line:
      Dekkavgift PV/VV 4 20,00 80,00
    """
    def normalize_money(s: str) -> str:
        # Remove thousand separators, convert comma to dot
        s = s.replace(" ", "")
        s = s.replace(".", "")
        s = s.replace(",", ".")
        return s

    lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
    items = []

    # Main tire line: product-id + desc + qty + 'stk' + unit + discount + total
    tire_re = re.compile(
        r"^(?P<product_id>\d{8,})\s+"
        r"(?P<desc>.+?)\s+"
        r"(?P<qty>\d+)\s+stk\s+"
        r"(?P<unit>[\d.,]+)\s+"
        r"(?P<discount>[\d.,]+)\s+"
        r"(?P<total>[\d.,]+)$",
        re.IGNORECASE,
    )

    # Environmental fee line: Dekkavgift PV/VV 4 20,00 80,00
    fee_re = re.compile(
        r"^(?P<name>Dekkavgift.*)\s+"
        r"(?P<qty>\d+)\s+"
        r"(?P<unit>[\d.,]+)\s+"
        r"(?P<total>[\d.,]+)$",
        re.IGNORECASE,
    )

    for line in lines:
        # 1) Tire line
        mt = tire_re.match(line)
        if mt:
            desc = mt.group("desc")
            qty_str = mt.group("qty")
            total_str = mt.group("total")

            quantity = qty_str  # keep as integer-like string
            product_price = normalize_money(total_str)

            items.append({
                "invoice_number": invoice_number,
                "invoice_date": invoice_date,
                "client_name": client_name,
                "product_name": desc,
                "quantity": quantity,
                "product_price": product_price,
                "currency": currency or "NOK",
            })
            continue

        # 2) Dekkavgift line
        mf = fee_re.match(line)
        if mf:
            name = mf.group("name")
            qty_str = mf.group("qty")
            total_str = mf.group("total")

            quantity = qty_str
            product_price = normalize_money(total_str)

            items.append({
                "invoice_number": invoice_number,
                "invoice_date": invoice_date,
                "client_name": client_name,
                "product_name": name,
                "quantity": quantity,
                "product_price": product_price,
                "currency": currency or "NOK",
            })
            continue

    return items



def extract_line_items_local(ocr_text: str):
    """
    LOCAL parser specialised for your Norwegian tire invoices.
    Returns:
      invoice_number, invoice_date, client_name, product_name,
      quantity, product_price, currency
    """
    invoice_number, invoice_date, client_name = extract_invoice_meta_basic(ocr_text)
    invoice_currency = detect_currency(ocr_text)

    upper = ocr_text.upper()

    # Special cases first
    if "NDI NORGE AS" in upper or "NDI NORGE A/S" in upper or "NDI NORGE" in upper:
        ndi_items = _parse_ndi_items(
            ocr_text, invoice_number, invoice_date, client_name, invoice_currency
        )
        if ndi_items:
            return ndi_items

    if "BESTDR" in upper or "BESTDRIVE" in upper or "BY CONTINENTAL" in upper:
        bd_items = _parse_bestdrive_items(
            ocr_text, invoice_number, invoice_date, client_name, invoice_currency
        )
        if bd_items:
            return bd_items

    # -------------- Fallback: old generic heuristic --------------
    items = []
    for raw_line in ocr_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower_line = line.lower()

        # Skip typical summary/totals/tax lines
        if any(keyword in lower_line for keyword in [
            "subtotal", "total", "tax", "vat", "mwst",
            "gross", "sales tax", "deposit", "powered by"
        ]):
            continue

        # Look for money patterns on the line
        money_matches = re.findall(MONEY_REGEX, line)
        if not money_matches:
            continue

        # Last amount on line = treat as total price for this row
        total_price = money_matches[-1]

        # Try to detect quantity as an integer token before the first money value
        tokens = line.split()
        quantity = ""
        if money_matches:
            first_money = money_matches[0]
            # Find index of the token that contains first_money
            first_money_idx = None
            for i, tok in enumerate(tokens):
                if first_money in tok:
                    first_money_idx = i
                    break

            if first_money_idx is not None and first_money_idx > 0:
                prev_tok = tokens[first_money_idx - 1]
                # Very simple: if previous token is a pure integer, treat as quantity
                if re.fullmatch(r"\d+", prev_tok):
                    quantity = prev_tok

        # Remove the money parts to get the text
        product_text = re.sub(MONEY_REGEX, "", line)
        product_text = re.sub(r"\s+", " ", product_text).strip(":- ")

        if not product_text:
            continue

        items.append({
            "invoice_number": invoice_number,
            "invoice_date": invoice_date,
            "client_name": client_name,
            "product_name": product_text,
            "quantity": quantity,
            "product_price": total_price.replace(",", "."),
            "currency": invoice_currency
        })

    return items


# ================== AI PARSING FUNCTION (CHATGPT) ==================
def extract_line_items_with_ai(ocr_text: str, client: OpenAI):
    """
    Send OCR text to ChatGPT and get line items as JSON.
    Each item should have:
      invoice_number, invoice_date, client_name,
      product_name, quantity, product_price, currency
    Returns: list[dict]
    """
    system_msg = (
        "You are an assistant that extracts structured invoice data from raw OCR text. "
        "You always return ONLY valid JSON, no explanation."
    )

    user_prompt = f"""
You will receive raw OCR text extracted from an invoice.

Your task is to identify all line items (products or services) from the invoice
and return them as a JSON array.

For each line item, extract:

- invoice_number
- invoice_date
- client_name
- product_name
- quantity
- product_price
- currency

Rules:
- If the invoice contains multiple products, return ONE JSON object per product.
- If quantity is not visible, use "1".
- If currency is visible (e.g. $, €, kr, NOK, PLN, GBP), map it to a 3-letter code
  when possible (USD, EUR, NOK, PLN, GBP). If unknown, use an empty string "".
- If product price includes VAT, just return that total into "product_price".
- If some invoice fields (like client_name or invoice_date) appear once,
  repeat them for every product row.
- If you cannot find a field, use an empty string "".
- Always return ONLY a JSON array, no extra text or comments.

Example of the ONLY valid output format:

[
  {{
    "invoice_number": "12345",
    "invoice_date": "2024-05-14",
    "client_name": "ABC Supplies",
    "product_name": "Brake Pads",
    "quantity": "4",
    "product_price": "89.90",
    "currency": "USD"
  }},
  {{
    "invoice_number": "12345",
    "invoice_date": "2024-05-14",
    "client_name": "ABC Supplies",
    "product_name": "Oil Filter",
    "quantity": "2",
    "product_price": "12.50",
    "currency": "USD"
  }}
]

Now extract the line items from this OCR text:

<<<OCR_TEXT_START>>>
{ocr_text}
<<<OCR_TEXT_END>>>
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content

    try:
        items = json.loads(content)
        if isinstance(items, dict):
            if "items" in items and isinstance(items["items"], list):
                return items["items"]
            return []
        if isinstance(items, list):
            return items
        return []
    except json.JSONDecodeError:
        return []


# ================== STREAMLIT APP ==================
st.set_page_config(page_title="Invoice OCR + Parser", layout="wide")
st.title("🧾 Invoice OCR + Line-Item Extractor")

st.write(
    "Upload a **single invoice file** (PDF or image) or a **ZIP** containing multiple invoices. "
    "The app will run OCR on each invoice, then extract product line items into a table."
)

uploaded_file = st.file_uploader(
    "Choose an invoice file or a ZIP with many invoices",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "webp", "zip"],
)

# Choose parsing method: Local (free) vs AI (ChatGPT)
parser_mode = st.radio(
    "Choose parsing method:",
    ["Local (free)", "AI (ChatGPT)"],
    index=0
)

# Only ask for API key if AI mode is selected
client = None
if parser_mode == "AI (ChatGPT)":
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    if not api_key:
        st.info("Please enter your OpenAI API key to use AI extraction, or switch to Local (free) mode.")
        st.stop()
    client = OpenAI(api_key=api_key)

if uploaded_file is not None:

    # ------------------------------------------------------------------
    # =========== CASE 1: ZIP WITH MANY FILES ===========
    # ------------------------------------------------------------------
    if uploaded_file.name.lower().endswith(".zip"):
        st.write("📦 **ZIP file detected — will process all supported files inside it.**")

        if st.button("🔍 Run OCR + Parse on all files in ZIP"):
            with st.spinner("Extracting ZIP and running OCR on each invoice..."):
                ocr_results = []  # list of (filename, text)

                # Save uploaded ZIP to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                    tmp_zip.write(uploaded_file.read())
                    zip_path = tmp_zip.name

                # Extract ZIP contents into a temporary directory
                extract_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Walk through extracted files
                for root, dirs, files in os.walk(extract_dir):
                    for filename in files:
                        file_path = os.path.join(root, filename)

                        # 1) Skip macOS junk files
                        if "__MACOSX" in file_path:
                            continue
                        if filename.startswith("._"):
                            continue

                        # 2) Only process supported file extensions
                        if not filename.lower().endswith(
                            (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
                        ):
                            continue

                        # 3) Run OCR safely
                        try:
                            text = extract_text(file_path)
                            ocr_results.append((filename, text))
                        except Exception as e:
                            ocr_results.append((filename, f"[ERROR processing file: {e}]"))

                st.success("✅ OCR completed for all supported files in the ZIP!")

                # Show raw OCR text for debugging
                st.subheader("📄 OCR Results (per file)")
                for filename, text in ocr_results:
                    st.markdown(f"**{filename}**")
                    st.text_area("Extracted text", text, height=200, key=filename)

                # Combine all texts into one big .txt for download
                combined_text = ""
                for filename, text in ocr_results:
                    combined_text += f"\n\n===== {filename} =====\n{text}"

                st.download_button(
                    label="💾 Download all OCR results as .txt",
                    data=combined_text,
                    file_name="all_invoices_text.txt",
                    mime="text/plain",
                )

            # -------- PARSING PART: build table with line items --------
            st.subheader("🤖 Parsed Line Items (All Invoices)")

            all_rows = []
            with st.spinner("Extracting line items with selected method..."):
                for filename, text in ocr_results:
                    if text.startswith("[ERROR"):
                        continue

                    if parser_mode == "AI (ChatGPT)":
                        items = extract_line_items_with_ai(text, client)
                    else:
                        items = extract_line_items_local(text)

                    for item in items:
                        row = {
                            "File Name": filename,
                            "Invoice Number": item.get("invoice_number", ""),
                            "Invoice Date": item.get("invoice_date", ""),
                            "Client Name": item.get("client_name", ""),
                            "Product Name": item.get("product_name", ""),
                            "Quantity": item.get("quantity", ""),
                            "Product Price": item.get("product_price", ""),
                            "Currency": item.get("currency", ""),
                        }
                        all_rows.append(row)

            if all_rows:
                df = pd.DataFrame(all_rows)
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download table as CSV",
                    data=csv_data,
                    file_name="invoices_line_items.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No line items were extracted. Check OCR quality or try another invoice format.")

    # ------------------------------------------------------------------
    # =========== CASE 2: SINGLE PDF/IMAGE ===========
    # ------------------------------------------------------------------
    else:
        st.write(f"**File name:** {uploaded_file.name}")

        # Optional: preview image
        if uploaded_file.type and uploaded_file.type.startswith("image/"):
            st.image(uploaded_file, caption="Uploaded image preview", use_column_width=True)

        if st.button("🔍 Run OCR + Parse on this file"):
            with st.spinner("Running OCR..."):

                # Save uploaded file to a temp file
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Run OCR
                try:
                    text = extract_text(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            st.success("✅ OCR completed!")

            st.subheader("📄 OCR Text")
            st.text_area("Extracted text", text, height=400)

            st.download_button(
                label="💾 Download OCR text as .txt",
                data=text,
                file_name=os.path.splitext(uploaded_file.name)[0] + "_text.txt",
                mime="text/plain",
            )

            # -------- PARSING PART: table for this one file --------
            st.subheader("🤖 Parsed Line Items (This Invoice)")

            if parser_mode == "AI (ChatGPT)":
                with st.spinner("Using AI to extract line items..."):
                    items = extract_line_items_with_ai(text, client)
            else:
                with st.spinner("Extracting line items locally (no AI)..."):
                    items = extract_line_items_local(text)

            rows = []
            for item in items:
                row = {
                    "File Name": uploaded_file.name,
                    "Invoice Number": item.get("invoice_number", ""),
                    "Invoice Date": item.get("invoice_date", ""),
                    "Client Name": item.get("client_name", ""),
                    "Product Name": item.get("product_name", ""),
                    "Quantity": item.get("quantity", ""),
                    "Product Price": item.get("product_price", ""),
                    "Currency": item.get("currency", ""),
                }
                rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download table as CSV",
                    data=csv_data,
                    file_name="invoice_line_items.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No line items were extracted. Check OCR text or invoice format.")
