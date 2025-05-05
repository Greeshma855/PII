from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import onnxruntime as ort
from transformers import RobertaTokenizerFast
import io
import os
import gc
import requests

HF_REPO = "https://huggingface.co/Greeshma06/PiiMasking/resolve/main/"
ONNX_PATH = "roberta_pii.onnx"
TOKENIZER_DIR = "fine_tuned_roberta_pii"
TOKENIZER_FILES = [
    "special_tokens_map.json", "tokenizer_config.json",
    "tokenizer.json", "vocab.json", "merges.txt"
]

# Download ONNX model if not already present
if not os.path.exists(ONNX_PATH):
    print("🔽 Downloading ONNX model...")
    with open(ONNX_PATH, "wb") as f:
        f.write(requests.get(HF_REPO + ONNX_PATH).content)

# Download tokenizer files if not already present
os.makedirs(TOKENIZER_DIR, exist_ok=True)
for file in TOKENIZER_FILES:
    path = os.path.join(TOKENIZER_DIR, file)
    if not os.path.exists(path):
        print(f"🔽 Downloading {file}...")
        with open(path, "wb") as f:
            f.write(requests.get(HF_REPO + f"{TOKENIZER_DIR}/{file}").content)

# Set label map
LABEL_MAP = {
    1: "aadhaar_id", 2: "account_name", 3: "account_number", 4: "address",
    5: "age", 6: "amount", 7: "bank", 8: "bban", 9: "bic", 10: "bitcoin_address",
    11: "building_number", 12: "city", 13: "company_name", 14: "county",
    15: "credit_card_cvv", 16: "credit_card_issuer", 17: "credit_card_number",
    18: "currency", 19: "currency_code", 20: "currency_name", 21: "currency_symbol",
    22: "date", 23: "date_of_birth", 24: "driver_license", 25: "email",
    26: "ethereum_address", 27: "first_name", 28: "full_name", 29: "gender",
    30: "iban", 31: "ip", 32: "ipv4", 33: "ipv6", 34: "job_area",
    35: "job_descriptor", 36: "job_title", 37: "job_type", 38: "last_name",
    39: "latitude", 40: "license_plate", 41: "litecoin_address", 42: "longitude",
    43: "mac", 44: "masked_number", 45: "middle_name", 46: "pan_number",
    47: "password", 48: "phone_imei", 49: "phone_number", 50: "pin",
    51: "prefix", 52: "secondary_address", 53: "sex", 54: "ssn",
    55: "state", 56: "street", 57: "street_address", 58: "suffix",
    59: "time", 60: "url", 61: "user_agent", 62: "username", 63: "vehicle_vin",
    64: "vehicle_vrm", 65: "zip_code"
}

# Memoize the tokenizer and ONNX session using lru_cache for improved performance
from functools import lru_cache

@lru_cache()
def get_tokenizer():
    return RobertaTokenizerFast.from_pretrained(TOKENIZER_DIR)

@lru_cache()
def get_session():
    return ort.InferenceSession(ONNX_PATH)


# Main PII masking function
def mask_pii(text: str):
    tokenizer = get_tokenizer()
    session = get_session()
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        max_length=512,
        stride=128,
        truncation=True,
        padding=False
    )

    pii_spans = [0] * len(text)

    for input_ids, attention_mask, offset_mapping in zip(
        encoded["input_ids"], encoded["attention_mask"], encoded["offset_mapping"]
    ):
        ort_inputs = {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
        }
        logits = session.run(None, ort_inputs)[0]
        predicted_labels = logits.argmax(axis=-1)[0]

        for (start, end), label in zip(offset_mapping, predicted_labels):
            if start == end:
                continue
            if label in LABEL_MAP:
                for i in range(start, end):
                    if i < len(pii_spans):
                        pii_spans[i] = label

    # Build masked text and collect highlights
    result = []
    highlights = []
    i = 0
    while i < len(text):
        label = pii_spans[i]
        if label in LABEL_MAP:
            start = i
            while i < len(text) and pii_spans[i] == label:
                i += 1
            length = i - start
            result.append("X" * length)
            highlights.append({
                "start": start,
                "end": i,
                "label": LABEL_MAP[label]
            })
        else:
            result.append(text[i])
            i += 1

    masked_text = "".join(result)
    return masked_text, highlights

# FastAPI app definition
app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print("📩 File upload started...")
    content = await file.read()
    print(f"📄 File size: {len(content)} bytes")

    try:
        text = extract_text(file, content)
    except ValueError as e:
        return {"error": str(e)}

    print("🧠 Running inference...")
    masked_text, highlights = mask_pii(text)

    original_name = file.filename.rsplit(".", 1)[0]
    masked_filename = f"{original_name}_masked.txt"

    with open(masked_filename, "w", encoding="utf-8") as f:
        f.write("\n             PII Detection on text            \n")
        f.write("------ Original Text ------\n")
        f.write(text)
        f.write("\n\n------ Masked Text ------\n")
        f.write(masked_text)

    print(f"✅ Masking done. File saved as: {masked_filename}")
    
    # Clean up memory
    gc.collect()

    return {
        "message": "File masked successfully",
        "masked_filename": masked_filename,
        "masked_text": masked_text,
        "highlights": highlights
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    return FileResponse(filename, media_type="text/plain", filename=filename)
