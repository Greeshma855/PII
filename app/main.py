from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from transformers import RobertaTokenizerFast
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import io, os, gc
from functools import lru_cache
import tempfile

app = FastAPI()

# Constants
REPO_ID = "Greeshma06/PiiMasking"
ONNX_FILENAME = "roberta_pii.onnx"
TOKENIZER_SUBFOLDER = "fine_tuned_roberta_pii"

# Download ONNX model from Hugging Face Hub
ONNX_PATH = hf_hub_download(repo_id=REPO_ID, filename=ONNX_FILENAME)

# Label map
LABEL_MAP = {
    1: "aadhaar_id", 2: "account_name", 3: "account_number", 4: "address", 5: "age",
    6: "amount", 7: "bank", 8: "bban", 9: "bic", 10: "bitcoin_address", 11: "building_number",
    12: "city", 13: "company_name", 14: "county", 15: "credit_card_cvv", 16: "credit_card_issuer",
    17: "credit_card_number", 18: "currency", 19: "currency_code", 20: "currency_name",
    21: "currency_symbol", 22: "date", 23: "date_of_birth", 24: "driver_license", 25: "email",
    26: "ethereum_address", 27: "first_name", 28: "full_name", 29: "gender", 30: "iban",
    31: "ip", 32: "ipv4", 33: "ipv6", 34: "job_area", 35: "job_descriptor", 36: "job_title",
    37: "job_type", 38: "last_name", 39: "latitude", 40: "license_plate", 41: "litecoin_address",
    42: "longitude", 43: "mac", 44: "masked_number", 45: "middle_name", 46: "pan_number",
    47: "password", 48: "phone_imei", 49: "phone_number", 50: "pin", 51: "prefix",
    52: "secondary_address", 53: "sex", 54: "ssn", 55: "state", 56: "street",
    57: "street_address", 58: "suffix", 59: "time", 60: "url", 61: "user_agent",
    62: "username", 63: "vehicle_vin", 64: "vehicle_vrm", 65: "zip_code"
}

# Cached loading
@lru_cache()
# def get_tokenizer():
#     return RobertaTokenizerFast.from_pretrained(f"{REPO_ID}/{TOKENIZER_SUBFOLDER}")
def get_tokenizer():
    return RobertaTokenizerFast.from_pretrained(
        "Greeshma06/PiiMasking",
        subfolder="fine_tuned_roberta_pii"
    )



@lru_cache()
def get_session():
    return ort.InferenceSession(ONNX_PATH)

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

@app.get("/")
async def root():
    return {"message": "API is working"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unreadable text file")

    masked_text, highlights = mask_pii(text)

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix="_masked.txt") as tmp_file:
        tmp_file.write("\n             PII Detection on text            \n")
        tmp_file.write("------ Original Text ------\n")
        tmp_file.write(text)
        tmp_file.write("\n\n------ Masked Text ------\n")
        tmp_file.write(masked_text)
        temp_filename = tmp_file.name

    gc.collect()

    return {
        "message": "File masked successfully",
        "masked_filename": os.path.basename(temp_filename),
        "masked_text": masked_text,
        "highlights": highlights
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(temp_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(temp_path, media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
