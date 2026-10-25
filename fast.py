import io
import os

from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import FileResponse
from PIL import Image

# Import from the PyTorch file (same folder). Importing does NOT train anything.
from garbage_classifier import CLASS_NAMES
from garbage_classifier import load_model
from garbage_classifier import predict_image

# Load the trained model ONCE when the server starts
model = load_model()

app = FastAPI(title="Garbage Classifier API")

# index.html lives in the "static" folder next to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(BASE_DIR, "static", "index.html")

# Reject very large uploads (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


@app.get("/")
def home():
    return FileResponse(index_path)


@app.get("/health")
def health():
    # Lightweight endpoint for uptime monitors (keeps the server awake)
    return {"status": "ok"}


@app.get("/classes")
def classes():
    # The web page can use this to show the list of classes
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Only accept images
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    # Read the uploaded file and check its size
    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image is too large (max 10 MB)")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this image")

    # All the PyTorch work happens inside predict_image()
    return predict_image(model, image)