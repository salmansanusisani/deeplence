import io
import os
import tempfile
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.provenance import inspect_c2pa
from app.sightengine import analyze_image as analyze_with_sightengine
from app.video import analyze_video

app = FastAPI(title="AuthentiCheck - AI Content Detector")

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    raw_bytes = await file.read()

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than the 25 MB analysis limit")

    try:
        if ext in IMAGE_EXTENSIONS:
            pil_image = Image.open(io.BytesIO(raw_bytes))
            pil_image.load()
            provenance = inspect_c2pa(raw_bytes, ext)
            result = analyze_with_sightengine(raw_bytes, filename)
            result["provenance"] = provenance
            result["media_type"] = "image"
            return JSONResponse(result)

        elif ext in VIDEO_EXTENSIONS:
            provenance = inspect_c2pa(raw_bytes, ext)
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            try:
                result = analyze_video(tmp_path, raw_bytes, provenance)
            finally:
                os.unlink(tmp_path)
            result["media_type"] = "video"
            return JSONResponse(result)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "image_detector": "sightengine_genai"}
