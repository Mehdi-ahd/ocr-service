import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp", "pdf"}
ALLOWED_LANGUAGES = {"fra", "eng", "fra+eng", "eng+fra"}

app = FastAPI(
    title="Foncier OCR Micro-service",
    description="OCRmyPDF exclusif (force-ocr + sidecar) — utilisé par Laravel OcrService en mode http (InfinityFree).",
    version="3.0.0-batch",
)

cors_origins = [o.strip() for o in os.getenv("OCR_CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("OCR_TOKEN", "").strip()
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid token")


def _sanitize_language(lang: str) -> str:
    lang = lang.strip().lower()
    if lang not in ALLOWED_LANGUAGES:
        if lang in {"fra", "eng"}:
            return lang
        return "fra"
    return lang


def _ocr_pdf_with_ocrmypdf(pdf_path: str, language: str) -> str:
    """OCR exclusif via OCRmyPDF --force-ocr + sidecar. Lève RuntimeError si échec."""
    sidecar = tempfile.mktemp(prefix="ocr_sidecar_", suffix=".txt")
    output_pdf = tempfile.mktemp(prefix="ocr_out_", suffix=".pdf")
    lang = language.strip() or "fra"

    # OCRmyPDF pipeline : force-ocr (ignore texte existant), deskew/clean auto, tesseract fra
    # --optimize 0 : pas de recompression (qualité max) ; --oversample 300 pour images basse def
    cmd = [
        "ocrmypdf",
        "--force-ocr",
        "-l", lang,
        "--optimize", "0",
        "--oversample", "300",
        "--output-type", "pdf",
        "--sidecar", sidecar,
        "--tesseract-timeout", "90",
        "--jobs", "1",
        pdf_path,
        output_pdf,
    ]
    extra = os.getenv("OCR_OCRMYPDF_ARGS", "").strip()
    if extra:
        # permet d'injecter --deskew --clean etc sans toucher au code
        cmd[1:1] = shlex.split(extra)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"ocrmypdf failed (code={proc.returncode}): {(proc.stderr or proc.stdout)[:800]}")
        if not Path(sidecar).exists():
            raise RuntimeError("ocrmypdf: sidecar manquant")
        text = Path(sidecar).read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            raise RuntimeError("ocrmypdf: sidecar vide")
        return text
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ocrmypdf timeout: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError(f"ocrmypdf not found: {e}") from e
    finally:
        for p in (sidecar, output_pdf):
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass


def _ocr_image_via_ocrmypdf(image_path: str, language: str) -> str:
    """Image -> PDF (img2pdf) -> OCRmyPDF sidecar. Exclusif."""
    try:
        import img2pdf
    except ImportError as e:
        raise RuntimeError(f"img2pdf missing: {e}") from e

    pdf_tmp = tempfile.mktemp(prefix="ocr_img_", suffix=".pdf")
    try:
        with open(image_path, "rb") as f:
            pdf_bytes = img2pdf.convert(f)
        Path(pdf_tmp).write_bytes(pdf_bytes)
        return _ocr_pdf_with_ocrmypdf(pdf_tmp, language)
    finally:
        try:
            if Path(pdf_tmp).exists():
                Path(pdf_tmp).unlink()
        except Exception:
            pass


def _extract_text(file_path: str, ext: str, language: str) -> tuple[str, float | None]:
    """Route exclusivement OCRmyPDF selon l'extension. Retourne (text, None)."""
    ext = ext.lower().lstrip(".")
    if ext == "pdf":
        text = _ocr_pdf_with_ocrmypdf(file_path, language)
        return text, None
    if ext in {"png", "jpg", "jpeg", "tiff", "bmp"}:
        text = _ocr_image_via_ocrmypdf(file_path, language)
        return text, None
    raise RuntimeError(f"Extension non supportée pour OCRmyPDF: .{ext}")


def _ocr_and_callback(
    tmp_path: str,
    lang: str,
    filename: str,
    callback_url: str,
    callback_token: str | None,
    job_id: str | None,
    document_id: str | None = None,
) -> None:
    """Tâche de fond exclusive OCRmyPDF puis POST vers callback_url."""
    try:
        ext = Path(tmp_path).suffix.lower().lstrip(".")
        # Si l'extension a été perdue (tmp sans suffix), deviner via filename
        if ext not in ALLOWED_EXTENSIONS:
            ext = Path(filename).suffix.lower().lstrip(".") or "pdf"
        text, confidence = _extract_text(tmp_path, ext, lang)
        payload: dict = {"jobId": job_id, "text": text, "confidence": confidence, "language": lang, "filename": filename, "status": "ok"}
        if document_id:
            payload["documentId"] = document_id
        _post_callback(callback_url, callback_token, payload)
    except subprocess.TimeoutExpired:
        _post_callback(callback_url, callback_token, {"jobId": job_id, "documentId": document_id, "error": "OCR timeout", "status": "error"})
    except Exception as exc:
        _post_callback(callback_url, callback_token, {"jobId": job_id, "documentId": document_id, "error": str(exc)[:800], "status": "error"})
    finally:
        if Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


def _post_callback(callback_url: str, callback_token: str | None, payload: dict) -> None:
    headers = {"Content-Type": "application/json"}
    if callback_token:
        headers["Authorization"] = f"Bearer {callback_token}"
    try:
        httpx.post(callback_url, json=payload, headers=headers, timeout=10)
    except Exception:
        pass


@app.post("/receive")
async def receive(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    language: str = Form(default="fra"),
    callback_url: str = Form(...),
    job_id: str = Form(default=""),
    callback_token: str = Form(default=""),
    document_ids: Optional[List[str]] = Form(default=None),
    _: None = Depends(verify_token),
) -> JSONResponse:
    """Réception async batch : 202 immédiat puis OCRmyPDF en tâche de fond qui rappelle callback_url.
    Accepte un ou plusieurs fichiers. Retourne une liste de jobId.
    """
    lang = _sanitize_language(language)
    if not callback_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="callback_url invalide")

    total_files = len(files)
    jobs = []

    for idx, file in enumerate(files):
        filename = file.filename or "upload"
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in ALLOWED_EXTENSIONS:
            if file.content_type and "pdf" in file.content_type:
                ext = "pdf"
            elif ext == "":
                ext = "png"
            else:
                raise HTTPException(status_code=400, detail=f"Extension non supportée: .{ext}")

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Fichier vide")
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 10MB)")

        suffix = f".{ext}"
        tmp_path = tempfile.mktemp(prefix="ocr_batch_", suffix=suffix)
        with open(tmp_path, "wb") as f:
            f.write(content)

        file_job_id = job_id if total_files == 1 else f"{job_id}_{idx}" if job_id else None
        file_document_id = None
        if document_ids and idx < len(document_ids):
            file_document_id = document_ids[idx] or None

        background_tasks.add_task(
            _ocr_and_callback,
            tmp_path,
            lang,
            filename,
            callback_url,
            callback_token or None,
            file_job_id,
            file_document_id,
        )
        jobs.append({"jobId": file_job_id, "filename": filename})

    return JSONResponse({"received": True, "jobs": jobs}, status_code=202)


@app.get("/")
def root() -> dict:
    return {"service": "foncier-ocr", "docs": "/docs", "health": "/health", "extract": "POST /extract", "receive": "POST /receive (batch)", "version": "3.0.0-batch"}
