import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from PIL import Image, ImageEnhance, ImageOps

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp", "pdf"}
ALLOWED_LANGUAGES = {"fra", "eng", "fra+eng", "eng+fra"}

app = FastAPI(
    title="Foncier OCR Micro-service",
    description="Tesseract + pdftoppm — utilisé par Laravel OcrService en mode http (InfinityFree).",
    version="1.0.0",
)

# CORS — autorise InfinityFree + local dev ; resserrer via OCR_CORS_ORIGINS si besoin
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
        # accepte "fra" ou "eng" seul, sinon fallback
        if lang in {"fra", "eng"}:
            return lang
        return "fra"
    return lang


def _preprocess_image(image_path: str) -> str | None:
    """Prétraitement léger Pillow (mayaram/laravel-ocr-like) : grayscale -> upscale 2x si <2000px -> autocontrast -> contraste x1.5 -> sharpen.

    Binarisation/median désactivés par défaut (provoquaient 6807→6307). Activer via OCR_BINARIZE_THRESHOLD=140 si besoin.
    """
    try:
        img = Image.open(image_path)

        if img.mode != "L":
            img = img.convert("L")

        if img.width < 2000:
            img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)

        img = ImageOps.autocontrast(img, cutoff=0.5)
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = img.filter(ImageFilter.SHARPEN)

        # Opt-in : median + binarisation uniquement si seuil explicite
        threshold = int(os.getenv("OCR_BINARIZE_THRESHOLD", "0"))
        if 0 < threshold < 255:
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = img.point(lambda p, t=threshold: 255 if p > t else 0, mode="L")

        out = tempfile.mktemp(prefix="ocr_pre_", suffix=".png")
        img.save(out, "PNG", dpi=(300, 300))

        return out
    except Exception:
        return None


def _convert_pdf_to_png(pdf_path: str) -> str:
    """Convertit la première page du PDF en PNG via pdftoppm. Retourne chemin PNG."""
    pdftoppm = os.getenv("PDFTOPPM_PATH", "pdftoppm")
    tmp_png_base = tempfile.mktemp(prefix="ocr_")  # pdftoppm ajoute .png
    cmd = f"{shlex.quote(pdftoppm)} -png -r 300 -singlefile {shlex.quote(pdf_path)} {shlex.quote(tmp_png_base)} 2>/dev/null"
    result = subprocess.run(cmd, shell=True)
    generated = tmp_png_base + ".png"
    if result.returncode != 0 or not Path(generated).exists():
        raise RuntimeError(f"pdftoppm failed (code={result.returncode}) for {pdf_path}")
    return generated


def _run_tesseract(image_path: str, language: str) -> tuple[str, float | None]:
    """Lance tesseract avec prétraitement Pillow. PSM configurable via OCR_PSM (défaut 4 single column)."""
    tesseract_bin = os.getenv("TESSERACT_BINARY", "tesseract")
    psm = os.getenv("OCR_PSM", "4").strip()  # 4 = single column (CIP), 6 = uniform block — testé meilleur sur CIP
    if psm not in {str(i) for i in range(14)}:
        psm = "4"

    # Prétraitement Pillow (upscale + contraste + binarisation) — fallback image brute si échec
    preprocessed: str | None = None
    ocr_image = image_path
    if os.getenv("OCR_PREPROCESS", "1").strip() not in {"0", "false", "no"}:
        preprocessed = _preprocess_image(image_path)
        if preprocessed and Path(preprocessed).exists():
            ocr_image = preprocessed

    try:
        cmd = [tesseract_bin, ocr_image, "stdout", "-l", language, "--psm", psm, "--oem", "1"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"tesseract failed: {proc.stderr[:500]}")
        text = proc.stdout.strip()

        # Fallback PSM 3 si texte trop court
        if len(text) < 20 and psm in {"4", "6"}:
            try:
                fallback_cmd = [tesseract_bin, ocr_image, "stdout", "-l", language, "--psm", "3", "--oem", "1"]
                fb_proc = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=30)
                if fb_proc.returncode == 0 and len(fb_proc.stdout.strip()) > len(text):
                    text = fb_proc.stdout.strip()
            except Exception:
                pass

        confidence = None
        try:
            tsv_cmd = [tesseract_bin, ocr_image, "stdout", "-l", language, "--psm", psm, "--oem", "1", "tsv"]
            tsv_proc = subprocess.run(tsv_cmd, capture_output=True, text=True, timeout=30)
            if tsv_proc.returncode == 0:
                lines = tsv_proc.stdout.strip().splitlines()
                confs: list[int] = []
                for line in lines[1:]:  # skip header
                    parts = line.split("\t")
                    if len(parts) >= 11:
                        try:
                            c = int(parts[10])
                            if c >= 0:
                                confs.append(c)
                        except ValueError:
                            continue
                if confs:
                    confidence = round(sum(confs) / len(confs), 2)
        except Exception:
            pass
        return text, confidence
    finally:
        if preprocessed and Path(preprocessed).exists():
            try:
                Path(preprocessed).unlink()
            except Exception:
                pass


@app.get("/health")
def health() -> dict:
    tesseract_bin = os.getenv("TESSERACT_BINARY", "tesseract")
    pdftoppm_bin = os.getenv("PDFTOPPM_PATH", "pdftoppm")
    checks: dict[str, str] = {}
    for name, bin_path in [("tesseract", tesseract_bin), ("pdftoppm", pdftoppm_bin)]:
        proc = subprocess.run(f"which {shlex.quote(bin_path)} 2>/dev/null; {shlex.quote(bin_path)} --version 2>&1 | head -n1", shell=True, capture_output=True, text=True)
        checks[name] = proc.stdout.strip()[:200] or "not found"
    return {"status": "ok", "service": "foncier-ocr", "checks": checks}


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    language: str = Form(default="fra"),
    _: None = Depends(verify_token),
) -> JSONResponse:
    lang = _sanitize_language(language)
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        # tente de détecter via content_type
        if file.content_type and "pdf" in file.content_type:
            ext = "pdf"
        elif ext == "":
            ext = "png"
        else:
            raise HTTPException(status_code=400, detail=f"Extension non supportée: .{ext}")

    # Sauvegarde temporaire
    suffix = f".{ext}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Fichier vide")
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 10MB)")
        tmp.write(content)
        tmp_path = tmp.name

    generated_png: str | None = None
    try:
        image_path = tmp_path
        if ext == "pdf":
            generated_png = _convert_pdf_to_png(tmp_path)
            image_path = generated_png

        text, confidence = _run_tesseract(image_path, lang)

        return JSONResponse(
            {
                "text": text,
                "confidence": confidence,
                "language": lang,
                "filename": filename,
            }
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="OCR timeout")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for p in [tmp_path, generated_png]:
            if p and Path(p).exists():
                try:
                    Path(p).unlink()
                except Exception:
                    pass


def _ocr_and_callback(
    tmp_path: str,
    generated_png: str | None,
    lang: str,
    filename: str,
    callback_url: str,
    callback_token: str | None,
    job_id: str | None,
    document_id: str | None = None,
) -> None:
    """Tâche de fond : OCR puis POST vers callback_url."""
    try:
        image_path = tmp_path
        if generated_png and Path(generated_png).exists():
            image_path = generated_png
        elif Path(tmp_path).suffix.lower() == ".pdf":
            # Si le PDF n'a pas été converti avant (fallback)
            try:
                generated_png = _convert_pdf_to_png(tmp_path)
                image_path = generated_png
            except Exception as e:
                _post_callback(callback_url, callback_token, {"jobId": job_id, "documentId": document_id, "error": f"pdftoppm failed: {e}", "status": "error"})
                return

        text, confidence = _run_tesseract(image_path, lang)
        payload: dict = {"jobId": job_id, "text": text, "confidence": confidence, "language": lang, "filename": filename, "status": "ok"}
        if document_id:
            payload["documentId"] = document_id
        _post_callback(callback_url, callback_token, payload)
    except subprocess.TimeoutExpired:
        _post_callback(callback_url, callback_token, {"jobId": job_id, "documentId": document_id, "error": "OCR timeout", "status": "error"})
    except Exception as exc:
        _post_callback(callback_url, callback_token, {"jobId": job_id, "documentId": document_id, "error": str(exc)[:500], "status": "error"})
    finally:
        for p in [tmp_path, generated_png]:
            if p and Path(p).exists():
                try:
                    Path(p).unlink()
                except Exception:
                    pass


def _post_callback(callback_url: str, callback_token: str | None, payload: dict) -> None:
    headers = {"Content-Type": "application/json"}
    if callback_token:
        headers["Authorization"] = f"Bearer {callback_token}"
    try:
        # Fire-and-forget, timeout 10s pour le POST callback
        httpx.post(callback_url, json=payload, headers=headers, timeout=10)
    except Exception:
        pass


@app.post("/receive")
async def receive(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form(default="fra"),
    callback_url: str = Form(...),
    job_id: str = Form(default=""),
    callback_token: str = Form(default=""),
    document_id: str = Form(default=""),
    _: None = Depends(verify_token),
) -> JSONResponse:
    """Réception async : 202 immédiat puis OCR en tâche de fond qui rappelle callback_url."""
    lang = _sanitize_language(language)
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        if file.content_type and "pdf" in file.content_type:
            ext = "pdf"
        elif ext == "":
            ext = "png"
        else:
            raise HTTPException(status_code=400, detail=f"Extension non supportée: .{ext}")

    suffix = f".{ext}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Fichier vide")
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 10MB)")
        tmp.write(content)
        tmp_path = tmp.name

    generated_png: str | None = None
    # Pré-conversion PDF pour ne pas bloquer le thread principal sur pdftoppm
    if ext == "pdf":
        try:
            generated_png = _convert_pdf_to_png(tmp_path)
        except Exception as exc:
            # On laisse la tâche de fond gérer l'erreur et notifier le callback
            pass

    # Validation callback_url
    if not callback_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="callback_url invalide")

    background_tasks.add_task(
        _ocr_and_callback,
        tmp_path,
        generated_png,
        lang,
        filename,
        callback_url,
        callback_token or None,
        job_id or None,
        document_id or None,
    )

    return JSONResponse({"received": True, "jobId": job_id or None, "filename": filename}, status_code=202)


@app.get("/")
def root() -> dict:
    return {"service": "foncier-ocr", "docs": "/docs", "health": "/health", "extract": "POST /extract"}
