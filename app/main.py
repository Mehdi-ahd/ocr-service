import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    """Lance tesseract et retourne (texte, confidence)."""
    tesseract_bin = os.getenv("TESSERACT_BINARY", "tesseract")
    # --psm 3 (auto), --oem 1 (LSTM) par défaut
    cmd = [tesseract_bin, image_path, "stdout", "-l", language, "--psm", "3"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed: {proc.stderr[:500]}")
    text = proc.stdout.strip()
    # Optionnel : extraire confidence via tsv si besoin
    confidence = None
    try:
        tsv_cmd = [tesseract_bin, image_path, "stdout", "-l", language, "--psm", "3", "tsv"]
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


@app.get("/")
def root() -> dict:
    return {"service": "foncier-ocr", "docs": "/docs", "health": "/health", "extract": "POST /extract"}
