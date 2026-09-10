# Foncier OCR Micro-service

Micro-service FastAPI qui expose **Tesseract OCR** (`fra` + `eng`) + **Poppler** (`pdftoppm`) pour l'application Laravel `foncier-benin-projet-memoire` en mode `OCR_MODE=http`.

Pensé pour **InfinityFree** (mutualisé sans binaires système) — Laravel délègue l'extraction via HTTP. Déployable sur **Render** (Docker) ou en local via `docker compose`.

## Architecture

```
Laravel (InfinityFree, OCR_MODE=http)
   │  POST /extract + Bearer token + multipart file
   ▼
ocr_docker (FastAPI:8000)
   ├─ pdftoppm -png -r 300 -singlefile  (si PDF)
   └─ tesseract -l fra --psm 3  → {text, confidence}
```

Laravel utilise `config/ocr.php:8` (`OCR_MODE`, `OCR_SERVICE_URL`, `OCR_SERVICE_TOKEN`) et `app/Services/OcrService.php:extraireViaHttp`.

## Prérequis

- Docker + Compose (local)
- Render : repo Git indépendant (ce dossier)

## Local

```bash
cp .env.example .env
docker compose up --build
curl http://localhost:8001/health
curl -F file=@tests/fixtures/sample.png http://localhost:8001/extract
curl -F file=@tests/fixtures/sample.pdf -F language=fra http://localhost:8001/extract
# Avec token
OCR_TOKEN=secret docker compose up --build
curl -H "Authorization: Bearer secret" -F file=@sample.png http://localhost:8001/extract
```

Laravel `.env` local :

```env
OCR_MODE=http
OCR_SERVICE_URL=http://localhost:8001
OCR_SERVICE_TOKEN=
OCR_SERVICE_TIMEOUT=60
```

## Render (Docker)

1. Créer un **Web Service** → Connecter ce repo → **Docker** (détecte `Dockerfile`).
2. Render injecte `$PORT` → `Dockerfile` lance `uvicorn --port ${PORT:-8000}`.
3. Variables d'environnement Render :
   - `OCR_TOKEN` = token secret (optionnel mais recommandé)
   - `OCR_CORS_ORIGINS` = `https://ton-site.infinityfreeapp.com` (ou `*` en test)
   - `TESSERACT_BINARY` / `PDFTOPPM_PATH` laissés par défaut
4. Déployer → URL genre `https://foncier-ocr.onrender.com` → renseigner dans Laravel InfinityFree :

```env
OCR_MODE=http
OCR_SERVICE_URL=https://foncier-ocr.onrender.com
OCR_SERVICE_TOKEN=ton_secret_render
OCR_SERVICE_TIMEOUT=60
OCR_SERVICE_LANGUAGE=fra
```

> Astuce Render free : le service s'endort après inactivité (~15 min) → première requête lente. Prévoir un healthcheck/ping ou passer en plan payant si besoin.

## API

| Endpoint | Méthode | Auth | Body | Réponse |
|---|---|---|---|---|
| `/` | GET | non | — | `{service, docs, health}` |
| `/health` | GET | non | — | `{status, checks: {tesseract, pdftoppm}}` |
| `/docs` | GET | non | — | Swagger UI |
| `/extract` | POST | Bearer si `OCR_TOKEN` défini | `file` (multipart, max 10 MB), `language=fra\|eng\|fra+eng` | `{text, confidence, language, filename}` |

Erreurs : `401` token invalide, `400` extension vide, `413` >10 MB, `504` timeout, `500` tesseract/pdftoppm.

## Sécurité

- Définir `OCR_TOKEN` côté Render et `OCR_SERVICE_TOKEN` côté Laravel.
- Restreindre `OCR_CORS_ORIGINS` en prod.
- Limite 10 MB, timeout 60s, extension whitelist `png/jpg/jpeg/tiff/bmp/pdf`.

## Dépannage

```bash
docker compose logs -f ocr
curl http://localhost:8001/health
# Laravel
php artisan tinker
>>> app(App\Services\OcrService::class)->extraireTexte(storage_path('app/public/test.png'));
```

Voir aussi `../foncier-benin-projet-memoire/DEPLOY_INFINITYFREE.md` côté Laravel.
