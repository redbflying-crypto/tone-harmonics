# Tone Harmonics — Vercel edition

The Voice Harmonics Analyzer, adapted to deploy on **Vercel**. The Render
edition lives separately in `../code`.

| Part | On Vercel |
|---|---|
| `public/` (page, JavaScript, CSS, AudioWorklet) | Served as static files by Vercel's CDN |
| `api/index.py` (Flask) | One Python serverless function; `vercel.json` rewrites `/api/*` to it |
| `harmonics/` | Analysis package, bundled into the function (`includeFiles`) |

Recording, playback, rename and delete happen in the browser. Takes are stored
in the browser's IndexedDB. The function only receives a WAV file when you
click **Analyse**, and it never stores it.

## What differs from the Render edition

- **40 s maximum per take.** Vercel functions accept request bodies up to
  4.5 MB, and 40 s of 48 kHz 16-bit WAV is about 3.84 MB. Takes recorded above
  48 kHz are resampled to 48 kHz in the browser. Larger takes get a clear
  message instead of a failed upload.
- **No gunicorn or Procfile:** Vercel runs the Flask app itself.
- **`maxDuration` is 60 s** for the function. A 40 s take analyses in a few
  seconds.
- **Cold starts:** the first request after a pause loads NumPy and SciPy, which
  takes a few seconds.

## Run locally

```bash
cd "D:\voice harmonics\vercel"
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python dev_server.py
```

Open **http://localhost:3000**. `dev_server.py` imitates Vercel: it serves
`public/` at `/` and the function at `/api/*`. It is not deployed. With the
Vercel CLI installed, `vercel dev` does the same.

Tests (no microphone needed):

```bash
.venv\Scripts\python -m pytest -q
```

## Deploy

1. Push this folder to GitHub (repository `redbflying-crypto/tone-harmonics`).
2. On https://vercel.com, click **Add New → Project**, import
   `tone-harmonics` and keep the defaults. Leave the Framework Preset set to
   **Other**: `vercel.json` already sets the output directory, the function
   and the rewrites. Then click **Deploy**.
3. Every later `git push` to `main` redeploys automatically.

Vercel serves the site over HTTPS, which browsers require before they allow
the microphone.

## Files

```
api/index.py           Flask function: /api/health, /api/analyze
harmonics/             YIN pitch tracking, segmentation, harmonic measurement
public/index.html      page
public/static/         app.js, style.css, recorder-worklet.js
vercel.json            output dir, function settings, rewrites, security headers
requirements.txt       flask, numpy, scipy (pinned)
dev_server.py          local stand-in for Vercel (not deployed)
tests/                 pytest suite, including layout and config checks
```
