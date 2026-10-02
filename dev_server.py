"""Local stand-in for Vercel: serves public/ like the CDN and /api/* from api/index.py.

    python dev_server.py      -> http://127.0.0.1:3000

Not deployed (listed in .vercelignore).  With the Vercel CLI, `vercel dev`
does the same thing.
"""

import os
import sys
from pathlib import Path

from flask import Flask, send_from_directory
from werkzeug.serving import run_simple

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
sys.path.insert(0, str(ROOT / "api"))

from index import app as api_app  # noqa: E402

cdn = Flask(__name__, static_folder=None)


@cdn.get("/")
def _index():
    return send_from_directory(PUBLIC, "index.html")


@cdn.get("/<path:name>")
def _public(name):
    return send_from_directory(PUBLIC, name)


def application(environ, start_response):
    """Route like vercel.json: /api/* to the function, everything else to public/."""
    target = api_app if environ.get("PATH_INFO", "").startswith("/api/") else cdn
    return target(environ, start_response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    print(f"Vercel-like dev server on http://127.0.0.1:{port}")
    run_simple("127.0.0.1", port, application, threaded=True)
