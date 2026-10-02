"""Static checks of the Vercel layout (catches mistakes before a deploy)."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))


def test_function_and_rewrite():
    assert (ROOT / "api" / "index.py").is_file()
    fn = CONFIG["functions"]["api/index.py"]
    assert fn["includeFiles"] == "harmonics/**"
    assert (ROOT / "harmonics" / "__init__.py").is_file()
    assert {"source": "/api/(.*)", "destination": "/api/index"} in CONFIG["rewrites"]


def test_only_index_is_a_function():
    # Every .py under api/ becomes its own function on Vercel.
    assert sorted(p.name for p in (ROOT / "api").rglob("*.py")) == ["index.py"]


def test_public_assets_referenced_by_page_exist():
    public = ROOT / CONFIG["outputDirectory"]
    html = (public / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
    assert refs, "index.html references no static assets"
    for ref in refs:
        assert (public / ref.lstrip("/")).is_file(), ref
    js = (public / "static" / "app.js").read_text(encoding="utf-8")
    worklet = re.search(r"addModule\('([^']+)'\)", js).group(1)
    assert (public / worklet.lstrip("/")).is_file()


def test_dev_server_serves_public_like_the_cdn():
    from werkzeug.test import Client

    import dev_server

    c = Client(dev_server.application)
    assert c.get("/").status_code == 200
    r = c.get("/static/app.js")
    assert r.status_code == 200 and "javascript" in r.headers["Content-Type"]
    assert c.get("/static/style.css").headers["Content-Type"].startswith("text/css")
    assert c.get("/api/health").get_json() == {"status": "ok"}


def test_requirements_are_pinned_and_lean():
    reqs = (ROOT / "requirements.txt").read_text().split()
    assert all("==" in r for r in reqs)
    assert {r.split("==")[0] for r in reqs} == {"flask", "numpy", "scipy"}


def test_csp_header_present():
    headers = {h["key"]: h["value"] for rule in CONFIG["headers"] for h in rule["headers"]}
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert headers["Permissions-Policy"] == "microphone=(self)"
