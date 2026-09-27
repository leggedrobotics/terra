"""A loopback-only viewer service using Python's standard library."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

STATIC_DIR = Path(__file__).with_name("static")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
    "/static/style.css": ("style.css", "text/css; charset=utf-8"),
    "/static/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
}


def make_server(*, session=None, replay=None, port=8765):
    """Create a server; exactly one of a manual session or a replay is required."""
    if (session is None) == (replay is None):
        raise ValueError("Provide exactly one manual session or replay.")
    if replay is not None:
        from .replay import validate_replay

        validate_replay(replay)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, format, *args):
            pass

        def reply(
            self,
            status,
            body,
            content_type="application/json; charset=utf-8",
            download=None,
        ):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            if download:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{download}"'
                )
            self.end_headers()
            self.wfile.write(body)

        def same_origin(self):
            allowed_hosts = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            host = self.headers.get("Host", "")
            if host not in allowed_hosts:
                self.reply(
                    403, {"error": "Use the printed loopback URL to open the viewer."}
                )
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin not in {
                f"http://{h}" for h in allowed_hosts
            }:
                self.reply(403, {"error": "Cross-origin access is not allowed."})
                return False
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.reply(403, {"error": "Cross-site access is not allowed."})
                return False
            return True

        def do_GET(self):
            if not self.same_origin():
                return
            route = urlsplit(self.path).path
            if route in ASSETS:
                name, content_type = ASSETS[route]
                asset = STATIC_DIR / name
                if not asset.is_file():
                    self.reply(
                        503,
                        {
                            "error": "Viewer assets missing. Run npm ci && npm run build in terra/viewer3d/web."
                        },
                    )
                else:
                    self.reply(200, asset.read_bytes(), content_type)
            elif route in ("/api/session", "/api/replay"):
                with lock:
                    data = session.recorder.to_dict() if session is not None else replay
                if route == "/api/session":
                    self.reply(
                        200,
                        {
                            "mode": "manual" if session is not None else "replay",
                            "replay": data,
                        },
                    )
                else:
                    self.reply(200, data, download="terra-replay.json")
            elif route == "/favicon.ico":
                self.reply(204, b"", "image/x-icon")
            else:
                self.reply(404, {"error": "Not found."})

        def do_POST(self):
            if not self.same_origin():
                return
            route = urlsplit(self.path).path
            if route not in ("/api/action", "/api/reset"):
                self.reply(404, {"error": "Not found."})
                return
            if session is None:
                self.reply(405, {"error": "A recording cannot accept manual actions."})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.reply(415, {"error": "Send application/json."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Request body must be between 1 and 4096 bytes.")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Request must be a JSON object.")
                with lock:
                    if route == "/api/action":
                        if set(data) != {"action"}:
                            raise ValueError("Send exactly one action field.")
                        result = {"frame": session.step(data["action"])}
                    else:
                        if data:
                            raise ValueError("Reset expects an empty JSON object.")
                        result = session.reset()
                self.reply(200, result)
            except TimeoutError:
                self.reply(408, {"error": "Request body timed out."})
            except (ValueError, UnicodeDecodeError) as exc:
                self.reply(400, {"error": str(exc)})
            except RuntimeError as exc:
                self.reply(409, {"error": str(exc)})
            except Exception:
                # Tracebacks remain on the local terminal, never in a browser response.
                import traceback

                traceback.print_exc()
                self.reply(
                    500,
                    {
                        "error": "Terra could not complete this action. See the local terminal."
                    },
                )

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
