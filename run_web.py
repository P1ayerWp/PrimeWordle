"""One-click launcher for the Gödel-Wordle (Prime Decoding) Web App."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from web.server import run_server


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch PrimeWordle Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open browser automatically"
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    if _is_port_open(args.host, args.port):
        print(f"PrimeWordle is already running at {url} — opening browser...")
        if not args.no_browser:
            webbrowser.open(url)
        return

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
