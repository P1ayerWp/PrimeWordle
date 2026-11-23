"""Zero-dependency HTTP server for Gödel-Wordle (Prime Decoding) Web UI & JSON API."""

from __future__ import annotations

import json
import random
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from engine.deduction import analyze_session
from engine.game import (
    GameConfig,
    GameSession,
    generate_random_slot,
    load_saved_games,
    save_game_jsonl,
)
from engine.tokenizer import PrimeWordleTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_FILE = ROOT_DIR / "data" / "human_games.jsonl"

_LOCK = threading.Lock()
_SESSIONS: dict[str, GameSession] = {}
_TOKENIZER = PrimeWordleTokenizer()


def _load_or_get_session(game_id: str) -> GameSession | None:
    if game_id in _SESSIONS:
        return _SESSIONS[game_id]
    for record in load_saved_games(DATA_FILE):
        if record.get("game_id") == game_id:
            session = GameSession.from_dict(record)
            _SESSIONS[game_id] = session
            return session
    return None


def _get_latest_session() -> GameSession | None:
    games = load_saved_games(DATA_FILE)
    if not games:
        return None
    latest_id = games[0].get("game_id")
    if latest_id in _SESSIONS:
        return _SESSIONS[latest_id]
    session = GameSession.from_dict(games[0])
    _SESSIONS[session.game_id] = session
    return session


def _session_payload(
    session: GameSession,
    include_target: bool = False,
    include_tokens: bool = False,
) -> dict[str, Any]:
    data = session.to_dict(include_target=include_target)
    token_ids = _TOKENIZER.encode_session(session)
    data["token_count"] = len(token_ids)
    data["deduction"] = analyze_session(session)
    if include_tokens:
        data["token_ids"] = token_ids
        data["tokens"] = _TOKENIZER.decode(token_ids)
    return data


class PrimeWordleHandler(BaseHTTPRequestHandler):
    server_version = "PrimeWordle/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output clean
        pass

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "File not found"}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        include_tokens = qs.get("include_tokens", ["0"])[0] in ("1", "true", "yes")

        if path == "/" or path == "/index.html":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            if ".." in rel or "/" in rel or "\\" in rel:
                self._send_json({"error": "Invalid path"}, HTTPStatus.BAD_REQUEST)
                return
            mime_map = {
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".html": "text/html; charset=utf-8",
                ".svg": "image/svg+xml",
            }
            target = STATIC_DIR / rel
            self._send_file(target, mime_map.get(target.suffix, "application/octet-stream"))
            return

        if path == "/api/games":
            with _LOCK:
                games = load_saved_games(DATA_FILE)
            summaries = [
                {
                    "game_id": g.get("game_id"),
                    "created_at": g.get("created_at"),
                    "config": g.get("config"),
                    "status": g.get("status"),
                    "turns": g.get("turns", len(g.get("history", []))),
                    "peeked": g.get("peeked", False),
                }
                for g in games
            ]
            self._send_json({"games": summaries})
            return

        if path == "/api/games/latest":
            with _LOCK:
                session = _get_latest_session()
                if session is None:
                    self._send_json({"error": "No games found"}, HTTPStatus.NOT_FOUND)
                    return
                payload = _session_payload(session, include_tokens=include_tokens)
            self._send_json(payload)
            return

        m = re.fullmatch(r"/api/games/([a-zA-Z0-9_-]+)(?:/state)?", path)
        if m:
            game_id = m.group(1)
            with _LOCK:
                session = _load_or_get_session(game_id)
                if session is None:
                    self._send_json({"error": "Game not found"}, HTTPStatus.NOT_FOUND)
                    return
                payload = _session_payload(session, include_tokens=include_tokens)
            self._send_json(payload)
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, HTTPStatus.BAD_REQUEST)
            return

        if path == "/api/games":
            try:
                config = GameConfig.from_dict(body)
                seed = body.get("seed")
                seed_int = int(seed) if seed not in (None, "") else None
                session = GameSession(config=config, seed=seed_int)
                with _LOCK:
                    _SESSIONS[session.game_id] = session
                    save_game_jsonl(session, DATA_FILE)
                    payload = _session_payload(session)
                self._send_json(payload, HTTPStatus.CREATED)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        m = re.fullmatch(r"/api/games/([a-zA-Z0-9_-]+)/guess", path)
        if m:
            game_id = m.group(1)
            with _LOCK:
                session = _load_or_get_session(game_id)
                if session is None:
                    self._send_json({"error": "Game not found"}, HTTPStatus.NOT_FOUND)
                    return
                try:
                    guess_slots = body.get("guess_slots")
                    if not isinstance(guess_slots, list):
                        raise ValueError("Missing 'guess_slots' list.")
                    turn = session.submit_guess(guess_slots)
                    save_game_jsonl(session, DATA_FILE)
                    payload = _session_payload(session)
                    payload["latest_turn"] = turn.to_dict()
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
            self._send_json(payload)
            return

        m = re.fullmatch(r"/api/games/([a-zA-Z0-9_-]+)/peek", path)
        if m:
            game_id = m.group(1)
            with _LOCK:
                session = _load_or_get_session(game_id)
                if session is None:
                    self._send_json({"error": "Game not found"}, HTTPStatus.NOT_FOUND)
                    return
                session.peeked = True
                save_game_jsonl(session, DATA_FILE)
                payload = _session_payload(session, include_target=True)
            self._send_json(payload)
            return

        m = re.fullmatch(r"/api/games/([a-zA-Z0-9_-]+)/notes", path)
        if m:
            game_id = m.group(1)
            with _LOCK:
                session = _load_or_get_session(game_id)
                if session is None:
                    self._send_json({"error": "Game not found"}, HTTPStatus.NOT_FOUND)
                    return
                if "symbol_marks" in body and isinstance(body["symbol_marks"], dict):
                    session.notes["symbol_marks"] = body["symbol_marks"]
                if "scratchpad" in body and isinstance(body["scratchpad"], str):
                    session.notes["scratchpad"] = body["scratchpad"]
                save_game_jsonl(session, DATA_FILE)
            self._send_json({"ok": True, "notes": session.notes})
            return

        m = re.fullmatch(r"/api/games/([a-zA-Z0-9_-]+)/random_guess", path)
        if m:
            game_id = m.group(1)
            with _LOCK:
                session = _load_or_get_session(game_id)
                if session is None:
                    self._send_json({"error": "Game not found"}, HTTPStatus.NOT_FOUND)
                    return
                rng = random.SystemRandom()
                slots = [
                    generate_random_slot(session.config, rng)
                    for _ in range(session.config.code_length)
                ]
            self._send_json({"guess_slots": slots})
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), PrimeWordleHandler)
    print(f"PrimeWordle server running at http://{host}:{port}")
    print(f"Game logs saved to: {DATA_FILE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
