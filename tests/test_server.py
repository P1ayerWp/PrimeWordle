"""End-to-end API smoke tests for the PrimeWordle Web server."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from web import server as web_server


class TestWebServerAPI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_data = web_server.DATA_FILE
        web_server.DATA_FILE = Path(self._tmp.name) / "test_games.jsonl"
        web_server._SESSIONS.clear()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_server.PrimeWordleHandler)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        web_server.DATA_FILE = self._orig_data
        web_server._SESSIONS.clear()
        self._tmp.cleanup()

    def _req(self, path: str, method: str = "GET", body: dict | None = None) -> tuple[int, bytes]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()

    def test_static_and_full_game_api_flow(self) -> None:
        # 1. Static assets
        code, html = self._req("/")
        self.assertEqual(code, 200)
        self.assertIn(b"PrimeWordle", html)

        code, js = self._req("/static/app.js")
        self.assertEqual(code, 200)
        self.assertIn(b"computeBigProduct", js)

        # 2. Create a K=10 game
        code, raw = self._req(
            "/api/games",
            method="POST",
            body={
                "num_primes": 10,
                "max_multiplier": 10,
                "code_length": 4,
                "allow_underfill": True,
                "seed": 123,
            },
        )
        self.assertEqual(code, 201)
        game = json.loads(raw)
        gid = game["game_id"]
        self.assertNotIn("target", game)

        # 3. Random guess
        code, raw = self._req(f"/api/games/{gid}/random_guess", method="POST", body={})
        self.assertEqual(code, 200)
        rand_slots = json.loads(raw)["guess_slots"]
        self.assertEqual(len(rand_slots), 4)

        # 4. Submit random guess
        code, raw = self._req(
            f"/api/games/{gid}/guess", method="POST", body={"guess_slots": rand_slots}
        )
        self.assertEqual(code, 200)
        after_guess = json.loads(raw)
        self.assertEqual(after_guess["turns"], 1)
        self.assertGreater(after_guess["token_count"], 10)

        # 5. Cheat peek
        code, raw = self._req(f"/api/games/{gid}/peek", method="POST", body={})
        self.assertEqual(code, 200)
        peeked = json.loads(raw)
        self.assertTrue(peeked["peeked"])
        self.assertIn("target", peeked)
        target_slots = peeked["target"]["slots"]

        # 6. Save notes
        code, raw = self._req(
            f"/api/games/{gid}/notes",
            method="POST",
            body={"symbol_marks": {"0": "confirmed"}, "scratchpad": "test note"},
        )
        self.assertEqual(code, 200)

        # 7. Win by submitting reversed order of target slots
        unordered_target = [list(reversed(s)) for s in target_slots]
        code, raw = self._req(
            f"/api/games/{gid}/guess", method="POST", body={"guess_slots": unordered_target}
        )
        self.assertEqual(code, 200)
        won = json.loads(raw)
        self.assertEqual(won["status"], "won")
        self.assertEqual(won["turns"], 2)

        # 8. List saved games & read latest state with tokens + deduction
        code, raw = self._req("/api/games")
        self.assertEqual(code, 200)
        games = json.loads(raw)["games"]
        self.assertEqual(len(games), 1)
        self.assertEqual(games[0]["status"], "won")

        code, raw = self._req("/api/games/latest?include_tokens=1")
        self.assertEqual(code, 200)
        latest = json.loads(raw)
        self.assertEqual(latest["game_id"], gid)
        self.assertIn("deduction", latest)
        self.assertIn("token_ids", latest)
        self.assertIn("tokens", latest)
        self.assertEqual(latest["deduction"]["solved_count"], 4)


if __name__ == "__main__":
    unittest.main()
