from __future__ import annotations

import base64
import hashlib
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
from dotenv import load_dotenv

AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.events.readonly"


class _CallbackServer(HTTPServer):
    expected_state: str
    authorization_code: str | None = None
    authorization_error: str | None = None


class _CallbackHandler(BaseHTTPRequestHandler):
    server: _CallbackServer

    def do_GET(self) -> None:
        query = parse_qs(urlsplit(self.path).query)
        state = query.get("state", [""])[0]
        code = query.get("code", [""])[0]
        error = query.get("error", [""])[0]
        if state != self.server.expected_state:
            self.server.authorization_error = "OAuth state verification failed"
            self.send_response(400)
            message = b"Authorization failed. You may close this tab."
        elif error:
            self.server.authorization_error = error
            self.send_response(400)
            message = b"Authorization was not completed. You may close this tab."
        elif code:
            self.server.authorization_code = code
            self.send_response(200)
            message = b"Calendar access approved. You may close this tab."
        else:
            self.server.authorization_error = "Google returned no authorization code"
            self.send_response(400)
            message = b"Authorization failed. You may close this tab."
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(message)))
        self.end_headers()
        self.wfile.write(message)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _build_authorization_url(
    client_id: str, redirect_uri: str, state: str, code_challenge: str
) -> str:
    return f"{AUTHORIZATION_ENDPOINT}?{
        urlencode(
            {
                'client_id': client_id,
                'redirect_uri': redirect_uri,
                'response_type': 'code',
                'scope': CALENDAR_SCOPE,
                'access_type': 'offline',
                'prompt': 'consent',
                'state': state,
                'code_challenge': code_challenge,
                'code_challenge_method': 'S256',
            }
        )
    }"


def _exchange_code(
    code: str,
    code_verifier: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> str:
    response = httpx.post(
        TOKEN_ENDPOINT,
        data={
            "code": code,
            "code_verifier": code_verifier,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=20,
    )
    response.raise_for_status()
    token_response = response.json()
    refresh_token = token_response.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise RuntimeError(
            "Google did not return a refresh token; revoke this app's access and run again"
        )
    return refresh_token


def main() -> None:
    load_dotenv()
    client_id = os.getenv("GOOGLE_CALENDAR_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise SystemExit(
            "Set GOOGLE_CALENDAR_CLIENT_ID and GOOGLE_CALENDAR_CLIENT_SECRET in .env first"
        )

    state = secrets.token_urlsafe(32)
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("ascii")).digest())
        .decode("ascii")
        .rstrip("=")
    )

    server = _CallbackServer(("127.0.0.1", 0), _CallbackHandler)
    server.expected_state = state
    server.timeout = 180
    redirect_uri = f"http://127.0.0.1:{server.server_address[1]}/"
    authorization_url = _build_authorization_url(
        client_id, redirect_uri, state, code_challenge
    )
    callback_thread = threading.Thread(target=server.handle_request, daemon=True)
    callback_thread.start()
    print("Opening Google consent in your browser. If it does not open, visit:\n")
    print(authorization_url)
    webbrowser.open(authorization_url)
    callback_thread.join(timeout=185)
    server.server_close()

    if callback_thread.is_alive():
        raise SystemExit("Timed out waiting for Google OAuth approval")
    if server.authorization_error:
        raise SystemExit(f"Google OAuth failed: {server.authorization_error}")
    if not server.authorization_code:
        raise SystemExit("Google OAuth completed without an authorization code")

    refresh_token = _exchange_code(
        server.authorization_code,
        code_verifier,
        client_id,
        client_secret,
        redirect_uri,
    )
    print("\nAdd this line to .env and keep it private:")
    print(f"GOOGLE_CALENDAR_REFRESH_TOKEN={refresh_token}")


if __name__ == "__main__":
    main()
