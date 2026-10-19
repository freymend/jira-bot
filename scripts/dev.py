from __future__ import annotations

import argparse
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO
from urllib.error import URLError
from urllib.request import urlopen

import certifi
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
ATTENDEE_ROOT = ROOT.parent / "attendee"
STATE_ROOT = ROOT / ".dev-run"
TLS_ROOT = ROOT / "certs"
TLS_PROXY_HOSTNAME = "api-tls"
STARTUP_TIMEOUT_SECONDS = 600


def _setting(name: str, default: str) -> str:
    value = os.environ.get(name) or dotenv_values(ROOT / ".env").get(name)
    return str(value) if value else default


def _is_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as response:
            return response.status == 200
    except OSError, URLError:
        return False


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_matches_service(pid: int, name: str) -> bool:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    command = result.stdout.lower()
    if name == "gemma":
        return "vllm" in command and "serve" in command
    if name == "dashboard":
        return "uvicorn" in command and "app.dashboard.server:app" in command
    return "uvicorn" in command and "mlx_transcriber.main" in command


def _wait_until_ready(name: str, url: str, pid: int, log: TextIO) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    print(f"Waiting for {name} at {url}; startup output is in {log.name}.", flush=True)
    while time.monotonic() < deadline:
        if _is_ready(url):
            print(f"{name} is ready.", flush=True)
            return
        if not _process_exists(pid):
            raise RuntimeError(f"{name} exited during startup; see {log.name}")
        time.sleep(5)
    raise TimeoutError(f"{name} did not become ready; see {log.name}")


def _start_native_service(
    name: str,
    command: list[str],
    cwd: Path,
    health_url: str,
) -> None:
    STATE_ROOT.mkdir(exist_ok=True)
    pid_file = STATE_ROOT / f"{name}.pid"
    log_file = STATE_ROOT / f"{name}.log"

    if _is_ready(health_url):
        print(f"{name} is already running; leaving it unmanaged.")
        return

    if pid_file.exists():
        try:
            existing_pid = int(pid_file.read_text().strip())
        except ValueError:
            existing_pid = -1
        if (
            existing_pid > 0
            and _process_exists(existing_pid)
            and _pid_matches_service(existing_pid, name)
        ):
            with log_file.open("a", encoding="utf-8") as log:
                _wait_until_ready(name, health_url, existing_pid, log)
            return
        pid_file.unlink(missing_ok=True)

    with log_file.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    with log_file.open("a", encoding="utf-8") as log:
        _wait_until_ready(name, health_url, process.pid, log)


def _stop_native_service(name: str) -> None:
    pid_file = STATE_ROOT / f"{name}.pid"
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        pid = -1
    if pid > 0 and _process_exists(pid) and _pid_matches_service(pid, name):
        print(f"Stopping managed {name} process ({pid}).")
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pid_file.unlink(missing_ok=True)
            return
        except PermissionError as exc:
            print(
                f"Could not stop {name} process ({pid}): {exc}. "
                "Leaving it running and retaining its PID file.",
                file=sys.stderr,
            )
            return
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _process_exists(pid):
            time.sleep(0.2)
        if _process_exists(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                print(
                    f"Could not force-stop {name} process ({pid}): {exc}. "
                    "Leaving it running and retaining its PID file.",
                    file=sys.stderr,
                )
                return
    pid_file.unlink(missing_ok=True)


def _stop_native_services() -> None:
    for name in ("dashboard", "gemma", "mlx-whisper"):
        _stop_native_service(name)


def _dashboard_command() -> list[str]:
    port = _setting("LOCAL_DASHBOARD_PORT", "5006")
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "app.dashboard.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        port,
    ]


def _whisper_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "mlx_transcriber.main:app",
        "--env-file",
        str(ROOT / ".env"),
        "--host",
        "0.0.0.0",
        "--port",
        "8101",
    ]


def _require_setup() -> None:
    required_paths = (
        ROOT / ".env",
        ATTENDEE_ROOT / "dev.docker-compose.yaml",
        ATTENDEE_ROOT / ".env",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"Complete the local setup first; missing:\n{paths}")

    for executable in ("docker", "ffmpeg"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required command is not on PATH: {executable}")
    if shutil.which("vllm") is None and not _is_ready("http://127.0.0.1:8100/health"):
        raise RuntimeError(
            "Activate the vLLM-Metal environment before starting Gemma, or start "
            "Gemma separately on port 8100."
        )
    key = _setting("MLX_TRANSCRIBER_API_KEY", "")
    if not key or key == "replace-with-a-long-random-token":
        raise RuntimeError("Set MLX_TRANSCRIBER_API_KEY in the project .env file.")


def _openssl(openssl: str, *args: str | Path) -> None:
    subprocess.run(
        [openssl, *map(str, args)], check=True, capture_output=True, text=True
    )


def _ensure_local_tls() -> None:
    """Create the local CA and proxy certificate for Attendee's wss:// video stream.

    Attendee only streams video to wss:// URLs. The api-tls proxy serves the API's
    video websocket with a certificate from this CA, and Attendee's worker trusts the
    CA through a bundle that also keeps the public CAs. The CA key stays on the host.
    """
    ca_cert = TLS_ROOT / "local-ca.pem"
    ca_key = TLS_ROOT / "local-ca-key.pem"
    cert = TLS_ROOT / f"{TLS_PROXY_HOSTNAME}.pem"
    key = TLS_ROOT / f"{TLS_PROXY_HOSTNAME}-key.pem"
    if not all(path.exists() for path in (ca_cert, cert, key)):
        openssl = shutil.which("openssl")
        if openssl is None:
            raise RuntimeError("Required command is not on PATH: openssl")
        TLS_ROOT.mkdir(mode=0o700, exist_ok=True)
        ec_key = ("-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes")
        _openssl(
            openssl, "req", "-x509", *ec_key, "-keyout", ca_key, "-out", ca_cert,
            "-days", "825", "-subj", "/CN=jira-bot local development CA",
            "-addext", "basicConstraints=critical,CA:TRUE",
            "-addext", "keyUsage=critical,keyCertSign,cRLSign",
        )  # fmt: skip
        csr = TLS_ROOT / f"{TLS_PROXY_HOSTNAME}.csr"
        extensions = TLS_ROOT / f"{TLS_PROXY_HOSTNAME}.ext"
        extensions.write_text(
            f"subjectAltName=DNS:{TLS_PROXY_HOSTNAME}\n"
            "basicConstraints=critical,CA:FALSE\n"
            "keyUsage=critical,digitalSignature\n"
            "extendedKeyUsage=serverAuth\n"
        )
        try:
            _openssl(
                openssl, "req", *ec_key, "-keyout", key, "-out", csr,
                "-subj", f"/CN={TLS_PROXY_HOSTNAME}",
            )  # fmt: skip
            _openssl(
                openssl, "x509", "-req", "-in", csr, "-CA", ca_cert, "-CAkey", ca_key,
                "-set_serial", f"0x{secrets.token_hex(16)}", "-days", "825",
                "-extfile", extensions, "-out", cert,
            )  # fmt: skip
        finally:
            csr.unlink(missing_ok=True)
            extensions.unlink(missing_ok=True)
    bundle = TLS_ROOT / "attendee-ca-bundle.pem"
    bundle.write_text(
        Path(certifi.where()).read_text().rstrip("\n") + "\n" + ca_cert.read_text()
    )


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        check=check,
        text=True,
    )


def up() -> None:
    _require_setup()
    whisper_command = _whisper_command()
    model = _setting("LLM_MODEL", "mlx-community/gemma-4-e4b-it-4bit")
    context_tokens = int(_setting("LLM_CONTEXT_TOKENS", "16384"))
    if context_tokens < 8192:
        raise RuntimeError("LLM_CONTEXT_TOKENS must be at least 8192")
    vllm = shutil.which("vllm")
    try:
        if vllm is not None:
            gemma_command = [
                vllm,
                "serve",
                model,
                "--language-model-only",
                "--host",
                "0.0.0.0",
                "--port",
                "8100",
                "--max-model-len",
                str(context_tokens),
            ]
            _start_native_service(
                "gemma", gemma_command, ROOT, "http://127.0.0.1:8100/health"
            )
        _start_native_service(
            "mlx-whisper",
            whisper_command,
            ROOT,
            "http://127.0.0.1:8101/health",
        )
        _ensure_local_tls()
        _compose("up", "--build", "-d")
        dashboard_port = _setting("LOCAL_DASHBOARD_PORT", "5006")
        _start_native_service(
            "dashboard",
            _dashboard_command(),
            ROOT,
            f"http://127.0.0.1:{dashboard_port}/health/live",
        )
    except OSError, RuntimeError, subprocess.CalledProcessError, TimeoutError:
        _stop_native_services()
        raise
    dashboard_port = _setting("LOCAL_DASHBOARD_PORT", "5006")
    print("API, Attendee, MLX Whisper, and Gemma are running.")
    print(f"Local dashboard: http://127.0.0.1:{dashboard_port}")
    print(
        "Native logs: .dev-run/mlx-whisper.log, .dev-run/gemma.log, and .dev-run/dashboard.log"
    )


def restart_whisper() -> None:
    _require_setup()
    health_url = "http://127.0.0.1:8101/health"
    _stop_native_service("mlx-whisper")
    if _is_ready(health_url):
        raise RuntimeError(
            "MLX Whisper is running without a managed process ID; stop it manually "
            "before restarting it."
        )
    _start_native_service("mlx-whisper", _whisper_command(), ROOT, health_url)
    print("MLX Whisper restarted.")


def down() -> None:
    try:
        _compose("down", check=False)
    finally:
        _stop_native_services()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local meeting assistant stack"
    )
    parser.add_argument("command", choices=("up", "down", "restart-whisper"))
    args = parser.parse_args()
    try:
        if args.command == "up":
            up()
        elif args.command == "restart-whisper":
            restart_whisper()
        else:
            down()
    except (OSError, RuntimeError, subprocess.CalledProcessError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        _stop_native_services()
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
