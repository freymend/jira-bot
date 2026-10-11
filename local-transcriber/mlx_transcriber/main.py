from __future__ import annotations

import hmac
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Any, cast

import mlx_whisper
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
app = FastAPI(title="Local MLX Whisper Transcriber")
_transcription_lock = threading.Lock()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _format_utterances(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    utterances: list[dict[str, Any]] = []
    for segment in segments:
        words = segment.get("words") or []
        formatted_words = [
            {
                "word": str(word.get("word", "")),
                "start": float(word.get("start", 0.0)),
                "end": float(word.get("end", 0.0)),
            }
            for word in words
            if isinstance(word, dict) and str(word.get("word", "")).strip()
        ]
        if not formatted_words and str(segment.get("text", "")).strip():
            formatted_words.append(
                {
                    "word": str(segment["text"]),
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", 0.0)),
                }
            )
        if formatted_words:
            utterances.append({"words": formatted_words})
    return utterances


@app.post("/transcribe", response_model=None)
def transcribe(
    audio: Annotated[UploadFile, File()],
    language: Annotated[str | None, Form()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, Any] | JSONResponse:
    try:
        expected_token = os.getenv("MLX_TRANSCRIBER_API_KEY", "")
        if not expected_token:
            raise HTTPException(
                status_code=503, detail="Transcriber token is not configured"
            )
        expected_authorization = f"Bearer {expected_token}"
        if not authorization or not hmac.compare_digest(
            authorization, expected_authorization
        ):
            raise HTTPException(status_code=401, detail="Invalid transcriber token")

        model = os.getenv("MLX_WHISPER_MODEL", "mlx-community/whisper-turbo")
        with tempfile.TemporaryDirectory(prefix="mlx-whisper-") as temp_dir:
            audio_path = Path(temp_dir) / "audio.mp3"
            with audio_path.open("wb") as output_file:
                shutil.copyfileobj(audio.file, output_file)
            if audio_path.stat().st_size == 0:
                raise HTTPException(status_code=422, detail="Audio upload is empty")

            with _transcription_lock:
                result = mlx_whisper.transcribe(
                    str(audio_path),
                    path_or_hf_repo=model,
                    word_timestamps=True,
                    verbose=False,
                    language=language or None,
                )
    except HTTPException:
        raise
    except Exception:
        logger.exception("MLX Whisper transcription failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_code": "TRANSCRIPTION_FAILED"},
        )
    finally:
        audio.file.close()

    # mlx-whisper does not annotate its return value; transcribe() returns segment dictionaries.
    segments = cast(list[dict[str, Any]], result.get("segments", []))
    return {
        "status": "done",
        "result": {
            "transcription": {
                "full_transcript": str(result.get("text", "")).strip(),
                "utterances": _format_utterances(segments),
            }
        },
    }
