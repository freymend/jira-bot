from __future__ import annotations

import hmac
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import mlx_whisper
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
app = FastAPI(title="Local MLX Whisper Transcriber")
_transcription_lock = threading.Lock()


def _scratch_dir() -> Path:
    # Avoid the inherited TMPDIR: it may belong to a short-lived sandbox
    # session and disappear while this service keeps running.
    configured = os.getenv("MLX_TRANSCRIBER_TMP_DIR", "")
    path = (
        Path(configured) if configured else Path.home() / ".cache" / "mlx-transcriber"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


class WhisperWord(TypedDict, total=False):
    word: object
    start: object
    end: object


class WhisperSegment(TypedDict):
    words: list[WhisperWord]
    text: object
    start: object
    end: object


class TranscriptionWord(TypedDict):
    word: str
    start: float
    end: float


class TranscriptionUtterance(TypedDict):
    words: list[TranscriptionWord]


class TranscriptionContent(TypedDict):
    full_transcript: str
    utterances: list[TranscriptionUtterance]


class TranscriptionResult(TypedDict):
    transcription: TranscriptionContent


class TranscriptionResponse(TypedDict):
    status: Literal["done"]
    result: TranscriptionResult


@app.get("/health", response_model=None)
def health() -> dict[str, str] | JSONResponse:
    try:
        with tempfile.NamedTemporaryFile(dir=_scratch_dir()):
            pass
    except OSError:
        logger.exception("Transcriber scratch directory is not writable")
        return JSONResponse(
            status_code=503, content={"status": "error", "error_code": "NO_SCRATCH_DIR"}
        )
    return {"status": "ok"}


def _normalize_whisper_segments(value: object) -> list[WhisperSegment]:
    if not isinstance(value, list):
        return []
    segments: list[WhisperSegment] = []
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeError("MLX Whisper returned an invalid segment")
        raw_words = item.get("words")
        words: list[WhisperWord] = []
        if isinstance(raw_words, list):
            for word in raw_words:
                if isinstance(word, dict):
                    words.append(
                        {
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0),
                        }
                    )
        segments.append(
            {
                "words": words,
                "text": item.get("text", ""),
                "start": item.get("start", 0.0),
                "end": item.get("end", 0.0),
            }
        )
    return segments


def _as_float(value: object) -> float:
    if not isinstance(value, str | int | float):
        raise TypeError("MLX Whisper returned a nonnumeric timestamp")
    return float(value)


def _format_utterances(
    segments: list[WhisperSegment],
) -> list[TranscriptionUtterance]:
    utterances: list[TranscriptionUtterance] = []
    for segment in segments:
        formatted_words: list[TranscriptionWord] = []
        for word in segment["words"]:
            word_text = str(word.get("word", ""))
            if not word_text.strip():
                continue
            formatted_words.append(
                {
                    "word": word_text,
                    "start": _as_float(word.get("start", 0.0)),
                    "end": _as_float(word.get("end", 0.0)),
                }
            )
        if not formatted_words and str(segment.get("text", "")).strip():
            formatted_words.append(
                {
                    "word": str(segment["text"]),
                    "start": _as_float(segment.get("start", 0.0)),
                    "end": _as_float(segment.get("end", 0.0)),
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
) -> TranscriptionResponse | JSONResponse:
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
        with tempfile.TemporaryDirectory(
            prefix="mlx-whisper-", dir=_scratch_dir()
        ) as temp_dir:
            audio_path = Path(temp_dir) / "audio.mp3"
            with audio_path.open("wb") as output_file:
                shutil.copyfileobj(audio.file, output_file)
            if audio_path.stat().st_size == 0:
                raise HTTPException(status_code=422, detail="Audio upload is empty")

            with _transcription_lock:
                result: object = mlx_whisper.transcribe(
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

    if not isinstance(result, dict):
        raise RuntimeError("MLX Whisper returned an invalid transcription result")
    raw_segments = result.get("segments", [])
    if not isinstance(raw_segments, list):
        raise RuntimeError("MLX Whisper returned an invalid segment list")
    segments = _normalize_whisper_segments(raw_segments)
    return {
        "status": "done",
        "result": {
            "transcription": {
                "full_transcript": str(result.get("text", "")).strip(),
                "utterances": _format_utterances(segments),
            }
        },
    }
