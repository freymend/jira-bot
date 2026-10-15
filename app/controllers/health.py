from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import db

router = APIRouter()


@router.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health")
async def health() -> JSONResponse:
    checks = db.health_checks()
    healthy = all(checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "not_ready",
            "checks": {
                name: "ok" if passed else "unavailable"
                for name, passed in checks.items()
            },
        },
    )
