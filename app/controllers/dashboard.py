from fastapi import APIRouter, Depends

from app.controllers.dependencies import require_app_auth
from app.schemas import DashboardOverview
from app.services.dashboard import dashboard_overview as build_dashboard_overview

router = APIRouter()


@router.get("/dashboard/api/overview", dependencies=[Depends(require_app_auth)])
async def dashboard_overview(limit: int = 25) -> DashboardOverview:

    return await build_dashboard_overview(limit)
