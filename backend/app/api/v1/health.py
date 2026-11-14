from fastapi import APIRouter, Response, status

from app.core.database import database_is_available

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(response: Response) -> dict[str, str | bool]:
    database_available = database_is_available()
    if not database_available:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if database_available else "degraded",
        "database": database_available,
    }