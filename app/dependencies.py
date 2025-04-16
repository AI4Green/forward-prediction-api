from fastapi import HTTPException, Request, status

from app.core.config import Settings
from app.services.prediction import Prediction

settings = Settings()


def prediction(request: Request) -> Prediction:
    model = request.app.state.model
    return Prediction(model=model)


def require_auth(request: Request):
    if settings.auth_enabled:
        return _get_api_key(request)
    return None


def _get_api_key(request: Request) -> str:
    if not settings.auth_enabled:
        return ""

    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured but auth is enabled",
        )

    api_key = request.headers.get(settings.api_key_header)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorised request"
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden request"
        )

    return api_key
