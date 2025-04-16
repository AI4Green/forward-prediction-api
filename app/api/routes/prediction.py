from fastapi import APIRouter, Depends

from app.core.config import Settings
from app.dependencies import prediction, require_auth
from app.schemas.prediction import PredictionRequest, PredictionResult
from app.services.prediction import Prediction

settings = Settings()

router = APIRouter()


@router.post("/predict")
def predict(
    request: PredictionRequest,
    service: Prediction = Depends(prediction),
    _: str = Depends(require_auth),
) -> PredictionResult:
    """
    Predictions for a given SMILES string.
    """
    result = service.predict([request.smiles])
    return result
