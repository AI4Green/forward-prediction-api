import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.prediction import router as prediction_router
from app.core.config import Settings
from app.services.opennmt_model_runner import OpenNMTModelRunner

settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = OpenNMTModelRunner(
        n_best=30, beam_size=30, use_cuda=settings.use_cuda_if_available
    )
    model.load(model_dir=settings.model_dir, model_filename=settings.model_filename)

    app.state.model = model

    yield


# Initialize API Server
app = FastAPI(
    title="ML Model",
    description="Forward Prediction Model",
    version=settings.version,
    lifespan=lifespan,
)

app.include_router(prediction_router, prefix="/api")
