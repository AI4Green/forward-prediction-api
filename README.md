# Forward Prediction API

A FastAPI-based API that uses pre-trained model from the [ASKCOS project](https://gitlab.com/mlpds_mit/askcosv2/forward_predictor/augmented_transformer) for forward prediction.

Docker image can be found at: https://hub.docker.com/r/ai4greeneln/forward_prediction

### Local Development

1. **Install dependencies**
   ```bash
   # Install uv package manager
   pip install uv
   
   # Install project dependencies using uv
   uv sync
   ```
   Check [this](https://docs.astral.sh/uv/getting-started/installation/) for installing uv.

1. **Download trained models**
   ```bash
   chmod +x download_trained_models.sh
   ./download_trained_models.sh
   ```

1. **Run the application**
   ```bash
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t forward-prediction-api:latest .
   ```

1. **Run the container**
   ```bash
   docker run -d --name forward-prediction-api \
     -p 8000:8000 \
     -e AUTH_ENABLED=false \
     forward-prediction-api:latest
   ```

## API Endpoint

### POST `/api/predict`

**Headers:**
- `X-API-Key`: API key for auth (optional)

**Request body:**
```json
{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}'
```
