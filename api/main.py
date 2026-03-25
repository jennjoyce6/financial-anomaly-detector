from fastapi import FastAPI
from api.routes.detect import router

app = FastAPI(
    title       = "Fraud Detection Microservice",
    description = "Real-time financial transaction anomaly detection API",
    version     = "1.0.0",
)

app.include_router(router)

@app.get("/health")
def health_check():
    """Simple health check — confirms the service is running."""
    return {"status": "healthy", "service": "fraud-detection-api"}