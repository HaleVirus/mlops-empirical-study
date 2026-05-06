from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import time

app = FastAPI(title="MLOps Model Server")

# Prometheus metrics
REQUEST_COUNT = Counter("model_requests_total", "Total prediction requests", ["model"])
LATENCY = Histogram("model_latency_seconds", "Prediction latency", ["model"])

class TextInput(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()

@app.post("/predict/mnist")
def predict_mnist(digit_class: int):
    # Stub endpoint - replace with real inference
    start = time.time()
    REQUEST_COUNT.labels(model="mnist").inc()
    result = {"predicted_class": digit_class, "confidence": 0.98}
    LATENCY.labels(model="mnist").observe(time.time() - start)
    return result

@app.post("/predict/sentiment")
def predict_sentiment(input: TextInput):
    start = time.time()
    REQUEST_COUNT.labels(model="sentiment").inc()
    result = {"text": input.text[:50], "sentiment": "positive", "confidence": 0.87}
    LATENCY.labels(model="sentiment").observe(time.time() - start)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
