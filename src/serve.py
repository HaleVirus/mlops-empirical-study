from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse, HTMLResponse
import time
import random

app = FastAPI(title="MLOps Model Server")

# Prometheus metrics
REQUEST_COUNT = Counter("model_requests_total", "Total prediction requests", ["model"])
LATENCY = Histogram("model_latency_seconds", "Prediction latency", ["model"])

class TextInput(BaseModel):
    text: str

# --- ROOT ROUTE (this was missing) ---
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>MLOps Model Server is Running ✅</h2>
    <ul>
        <li><a href="/docs">API Docs (Swagger)</a></li>
        <li><a href="/health">Health Check</a></li>
        <li><a href="/metrics">Prometheus Metrics</a></li>
        <li><a href="/predict/sentiment/test">Test Sentiment</a></li>
        <li><a href="/predict/mnist/test">Test MNIST</a></li>
    </ul>
    """

@app.get("/health")
def health():
    return {"status": "ok", "message": "MLOps server is healthy"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/predict/mnist/test")
def predict_mnist_test():
    start = time.time()
    REQUEST_COUNT.labels(model="mnist").inc()
    result = {
        "predicted_class": random.randint(0, 9),
        "confidence": round(random.uniform(0.85, 0.99), 4)
    }
    LATENCY.labels(model="mnist").observe(time.time() - start)
    return result

@app.get("/predict/sentiment/test")
def predict_sentiment_test():
    start = time.time()
    REQUEST_COUNT.labels(model="sentiment").inc()
    result = {
        "sentiment": random.choice(["positive", "negative"]),
        "confidence": round(random.uniform(0.80, 0.99), 4)
    }
    LATENCY.labels(model="sentiment").observe(time.time() - start)
    return result

@app.post("/predict/sentiment")
def predict_sentiment(input: TextInput):
    start = time.time()
    REQUEST_COUNT.labels(model="sentiment").inc()
    result = {
        "text": input.text[:50],
        "sentiment": "positive",
        "confidence": round(random.uniform(0.80, 0.99), 4)
    }
    LATENCY.labels(model="sentiment").observe(time.time() - start)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
