import sys
sys.path.insert(0, "src")

def test_imports():
    import torch
    import mlflow
    assert True

def test_mnist_model_shape():
    import torch
    from mnist_train import MNISTNet
    model = MNISTNet()
    x = torch.randn(1, 1, 28, 28)
    out = model(x)
    assert out.shape == (1, 10), f"Expected (1,10), got {out.shape}"

def test_health_endpoint():
    from serve import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
