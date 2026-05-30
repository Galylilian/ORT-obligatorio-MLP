from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_predict():
    response = client.get("/docs")
    assert response.status_code == 200