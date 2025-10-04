import pytest
from api.practice import app

@pytest.fixture
def client():
    return app.test_client()

def test_home(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<h1>This is Home Page</h1>" in resp.text
    assert "API is runnign sucessfully!" in resp.text
    assert "Use POST /predict" in resp.text
    assert "features" in resp.text
    
def test_predict(client):
    test_data = {
                    "features": [7,5,3,1]
                }
    resp = client.post("/predict", json= test_data)
    assert resp.status_code == 200
    assert resp.json == {'prediction': 0}