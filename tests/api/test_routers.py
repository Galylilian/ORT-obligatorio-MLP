
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_image_classification_endpoint(client):
    payload = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
    ]

    response = client.post("/classification/images", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "images" in data
    assert len(data["images"]) == 1

    assert "label" in data["images"][0]
    assert "score" in data["images"][0]


def test_image_classification_multiple_inputs(client):
    payload = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
        "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg"
    ]

    response = client.post("/classification/images", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert len(data["images"]) == 2


def test_image_classification_empty_input(client):
    payload = []

    response = client.post("/classification/images", json=payload)

    assert response.status_code == 200

    assert response.json()["images"] == []


def test_image_classification_invalid_input(client):
    response = client.post("/classification/images", json="not a list")

    assert response.status_code == 422


def test_image_classification_missing_field(client):
    response = client.post("/classification/images", json={})

    assert response.status_code == 422