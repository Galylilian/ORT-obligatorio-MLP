import torch
from src.core.model import get_model


def test_model_output_shape():
    """
    Verifica que el modelo devuelve una salida de tamaño correcto.
    """

    model = get_model()
    model.eval()

    # input fake (imagen 224x224 RGB)
    dummy_input = torch.randn(1, 3, 224, 224)

    output = model(dummy_input)

    # debe haber 2 clases
    assert output.shape == (1, 2)
