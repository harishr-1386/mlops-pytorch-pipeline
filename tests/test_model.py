import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import get_model


class TestGetModel:
    def test_resnet18_output_shape(self):
        model = get_model(architecture="resnet18", num_classes=10)
        model.eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_resnet18_num_classes_5(self):
        model = get_model(architecture="resnet18", num_classes=5)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    def test_unsupported_architecture_raises(self):
        with pytest.raises(ValueError, match="Unsupported architecture"):
            get_model(architecture="vgg16", num_classes=10)

    def test_model_is_nn_module(self):
        from torch import nn
        model = get_model()
        assert isinstance(model, nn.Module)


class TestHealthEndpointNoModel:
    def test_health_returns_503_without_model(self):
        import serve
        original = serve._model
        serve._model = None
        try:
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                serve.health()
            assert exc_info.value.status_code == 503
        finally:
            serve._model = original
