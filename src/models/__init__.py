"""Model factory for RADAR2026."""

from .model_bilstm_affinity import get_model as get_bilstm_affinity_model
from .model_shallow_cnn import get_model as get_shallow_cnn_model
from .model_unet import get_model as get_unet_model


def get_model(model_type: str = "unet", in_channels: int = 2):
    if model_type == "unet":
        return get_unet_model(in_channels=in_channels)
    if model_type == "shallow_cnn":
        return get_shallow_cnn_model(in_channels=in_channels)
    if model_type == "bilstm_affinity":
        return get_bilstm_affinity_model()
    raise ValueError(f"Unknown model_type: {model_type}. Use 'unet', 'shallow_cnn', or 'bilstm_affinity'.")


__all__ = ["get_model"]
