"""
Post-training optimization utilities (e.g., quantization)
"""
import torch

def quantize_dynamic(model, dtype=torch.qint8):
    """
    Apply dynamic quantization to a model.
    Args:
        model: torch.nn.Module
        dtype: quantization dtype (default: torch.qint8)
    Returns:
        quantized_model: quantized torch.nn.Module
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=dtype
    )
    return quantized_model

# Example for static quantization (requires calibration):
def prepare_static_quantization(model):
    """
    Prepare model for static quantization (inserts observers).
    """
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    return model

def convert_static_quantization(model):
    """
    Convert a calibrated model to a quantized version.
    """
    torch.quantization.convert(model, inplace=True)
    return model
