import torch
from transformers import AutoConfig, AutoModelForImageClassification
import os

def export_model_to_onnx(model_path: str, onnx_path: str):
    """
    Export the DocumentFigureClassifier model to ONNX format.
    
    Parameters
    ----------
    model_path : str
        Path to the directory containing the model files
    onnx_path : str
        Path where the ONNX model will be saved
    """
    # Load the model
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.eval()
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export the model
    torch.onnx.export(
        model,                     # PyTorch model
        dummy_input,              # Input tensor
        onnx_path,               # Output file path
        export_params=True,      # Store the trained weights
        opset_version=12,        # ONNX version
        do_constant_folding=True,# Optimize constant-folding
        input_names=['input'],   # Input tensor name
        output_names=['output'], # Output tensor name
        dynamic_axes={
            'input': {0: 'batch_size'},  # Variable batch size
            'output': {0: 'batch_size'}  # Variable batch size
        }
    )
    
    print(f"Model exported successfully to: {onnx_path}")

if __name__ == "__main__":
    # Example usage
    model_path = "path/to/your/model"
    onnx_path = os.path.join(model_path, "model.onnx")
    export_model_to_onnx(model_path, onnx_path)
