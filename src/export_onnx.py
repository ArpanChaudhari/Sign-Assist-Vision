import os
import torch
import torch.nn as nn
import numpy as np

# Prevent OpenMP library conflict on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# Neural Network Model
class SignModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Simple fully connected neural network
        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# Load the class names
classes = np.load("models/classes.npy", allow_pickle=True)

# Create the model
model = SignModel(len(classes))

# Load the trained model weights
model.load_state_dict(torch.load("models/model.pth"))

# Set model to evaluation mode
model.eval()


# Create sample input
# 63 values = 21 hand landmarks × 3 coordinates (x, y, z)
dummy_input = torch.randn(1, 63)


# Export PyTorch model to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    "models/model.onnx",
    export_params=True,
    opset_version=14,
    input_names=["input"],
    output_names=["output"]
)

print("Model exported successfully!")