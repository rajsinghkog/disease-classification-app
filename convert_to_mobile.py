import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.mobile_optimizer import optimize_for_mobile
import json

# Define the model architecture (must match training)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (128 // 8) * (128 // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    # Load Classes
    try:
        with open("classes.json", "r") as f:
            classes = json.load(f)
    except FileNotFoundError:
        print("classes.json not found, using default 5")
        classes = ['-K', '-N', '-P', 'FN', 'Not_Spinach'] # Fallback
    
    num_classes = len(classes)
    print(f"Converting model with {num_classes} classes...")

    # Load Model
    device = torch.device("cpu") # Mobile is usually CPU/NPU, export on CPU
    model = SimpleCNN(num_classes=num_classes)
    
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
    except FileNotFoundError:
        print("Error: model.pth not found.")
        return

    model.eval()

    # Create dummy input for tracing
    # Shape: (Batch Size, Channels, Height, Width)
    example_input = torch.rand(1, 3, 128, 128)

    # Trace
    traced_script_module = torch.jit.trace(model, example_input)

    # Optimize for Mobile
    optimized_traced_model = optimize_for_mobile(traced_script_module)

    # Save
    optimized_traced_model._save_for_lite_interpreter("model_mobile.ptl")
    print("Success! Model saved to 'model_mobile.ptl'")

if __name__ == "__main__":
    main()
