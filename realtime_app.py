import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np

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

# Config
IMAGE_SIZE = 128
import json
try:
    with open("classes.json", "r") as f:
        CLASSES = json.load(f)
except FileNotFoundError:
    CLASSES = ['-K', '-N', '-P', 'FN'] 
    print("Warning: classes.json not found, using fallback.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = SimpleCNN(num_classes=len(CLASSES))
    try:
        model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: model.pth not found.")
        return None
    model.to(DEVICE)
    model.eval()
    return model

def main():
    model = load_model()
    if model is None:
        return

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Initialize video capture
    # 0 is typically the default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    print("Starting video stream. Press 'q' to exit.")
    
    prev_frame_time = 0
    new_frame_time = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Preprocess the frame for the model
            # OpenCV captures in BGR, we need RGB for PyTorch/PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
            
            # Inference
            start_infer = time.time()
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            end_infer = time.time()
            
            prediction_class = CLASSES[predicted.item()]
            confidence_score = confidence.item() * 100

            # FPS Calculation
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            
            # Draw on frame (use the original BGR frame for OpenCV display)
            # Display Prediction
            text = f"Pred: {prediction_class} ({confidence_score:.1f}%)"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display FPS
            fps_text = f"FPS: {int(fps)}"
            cv2.putText(frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Display Inference Time
            # inf_text = f"Inf: {(end_infer - start_infer)*1000:.1f}ms"
            # cv2.putText(frame, inf_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Disease Classification Real-Time', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
