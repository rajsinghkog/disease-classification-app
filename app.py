import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import pandas as pd

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
try:
    with open("classes.json", "r") as f:
        CLASSES = json.load(f)
except FileNotFoundError:
    CLASSES = ['-K', '-N', '-P', 'Full Nutrition(FN)', 'Not_Found'] # Fallback
    print("Warning: classes.json not found, using fallback.")

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(CLASSES))
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
    except FileNotFoundError:
        st.error("Model file 'model.pth' not found. Please run 'train.py' first.")
        return None
    model.to(device)
    model.eval()
    return model

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def main():
    st.set_page_config(page_title="Deficiency Classification", page_icon="🌿")
    st.title("🌿 Deficiency Classification")
    st.write("Universal Inference App: Works on Desktop & Mobile")

    model = load_model()
    if model is None:
        return

    # Input method selection
    input_method = st.radio("Select Input Method:", ("Upload Image", "Camera"))
    
    image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    else:
        camera_file = st.camera_input("Take a picture")
        if camera_file is not None:
            image = Image.open(camera_file).convert('RGB')

    if image is not None:
        # Display Image
        st.image(image, caption='Input Image', width=300)
        
        # Inference
        try:
            input_tensor = process_image(image)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                prediction_class = CLASSES[predicted.item()]
                confidence_score = confidence.item() * 100
            
            # Results
            st.divider()
            st.subheader("Prediction Results")
            
            # Highlight result
            if prediction_class == "Not_Found":
                st.warning(f"**Prediction:** {prediction_class}")
               // st.info("This does not appear to be a  leaf.")
            else:
                st.success(f"**Prediction:** {prediction_class}")
            
            st.metric("Confidence", f"{confidence_score:.2f}%")
            
            # Bar Chart
            st.write("### Class Probabilities")
            probs_np = probabilities.cpu().numpy()[0]
            df = pd.DataFrame({
                "Class": CLASSES,
                "Probability": probs_np
            })
            st.bar_chart(df.set_index("Class"))
                
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()

