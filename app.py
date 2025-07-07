import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import json
import requests
import io

# Load model (for local inference)
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 38)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))  # Use 'cuda' if GPU available
model.eval()

# Load class mapping
with open("classes.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
friendly_names = {c: c.replace("___", " ").replace("_", " ") for c in idx_to_class.values()}

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("Plant Disease Classifier")
st.write("Upload a leaf image to detect plant diseases.")

# Option to choose prediction method
prediction_method = st.radio("Choose prediction method:", ("Local Inference", "Flask API"))

# Input for Flask API URL
api_url = st.text_input("Flask API URL (if using API)", "http://localhost:5000/predict")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if prediction_method == "Local Inference":
        # Local inference
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            confidence = probs[0, predicted.item()].item()
        prediction = friendly_names[idx_to_class[predicted.item()]]
        st.write(f"**Prediction (Local)**: {prediction} (Confidence: {confidence:.2%})")
    else:
        # Flask API inference
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            response = requests.post(api_url, files={"file": ("image.png", img_byte_arr, "image/png")})
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "Error")
                confidence = result.get("confidence", 0)
                st.write(f"**Prediction (Flask API)**: {prediction} (Confidence: {confidence:.2%})")
            else:
                st.error(f"API request failed: {response.status_code}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")