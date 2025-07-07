from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import json

app = Flask(__name__)

# Load model
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 38)
model.load_state_dict(torch.load("model.pth"))
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

@app.route("/predict", methods=["POST"])

def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "Invalid image format"}), 400

    try:
        image = Image.open(file).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        prediction = friendly_names[idx_to_class[predicted.item()]]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)