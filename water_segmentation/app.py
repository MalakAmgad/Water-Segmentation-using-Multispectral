import os
import torch
import numpy as np
import tifffile as tiff
import cv2
from flask import Flask, request, render_template, send_from_directory
from torchvision import transforms
import segmentation_models_pytorch as smp
from werkzeug.utils import secure_filename

# Flask app initialization
app = Flask(__name__)

# Model setup
MODEL_PATH = "model/deeplabv3_water_segmentation.pth"
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.DeepLabV3(encoder_name="resnet50", encoder_weights=None, in_channels=12, classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Image preprocessing function
def preprocess_image(image_path):
    try:
        img = tiff.imread(image_path)  # Read the TIFF image (H, W, 12)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32)
        img[:, :, [2, 3, 4]] /= 255.0  # Normalize RGB bands
        img[:, :, [i for i in range(12) if i not in [2, 3, 4]]] /= 10000.0  # Normalize other bands
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor
        return img.to(device)
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")

# Apply smoothing and refined thresholding
def refine_mask(mask):
    mask = cv2.GaussianBlur(mask, (3, 3), 0)  # Soft blur to reduce noise
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)  # Lower threshold
    
    # Morphological transformations
    kernel = np.ones((2, 2), np.uint8)  # Small kernel for fine adjustments
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove small noise
    
    return mask

# Predict segmentation mask
def predict_mask(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = refine_mask(mask)  # Apply refinement
        return mask

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            img_tensor = preprocess_image(filepath)
            mask = predict_mask(img_tensor)
            mask_filename = "mask_" + filename.replace(".tif", ".png")
            mask_path = os.path.join(app.config["UPLOAD_FOLDER"], mask_filename)
            cv2.imwrite(mask_path, mask)
            
            # Save the original image as PNG for display
            original_filename = "original_" + filename.replace(".tif", ".png")
            original_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
            img = tiff.imread(filepath)
            img_rgb = img[:, :, [2, 3, 4]]  # Extract RGB bands
            img_rgb = (img_rgb / np.max(img_rgb) * 255).astype(np.uint8)  # Normalize and convert
            cv2.imwrite(original_path, img_rgb)
            
            return render_template("index.html", uploaded_file=original_filename, mask_file=mask_filename)
        except Exception as e:
            return f"Error processing image: {e}"

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
