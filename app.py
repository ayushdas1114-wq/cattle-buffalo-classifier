from flask import Flask, request, render_template
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os
import uuid
import requests

# --------------------- Flask App ---------------------
app = Flask(__name__)

# --------------------- Paths ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import gdown

MODEL_PATH = os.path.join(MODEL_DIR, "cattle_buffalo_resnet50.h5")

if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading from Google Drive...")
    url = "https://drive.google.com/uc?id=14bPK0BKC1MAndCVhs8gKrPhWpVHRrMqz"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded successfully!")


CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.txt")

# --------------------- Download Model from Google Drive if Missing ---------------------
# Replace this with your file's Google Drive direct download URL
MODEL_GDRIVE_URL = "https://drive.google.com/uc?id=14bPK0BKC1MAndCVhs8gKrPhWpVHRrMqz&export=download"

if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading from Google Drive...")
    response = requests.get(MODEL_GDRIVE_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Model downloaded successfully!")

# --------------------- Load Model ---------------------
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --------------------- Load Class Labels ---------------------
if not os.path.exists(CLASS_INDICES_PATH):
    raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}.")

class_indices = {}
with open(CLASS_INDICES_PATH) as f:
    for line in f:
        idx, cls = line.strip().split(":")
        class_indices[int(idx)] = cls
class_names = [class_indices[i] for i in sorted(class_indices.keys())]

IMG_SIZE = (224, 224)

# --------------------- Prediction Function ---------------------
def predict_breed(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return class_names[pred_idx], round(confidence * 100, 2)

# --------------------- Routes ---------------------
@app.route("/", methods=["GET", "POST"])
def index():
    breed, confidence = None, None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            os.makedirs("uploads", exist_ok=True)
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join("uploads", unique_filename)
            file.save(filepath)
            breed, confidence = predict_breed(filepath)
            os.remove(filepath)  # optional: delete file after prediction
        else:
            return "❌ No file uploaded or empty filename"

    return render_template("index.html", breed=breed, confidence=confidence)

# --------------------- Run App ---------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
