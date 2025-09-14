from flask import Flask, request, render_template
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
import os
import gdown
from io import BytesIO

# --------------------- Flask App ---------------------
app = Flask(__name__)

# --------------------- Paths ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "cattle_buffalo_resnet50.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.txt")

# --------------------- Download Model from Google Drive if Missing ---------------------
if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading from Google Drive...")
    model_url = "https://drive.google.com/uc?id=14bPK0BKC1MAndCVhs8gKrPhWpVHRrMqz"
    gdown.download(model_url, MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")

# --------------------- Download Class Indices if Missing ---------------------
if not os.path.exists(CLASS_INDICES_PATH):
    print("Class indices not found locally. Downloading from Google Drive...")
    class_url = "https://drive.google.com/uc?id=1aGqz-jk_0p98ayx6_aapy_GXWzmdjuA6"
    gdown.download(class_url, CLASS_INDICES_PATH, quiet=False)
    print("✅ Class indices downloaded successfully!")

# --------------------- Load Model ---------------------
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --------------------- Load Class Labels ---------------------
class_indices = {}
with open(CLASS_INDICES_PATH) as f:
    for line in f:
        idx, cls = line.strip().split(":")
        class_indices[int(idx)] = cls

class_names = [class_indices[i] for i in sorted(class_indices.keys())]

IMG_SIZE = (224, 224)  # Change if your model expects a different input size

# --------------------- Routes ---------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            # Read image directly in memory
            img_bytes = file.read()
            img = image.load_img(BytesIO(img_bytes), target_size=IMG_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds, axis=1)[0]
            confidence = round(float(np.max(preds)) * 100, 2)
            prediction = class_names[pred_idx]

    return render_template("index.html", prediction=prediction, confidence=confidence)

# --------------------- Run App ---------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
