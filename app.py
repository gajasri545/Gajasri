import os
import uuid
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
PROTOTXT = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
POINTS = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# Load Model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Colorization function
def colorize_image(image_path):
    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    L_resized = cv2.resize(L, (224, 224))
    L_resized -= 50

    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    output_path = image_path.replace(".", "_colorized.")
    cv2.imwrite(output_path, colorized)
    return os.path.basename(output_path)

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            ext = os.path.splitext(file.filename)[1]
            filename = uuid.uuid4().hex + ext
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result_filename = colorize_image(filepath)
            return render_template("index.html", input_image=filename, output_image=result_filename)
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
