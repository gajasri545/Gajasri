"""
Image Colorization Script (Standalone)
- Run from terminal: python colorize.py --image path/to/gray_image.jpg

Make sure the following files are in the 'model/' directory:
    - colorization_deploy_v2.prototxt
    - colorization_release_v2.caffemodel
    - pts_in_hull.npy
"""

import os
import cv2 
import numpy as np 
import argparse
import sys

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Colorize a grayscale image")
parser.add_argument("-i", "--image", required=True, help="Path to the input black and white image")
parser.add_argument("-o", "--output", help="Path to save the colorized output image")
args = parser.parse_args()

input_image_path = args.image
output_image_path = args.output

# === Model Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

PROTOTXT = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
POINTS = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# === Check Files Exist ===
for file_path in [PROTOTXT, MODEL, POINTS]:
    if not os.path.exists(file_path):
        print(f"❌ Missing file: {file_path}")
        sys.exit(1)

# === Load Image ===
image = cv2.imread(input_image_path)
if image is None:
    print(f"❌ Could not read image: {input_image_path}")
    sys.exit(1)

# === Load Model ===
print("🔄 Loading colorization model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Add cluster centers as 1x1 convolution kernel
pts = pts.transpose().reshape(2, 313, 1, 1)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# === Preprocess Image ===
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# === Forward Pass ===
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L_original = cv2.split(lab)[0]
colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")

# === Show and/or Save ===
if output_image_path:
    cv2.imwrite(output_image_path, colorized)
    print(f"✅ Saved colorized image to: {output_image_path}")
else:
    cv2.imshow("Original", image)
    cv2.imshow("Colorized", colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()