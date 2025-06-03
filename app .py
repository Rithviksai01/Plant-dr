from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
model = tf.keras.models.load_model("doctorp_model.h5")

@app.route('/')
def home():
    return "🌱 Plant Doctor API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(image_data)).resize((256, 256))
    img = np.array(image).astype('float32') / 255.0
    img = img.reshape(1, 256, 256, 3)

    preds = model.predict(img)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return jsonify({
        "class_index": class_idx,
        "confidence": confidence
    })
