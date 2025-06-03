from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64

from disease_detection_ai_ import predict_image  # 👈 Import your function

app = Flask(doctorp_model.h5)

@app.route('/')
def home():
    return "🌿 Plant Disease API using .py model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).resize((256, 256))
        img = np.array(image).astype('float32') / 255.0
        img = img.reshape(1, 256, 256, 3)

        class_idx, confidence = predict_image(img)

        return jsonify({
            "class_index": int(class_idx),
            "confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


