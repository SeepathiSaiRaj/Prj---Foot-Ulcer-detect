# app.py
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import os

app = Flask(__name__)

# Load the updated trained model
model = load_model('dfu_transfer_mobilenet_model.h5')

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_ulcer(img):
    img_array = preprocess_image(img)
    prob = model.predict(img_array, verbose=0)[0][0]
    
    # Adjust threshold here
    threshold = 0.50  # You can adjust this threshold value based on your observations
    
    return {
        "prediction": "Ulcer Detected" if prob >= threshold else "No Ulcer Detected",
        "confidence": float(prob if prob >= threshold else 1 - prob),
        "class": 1 if prob >= threshold else 0,
        "threshold": threshold
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    result = img_data = diet_plan = medications = confidence = metrics_data = conf_matrix_img = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(file.stream).convert("RGB")
            img_io = io.BytesIO()
            img.save(img_io, 'PNG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode()
            img_data = f"data:image/png;base64,{img_base64}"

            prediction = predict_ulcer(img)
            result = prediction["prediction"]
            confidence = f"{prediction['confidence']*100:.1f}%"
            y_true = [prediction['class']]
            y_pred = [1 if prediction['confidence'] >= prediction['threshold'] else 0]

            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Non-Ulcer', 'Ulcer'],
                        yticklabels=['Non-Ulcer', 'Ulcer'])
            plt.title('Prediction Result')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            conf_matrix_img = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            metrics_data = {
                "threshold": prediction['threshold'],
                "confidence": prediction['confidence'],
                "prediction_class": "Ulcer" if prediction['class'] == 1 else "Non-Ulcer"
            }

            # Only provide diet and medication suggestions if Ulcer is detected
            if prediction["class"] == 1:  # If ulcer is detected
                diet_plan = [
                    "High-protein foods: eggs, fish, lean meats",
                    "Vitamin C rich foods: citrus fruits, bell peppers",
                    "Zinc-rich foods: nuts, seeds, whole grains",
                    "Avoid alcohol and tobacco",
                    "Limit processed sugars"
                ]
                medications = [
                    "Antibacterial ointments (e.g., Silver sulfadiazine)",
                    "Oral antibiotics if prescribed",
                    "Regular wound cleaning and dressing",
                    "Pain management as needed",
                    "Consider topical growth factors for chronic ulcers"
                ]
            else:  # No ulcer detected
                diet_plan = []
                medications = []

    return render_template('index.html', 
        result=result, 
        img_data=img_data,
        diet_plan=diet_plan,  # Only show diet plan if ulcer is detected
        medications=medications,  # Only show medications if ulcer is detected
        confidence=confidence,
        metrics_data=json.dumps(metrics_data) if metrics_data else None,
        conf_matrix_img=conf_matrix_img)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    img = Image.open(request.files['image'].stream).convert("RGB")
    prediction = predict_ulcer(img)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
