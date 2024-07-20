from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('traffic_sign_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32, 32))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        prediction = model.predict(img)
        class_id = np.argmax(prediction)
        return f"Predicted Class: {class_id}"

if __name__ == '__main__':
    app.run(debug=True)
