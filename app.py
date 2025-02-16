from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("fashion_mnist_cnn.h5")

# Class labels
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file).convert("L").resize((28, 28))  # Convert to grayscale
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Reshape for CNN input

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({"prediction": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
