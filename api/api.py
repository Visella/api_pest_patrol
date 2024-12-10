import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import os
import matplotlib.pyplot as plt

# Pest Information (Class Labels, Symptoms, and Management)
pest_info = {
    "Cashew anthracnose": {
        "symptoms": ["Dark spots and necrosis on leaves and fruits"],
        "management": ["Apply appropriate fungicide", "Ensure proper pruning and sanitation"]
    },
    "Cashew gumosis": {
        "symptoms": ["Gum oozing from bark", "Swelling or cracking of bark"],
        "management": ["Improve drainage", "Apply fungicide or bactericide"]
    },
    "Cashew healthy": {
        "symptoms": ["No visible infections", "Healthy leaves and fruits"],
        "management": ["Continue regular maintenance", "Monitor for early signs of pests"]
    },
    "Cashew leaf miner": {
        "symptoms": ["White trails on leaves", "Reduced leaf area"],
        "management": ["Remove infected leaves", "Apply insecticides"]
    },
    "Cashew red rust": {
        "symptoms": ["Reddish-brown pustules on leaves"],
        "management": ["Spray copper-based fungicides", "Remove infected plant parts"]
    }
}

# Define LABEL_CLASSES mapping based on the order of the labels the model is trained on
LABEL_CLASSES = list(pest_info.keys())

# Load your model
model = tf.keras.models.load_model('pest_detection_model.h5')
IMAGE_ROWS = 200
IMAGE_COLS = 200

# Ensure the temp directory exists
if not os.path.exists('temp'):
    os.makedirs('temp')

# Flask application setup
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if file:
            print(f"File received: {file.filename}")

            # Save the file temporarily
            temp_file_path = os.path.join('temp', file.filename)
            file.save(temp_file_path)

            # Load and preprocess the image
            image = Image.open(temp_file_path).convert('L')  # 'L' mode for grayscale
            image = image.resize((IMAGE_ROWS, IMAGE_COLS))  # Resize to match the model's expected input size
            print("Image loaded and converted to grayscale successfully")

            image_array = img_to_array(image) / 255.0  # Normalize the image
            image_array = np.reshape(image_array, (1, IMAGE_ROWS, IMAGE_COLS, 1))  # Reshape for the model

            # Predict the pest class
            predictions = model.predict(image_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)

            if predicted_class < len(LABEL_CLASSES):
                pest_name = LABEL_CLASSES[predicted_class]
                pest_data = pest_info[pest_name]

                return jsonify({
                    'predicted_class': pest_name,
                    'confidence': f"{confidence * 100:.2f}%",
                    'symptoms': pest_data['symptoms'],
                    'management': pest_data['management']
                })
            else:
                return jsonify({'error': "Predicted class is out of range."}), 400

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
