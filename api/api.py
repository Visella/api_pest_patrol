from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image  # Importing Pillow for image processing
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load your model (assuming you've saved it as 'model.h5')
model = tf.keras.models.load_model('pest_detection_model.h5')
label_class = ["Cashew anthracnose", "Cashew gumosis", "Cashew healthy", "Cashew leaf miner", "Cashew red rust"]

# Ensure the temp directory exists
if not os.path.exists('temp'):
    os.makedirs('temp')

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

            # Load the image and convert it to grayscale
            image = Image.open(temp_file_path).convert('L')  # 'L' mode for grayscale
            image = image.resize((200, 200))  # Resize to match the model's expected input size
            print("Image loaded and converted to grayscale successfully")
            
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0  # Normalize the image

            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)[0]

            return jsonify({'predicted_class': label_class[predicted_class]})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
