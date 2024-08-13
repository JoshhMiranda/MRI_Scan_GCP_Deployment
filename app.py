from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image
# import mlflow
# import mlflow.keras

app = Flask(__name__, template_folder='templates')
# app = application

alex_loaded = load_model("artifacts/mri_classifier_local_v3.h5")

# Load model from MLflow
# model_uri = "models:/mri_scan_classifier/latest"  # Adjust based on your model name and version
# alex_loaded = mlflow.keras.load_model(model_uri)

# Define your class labels based on your model training
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
def process_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = img_array.reshape((1, 256, 256, 3))
    return img_array

def predict_image(image_array):
    # Perform prediction
    prediction = alex_loaded.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    return predicted_class


@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    
    # Process the uploaded image
    img_array = process_image(file.stream)
    
    # Perform prediction
    predicted_class = predict_image(img_array)
    
    # Prepare the result to display in the HTML page
    prediction_result = f'Predicted class: {predicted_class}'

    return render_template('index.html', prediction=prediction_result)


if __name__=="__main__":
    app.run(host = "0.0.0.0", port=8080)
