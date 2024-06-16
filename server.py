from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import math
import io
from PIL import Image

app = Flask(__name__)

# Initialize the model
pretrained_model = models.densenet169(pretrained=False)

# Modify the classifier to match the default DenseNet169's output features (1664)
pretrained_model.classifier = nn.Linear(1664, 5)

# Load the model's trained weights
checkpoint = torch.load('./models/KneeNet.0', map_location=torch.device('cpu'))

# Remove the 'classifier' weights from the checkpoint to avoid the mismatch
if 'classifier.weight' in checkpoint:
    del checkpoint['classifier.weight']
if 'classifier.bias' in checkpoint:
    del checkpoint['classifier.bias']

# Load the rest of the model's weights
pretrained_model.load_state_dict(checkpoint, strict=False)

# Set the model to evaluation mode
pretrained_model.eval()

def default_preprocessing(image, min_hw_ratio=1, output_width=299, output_height=299):
    r, c = image.shape
    if c > r:
        c_to_keep = r * min_hw_ratio
        c_to_delete = c - c_to_keep
        remove_from_left = int(math.ceil(c_to_delete / 2))
        remove_from_right = int(math.floor(c_to_delete / 2))
        image_trimmed = image[:, remove_from_left:(c - remove_from_right)]
    else:
        r_to_keep = c * min_hw_ratio
        r_to_delete = r - r_to_keep
        remove_from_top = int(math.ceil(r_to_delete / 2))
        remove_from_bottom = int(math.floor(r_to_delete / 2))
        image_trimmed = image[remove_from_top:(r - remove_from_bottom), :]

    image_resampled = cv2.resize(image_trimmed, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
    image_clean = image_resampled - np.mean(image_resampled)
    image_clean = image_clean / np.std(image_clean)
    image_clean = (image_clean - np.min(image_clean)) / (np.max(image_clean) - np.min(image_clean))

    image_clean_stacked = np.stack((image_clean, image_clean, image_clean), axis=0)  # Shape (3, 299, 299)

    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    imagenet_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    image_clean_stacked = (image_clean_stacked - imagenet_mean) / imagenet_std

    return image_clean_stacked

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        image = Image.open(file.stream).convert('L')  # Convert to grayscale
        image = np.array(image).astype('float')

        processed_image = default_preprocessing(image)
        processed_image_tensor = torch.unsqueeze(torch.from_numpy(processed_image), 0).float()

        with torch.no_grad():
            outputs = pretrained_model(processed_image_tensor)

        prediction_idx = torch.argmax(outputs).item()
        prediction_name = f"Grade {prediction_idx}"  # Format as "Grade X"

        # Convert prediction to image and add text
        output_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(output_image, f"Prediction: {prediction_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert to PIL image for sending
        _, img_encoded = cv2.imencode('.png', output_image)
        img_io = io.BytesIO(img_encoded.tobytes())

        return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


