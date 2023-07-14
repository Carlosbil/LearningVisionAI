from flask import Flask, request, jsonify
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
from DogNet import DogClassifier
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas de la aplicación


# Load your trained model
model = DogClassifier()
model.load_state_dict(torch.load('dog_classifier.pth'))
model.eval()

if torch.cuda.is_available():
    model = model.to('cuda')

transform = Compose([
    Resize((64,64)),  # Resizes the image to 64x64
    ToTensor(),  # Converts the image into a tensor
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizes the tensors (mean and std deviation for 3 color channels)
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    image = Image.open(file)
    image_tensor = transform(image).unsqueeze(0)
    
    if torch.cuda.is_available():
        image_tensor = image_tensor.to('cuda')
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.round()

    result = {
        'prediction': 'dog' if prediction.item() == 0 else 'not a dog'
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
