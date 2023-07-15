from PIL import Image
import torch,os
from torchvision import transforms
from DogNet import DogClassifier
import matplotlib.pyplot as plt
import numpy as np
import random
import glob

transform = transforms.Compose([
    transforms.Resize((64,64)),  # Resizes the image to 64x64
    transforms.ToTensor(),  # Converts the image into a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizes the tensors (mean and std deviation for 3 color channels)
])

  


def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        image = image.to('cuda')
    return image

def get_random_image_paths(root_path, num_images):
    all_image_paths = glob.glob(root_path + "*/*.jpg")
    return random.sample(all_image_paths, num_images)

def predict_and_display_images():
    image_paths = get_random_image_paths('./dataset',2)

    model_path = 'dog_classifier.pth'
    if os.path.exists(model_path):
        print("El modelo existe.")
        model = DogClassifier()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        if torch.cuda.is_available():
            print("Modelo en CUDA")
            model = model.to('cuda')
            

    # Load, predict and plot each image
    fig, axes = plt.subplots(3, 3, figsize=(10,10))
    for i, image_path in enumerate(image_paths):
        image = load_image(image_path)
        output = model(image)
        prediction = output.round().item()
        ax = axes[i//3, i%3]
        img = Image.open(image_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Predicted: Dog' if prediction else 'Predicted: Not Dog')

    plt.tight_layout()
    plt.show()

predict_and_display_images()


