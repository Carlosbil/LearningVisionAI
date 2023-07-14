# LearningVisionAI
This repository is made only for learn about Vision Computing and neuronal networks

Redes Neuronales.docx explain the process and the code for everyone that want to join to this awensome world. 
Actually it is a draft, in spanish lenguage, but i will update it soon and let it able in other lenguages too

# Dog Detection with PyTorch and Flask

This project uses a convolutional neural network (CNN) implemented in PyTorch to classify images as containing a dog or not. The trained model is then deployed using a simple Flask web server that accepts image uploads and returns the classification results.

## Project Structure

- `DogNet.py` - Defines the CNN architecture used for the dog classifier.
- `train.py` - Trains the dog classifier using PyTorch and saves the trained model.
- `app.py` - Defines the Flask web server that uses the trained model to classify uploaded images.
- `dataset/dogs/images` - Contains images of dogs used for training.(https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
- `dataset/no_dogs/images` - Contains images of non-dogs used for training. (cifar100 dataset, it will download automatically)
