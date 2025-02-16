# Flask_CNN

Project Overview: 

This project builds and deploys a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the Fashion MNIST dataset. 

The model is then deployed as a web application using Flask.

1. Dataset – Fashion MNIST
   
  What is it?
  
  A dataset of 28x28 grayscale images of clothing items (T-shirts, trousers, pullovers, dresses, etc.).
  
  Why use it?
  
  It's a simple but effective dataset for image classification, for CNNs.

3. Model Building (CNN - Convolutional Neural Network)
   
  Framework: TensorFlow/Keras

  Layers Used:
  
  Conv2D: Extracts features using convolutional filters.
  
  MaxPooling2D: Reduces spatial dimensions to prevent overfitting.
  
  Flatten: Converts feature maps into a 1D vector.
  
  Dense (Fully Connected Layer): Makes final predictions.
  
  Softmax Activation: Converts outputs into probabilities for classification.
  
  Training: The model is trained on Fashion MNIST and saved as fashion_mnist_cnn.h5 for later use.

5. Saving and Loading the Model
   
   The trained CNN model is saved using: model.save("fashion_mnist_cnn.h5")
   
   When deploying the model in Flask, it is loaded using: model = load_model("fashion_mnist_cnn.h5")

6. Web App Deployment Using Flask
   
  What is Flask? A lightweight Python web framework used to create APIs.

  How it Works:
  
  The server listens for image upload requests. (example: Test Images: Bag, Pullover2, Dress, Shirt2...)
  
  The image is processed and passed through the CNN.
  
  The model predicts the class and returns a JSON response.

  Flask Code Structure:
  
  app.py: The main file that runs the Flask web server.
  
  static/css/style.css: Stores styles for the webpage.
  
  templates/index.html: The front-end HTML file.

7. Making Predictions
   
Flask receives an image, converts it into a format the model understands, and makes a prediction.

7. Deployment Options
   
Local Deployment: Run Flask on a local machine.

Cloud Deployment: Deploy on platforms like Heroku, AWS, or Render for public access.

Summary:

Train a CNN model on Fashion MNIST and save it as fashion_mnist_cnn.h5.

Load the saved model in Flask and create an API to predict clothing categories.

Build a simple web interface using HTML, CSS, and Flask.

Deploy and test the model by uploading images.
