import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Path to the HDF5 model file
model_path = './mosquito_model.h5'

# Load the trained model
model = load_model(model_path)

# Dictionary mapping class indices to species names
species_names = {0: 'Aedes Aegypti', 1: 'Anopheles Stephensi'}

def predict_species(image_path, threshold=0.6):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    max_probability = np.max(prediction)

    # Check if the confidence is below the threshold
    if max_probability < threshold:
        return "Unknown"
    else:
        # Get the species name
        species_name = species_names.get(predicted_class, 'Not the vector Aedes/Anopheles')
        return species_name

# Example usage
image_file = '.\Untitled.jpg'
predicted_species = predict_species(image_file)
print(f"The predicted species is: {predicted_species}")


