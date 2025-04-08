import tensorflow as tf

import numpy as np


def load_model(model_path):
    """
    Load the pre-trained model from the specified path.
    
    Parameters:
    - model_path: Path to the saved model.
    
    Returns:
    - model: Loaded TensorFlow model.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
def predict_genre(model,features, genre_labels):
    """
    Predict the genre of the audio features using the loaded model.
    
    Parameters:
    - model: Loaded TensorFlow model.
    - features: Preprocessed audio features.
    - genre_labels: List of genre labels for mapping predictions.
    
    Returns:
    - predicted_genre: Predicted genre label.
    """
    try:
        prediction = model.predict(features)
        print(prediction)
        predicted_class_index = np.argmax(prediction[0])
        predicted_genre = genre_labels[predicted_class_index]
        
        return predicted_genre
    except Exception as e:
        print(f"Error predicting genre: {e}")
        return None

