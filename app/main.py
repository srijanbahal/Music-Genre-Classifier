#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from src.data_loader import preprocess_audio
from src.model_predict import load_model, predict_genre
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Upload folder setup
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model and labels
MODEL_PATH = 'models/best_model.h5'
GENRE_LABELS = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

# Load model once on startup
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audioFile' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['audioFile']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.wav'):
            return jsonify({'error': 'Only .wav files allowed'}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        features = preprocess_audio(file_path)
        features = np.expand_dims(features, axis=0)

        predicted_genre = predict_genre(model, features, GENRE_LABELS)

        return jsonify({
            'genre': predicted_genre,
            'audioUrl': f'/static/uploads/{file.filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
