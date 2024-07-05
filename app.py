import os
import tempfile

from flask import Flask, request, jsonify

import librosa
import soundfile as sf

import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


encoder = LabelEncoder()
encoder.fit(['Bahaouddyn', 'Belvanie', 'Brel', 'Clement', 'Danielle', 'Emeric', 'Harlette', 'Ines', 'Nahomie', 'Ngoran', 'Sasha'])

# Load the TensorFlow model
# model = tf.keras.models.load_model('modelcnn.keras')
model = tf.keras.models.load_model('speaker_detection_gru.h5')

# Load the labels
labels = {label : clss for label, clss in enumerate(open("labels.txt").read().split("\n"))}



# Convertion des fichiers .m4a, .acc, .ogg en .wav
def convert_to_wav(filename):
    # Utiliser un fichier temporaire pour enregistrer le contenu de l'objet UploadedFile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_file:
        temp_file.write(open(filename, 'rb').read())
        temp_filename = temp_file.name

    # Charger le fichier temporaire avec librosa
    y, s = librosa.load(temp_filename, sr=16000)  # Charge le fichier et resample à 16000 Hz, ce qui nous permet de normaliser nos donnees
    yt, index = librosa.effects.trim(y, top_db=30, frame_length=512, hop_length=64) # top_db est le seuil en dB sous lequel le signal est considéré comme du silence
    
    # Créer un nouveau nom de fichier pour le fichier .wav
    new_filename = os.path.splitext(temp_filename)[0] + '.wav'  # Change l'extension du fichier
    
    # Écrire le fichier au format .wav
    sf.write(new_filename, yt, s)  # Écrit le fichier au format .wav
    
    # Supprimer le fichier temporaire .m4a
    os.remove(temp_filename)

    return new_filename

# Fonction pour l'extraction des caracteristiques
def extract_mfcc_features(filename, n_mfcc=13):
    try:
        audio_path = convert_to_wav(filename)
    except:
        pass

    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return np.array(mfcc_mean)

def prediction(audio_file):
    mfcc = extract_mfcc_features(audio_file)
    data = mfcc.reshape(1, 13)
    data_reshape = np.reshape(data, (data.shape[0], data.shape[1], 1))
    pred = model.predict(data_reshape, verbose=0)
    pred = np.argmax(pred, axis=1)
    pred_1d = pred.flatten()

    pred_decoded = encoder.inverse_transform(pred_1d)

    return pred_decoded


@app.route('/predict', methods=['POST'])
def predict():
    # Check if an audio file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file temporarily
    file.save(file.filename)
    
    # Make prediction
    predicted_label = prediction(file.filename)

    return jsonify({'predicted_label': str(predicted_label[0])})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
