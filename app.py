from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model('./speaker_detection_gru.h5')
# model = tf.keras.models.load_model('speaker_detection_gru.h5')

# Load the labels
labels = {label : clss for label, clss in enumerate(open("labels.txt").read().split("\n"))}


# Function to preprocess audio
def preprocess_audio(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Resample to 6000 Hz
    y = librosa.resample(y, orig_sr=sr, target_sr=8000)
    
    # Ensure the audio is at least 1 second long, pad if necessary
    if len(y) < 8000:
        y = np.pad(y, (0, 8000 - len(y)), mode='constant')
    else:
        y = y[:8000]
    
    # Reshape to the input shape required by the model
    y = y.reshape(1, -1, 1)
    
    return y

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an audio file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file temporarily
    file_path = 'temp.wav'
    file.save(file_path)
    
    # Preprocess the audio
    processed_audio = preprocess_audio(file_path)
    
    # Make prediction
    prediction = model.predict(processed_audio)
    
    # Assume the model returns a label index, convert it to a string label if necessary
    predicted_label = np.argmax(prediction, axis=1)[0]
    
    return jsonify({'predicted_label': labels[int(predicted_label)]})

if __name__ == '__main__':
    app.run(debug=True)
