# Flask API Documentation

### Endpoint: /predict
### Description: This endpoint accepts an audio file, preprocesses it, and uses a pre-trained TensorFlow model to predict the label of the speaker.

### HTTP Method: POST

### Request:

#### Content-Type: multipart/form-data
#### Parameters:
file: The audio file to be processed. Supported formats include .wav, .mp3, .flac, .ogg, and .au.
### Response:

#### Content-Type: application/json
#### Body:
On success: { "predicted_label": "label_name" }
On error: { "error": "error_message" }