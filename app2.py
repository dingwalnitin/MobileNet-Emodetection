import base64
import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Load the pre-trained model
model = load_model("model.h5")

# Define labels for emotion recognition
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the Haar Cascade for face detection
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Initialize Flask application and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

def preprocess_input_frame(frame):
    """
    Preprocess the input video frame for face detection and emotion recognition.
    - Converts the frame to grayscale
    - Detects faces and crops the first detected face
    - Resizes to the required dimensions (48x48 in your case)
    - Normalizes the data
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Haar Cascade to detect faces
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        # Take the first detected face and crop it
        x, y, w, h = faces[0]
        face = gray[y:y + h, x:x + w]

        # Resize the face to 48x48 (as expected by your model)
        face = cv2.resize(face, (48, 48))

        # Normalize the image data to [0, 1]
        face = face / 255.0

        # Expand to 3 channels (to match the model input shape)
        face = np.repeat(face[:, :, np.newaxis], 3, axis=2)

        # Reshape to (1, 48, 48, 3) for model input
        face = face.reshape(1, 48, 48, 3)

        return face
    else:
        return None

@socketio.on('frame')
def handle_frame(frame_base64):
    # Decode the base64 frame
    frame_bytes = base64.b64decode(frame_base64)
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_np, flags=cv2.IMREAD_COLOR)

    # Preprocess the frame
    preprocessed_frame = preprocess_input_frame(frame)

    if preprocessed_frame is not None:
        # Perform prediction
        predictions = model.predict(preprocessed_frame)[0]
        predicted_class_index = np.argmax(predictions)
        predicted_class = labels[predicted_class_index]

        # Create a dictionary with probabilities for all emotions
        emotion_probabilities = {emotion: float(probability) for emotion, probability in zip(labels, predictions)}

        # Emit the prediction and probabilities back to the client
        emit('prediction', {
            'emotion': predicted_class,
            'probabilities': emotion_probabilities
        })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)