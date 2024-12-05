import cv2
import numpy as np
import tensorflow as tf

# Load the model
model_path = "my_model.h5"
model = tf.keras.models.load_model(model_path)

# Path to the image
image_path = "test.jpeg"

# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected.")
        return None

    # Process the first detected face
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]  # Crop the face
    
    # Resize to (48, 48)
    face_resized = cv2.resize(face, (48, 48))
    
    # Normalize pixel values to range [0, 1]
    face_normalized = face_resized / 255.0

    return original, gray, face, face_resized

# Preprocess the face image
preprocessed = preprocess_face(image_path)

if preprocessed:
    original, gray, face, face_resized = preprocessed

    # Reshape the face to match model input (1, 48, 48, 1)
    preprocessed_face = face_resized.reshape(1, 48, 48, 1)  # Add batch dimension and channel dimension

    # Predict using the model
    predictions = model.predict(preprocessed_face)

    # Print the predictions
    print("Model Predictions:", predictions)

    # Class names (ensure these match the output of your model)
    class_names = ['Anger', 'Contentment', 'Joy', 'Neutral', 'Sadness']

    # Find the index of the maximum probability
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]

    print("Predicted Class:", predicted_class)

else:
    print("Failed to preprocess the image.")
