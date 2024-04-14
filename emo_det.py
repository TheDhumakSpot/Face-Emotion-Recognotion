import cv2
import numpy as np
from keras.models import load_model

model = load_model('emotiondetector6.h5')


video_capture = cv2.VideoCapture(0)

def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    img_resized = cv2.resize(img_gray, (48, 48))  # Resize image to 48x48
    img_normalized = img_resized / 255.0  # Normalize pixel values
    img_expanded = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
    img_expanded = np.expand_dims(img_expanded, axis=0)  # Add batch dimension
    return img_expanded

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = frame[y:y + h, x:x + w]
        image4model = preprocess_image(face_roi)
        face_emotion = model.predict(image4model)
        # predicted_emotion = predict_emotion(face_roi)
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        # print(face_emotion)
        # print(np.argmax(face_emotion))
        predicted_label = labels[np.argmax(face_emotion)]
        # print(predicted_label)
        cv2.putText(frame, f"Emotion: {predicted_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(36, 255, 12), 2)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
