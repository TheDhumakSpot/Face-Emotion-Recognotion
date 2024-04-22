import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from keras.models import load_model
import streamlit as st

model = load_model('facialemotionmodel.h5')

def preprocess_image4model(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    img_resized = cv2.resize(img_gray, (48, 48))  # Resize image to 48x48
    img_normalized = img_resized / 255.0  # Normalize pixel values
    img_expanded = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
    img_expanded = np.expand_dims(img_expanded, axis=0)  # Add batch dimension
    return img_expanded

def draw_rec_and_emotion(frame,x,y,w,h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face_roi = frame[y:y + h, x:x + w]
    image4model = preprocess_image4model(face_roi)
    face_emotion = model.predict(image4model)
    # predicted_emotion = predict_emotion(face_roi)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    # print(face_emotion)
    # print(np.argmax(face_emotion))
    predicted_label = labels[np.argmax(face_emotion)]
    # print(predicted_label)
    cv2.putText(frame, f"Emotion: {predicted_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame,predicted_label


def process(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangle around the faces and predict emotions
    for (x, y, w, h) in faces:
        frame,_ = draw_rec_and_emotion(frame, x, y, w, h)
    return frame

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title('Face Emotion Recognition')

option = st.radio("Select an option:", ("Browse", "Live"))

if option == "Browse":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        # st.image(image, caption='Uploaded Image', use_column_width=True)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangle around the faces and predict emotions
        for (x, y, w, h) in faces:
            pic,label = draw_rec_and_emotion(image, x, y, w, h)

        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        st.image(pic, caption='Uploaded Image', use_column_width=True)
        st.write('Predicted Emotion:', label)

if option == "Live":
    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
