import pickle
import cv2
import numpy as np

with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)

image_path = r"C:\Users\VARUN PN\Downloads\person_image.jpg"

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) > 0:

    if len(faces) > 1:
        print("More than one face found!!")
    else:
        x, y, w, h = faces[0]
        cropped_image = img[y:y+h, x:x+w]
        cropResized = cv2.resize(cropped_image, (256, 256))

        # Ensure the image has the right shape (batch_size, height, width, channels)
        cropResized = np.expand_dims(cropResized, axis=0)

        predictions = MODEL.predict(cropResized)[0][0]
        label = ""

        if predictions > 0.5:
            label = "Sad"
        else:
            label = "Happy"

        confidence = w * h / (gray.shape[0] * gray.shape[1])

        if confidence > 0.01:
            print("Emotion: ", label)
        else:
            print("No proper detection found!!")

else:
    print("No face found!!")
