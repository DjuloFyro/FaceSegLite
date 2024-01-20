from flask import Flask, request, Response
import cv2
import numpy as np

app = Flask(__name__)

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    file = request.files['file']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_with_boxes = detect_faces(img)

    _, img_encoded = cv2.imencode('.jpg', img_with_boxes)
    return Response(img_encoded.tostring(), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)