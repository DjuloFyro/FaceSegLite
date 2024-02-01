import sys
from flask import Flask, request, Response

sys.path.append('../utils/')
from FPN_model_utils import predict_with_fpn_resnet34
from MaskRCNN_ResNet50_model_utils import predict_with_mask_rcnn_resnet50

app = Flask(__name__)

@app.route('/upload_mask_42m', methods=['POST'])
def upload_mask_from_maskrcnn_resnet50():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    # Read the image via file.stream
    file = request.files['file']
    image_bytes = file.read()

    # Predict the mask with the model
    img_encoded = predict_with_mask_rcnn_resnet50(image_bytes)
    
    return Response(img_encoded, mimetype='image/jpeg')


@app.route('/upload_mask_21m', methods=['POST'])
def upload_mask_from_fpnn_resnet34():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    # Read the image via file.stream
    file = request.files['file']
    image_bytes = file.read()

    # Predict the mask with the model
    img_encoded = predict_with_fpn_resnet34(image_bytes)

    return Response(img_encoded, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)