import sys
from flask import Flask, request, Response, send_file

sys.path.append('../utils/')
from FPN_model_utils import predict_with_small_unet
from MaskRCNN_ResNet50_model_utils import predict_with_mask_rcnn_resnet50
from unet_utils import predict_with_unet

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


@app.route('/upload_small_unet', methods=['POST'])
def upload_mask_from_fpnn_resnet34():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    # Read the image via file.stream
    file = request.files['file']
    image_bytes = file.read()

    # Predict the mask with the model
    img_encoded = predict_with_small_unet(image_bytes)

    return Response(img_encoded, mimetype='image/jpeg')

@app.route('/upload_mask_unet', methods=['POST'])
def upload_mask_from_unet():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    # Read the image via file.stream
    file = request.files['file']
    image_bytes = file.read()

    # Predict the mask with the model
    img_encoded = predict_with_unet(image_bytes)

    return Response(img_encoded, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)