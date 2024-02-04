import sys
from flask import Flask, request, Response

sys.path.append('../utils/')
from MaskRCNN_ResNet50_model_utils import predict_with_mask_rcnn_resnet50
from medium_unet_utils import predict_with_medium_unet
from small_unet_only import predict_with_small_unet_only
from small_unet_pretained import predict_with_small_unet_pretained
from small_unet_with_yolo import predict_with_small_unet_and_yolo

app = Flask(__name__)


@app.route('/upload_mask_rcnn', methods=['POST'])
def upload_mask_from_maskrcnn_resnet50():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    # Read the image via file.stream
    file = request.files['file']
    image_bytes = file.read()

    # Predict the mask with the model
    img_encoded = predict_with_mask_rcnn_resnet50(image_bytes)
    
    return Response(img_encoded, mimetype='image/jpeg')


@app.route('/upload_small_unet_only', methods=['POST'])
def upload_mask_from_small_unet_only():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    # Read the image via file.stream
    file = request.files['file']
    image_bytes = file.read()

    # Predict the mask with the model
    img_encoded = predict_with_small_unet_only(image_bytes)

    return Response(img_encoded, mimetype='image/jpeg')

@app.route('/upload_small_unet_pretained', methods=['POST'])
def upload_mask_from_small_unet_pretained():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    # Read the image via file.stream
    file = request.files['file']
    image_bytes = file.read()

    # Predict the mask with the model
    img_encoded = predict_with_small_unet_pretained(image_bytes)

    return Response(img_encoded, mimetype='image/jpeg')

@app.route('/upload_small_unet_and_yolo', methods=['POST'])
def upload_mask_from_small_unet_and_yolo():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    # Read the image via file.stream
    file = request.files['file']
    image_bytes = file.read()

    # Predict the mask with the model
    img_encoded = predict_with_small_unet_and_yolo(image_bytes)

    return Response(img_encoded, mimetype='image/jpeg')

@app.route('/upload_medium_unet', methods=['POST'])
def upload_mask_from_medium_unet():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    # Read the image via file.stream
    file = request.files['file']
    image_bytes = file.read()

    # Predict the mask with the model
    img_encoded = predict_with_medium_unet(image_bytes)

    return Response(img_encoded, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)