import sys
import tensorflow as tf
from flask import Flask, request, Response, send_file

sys.path.append('../utils/')
from FPN_model_utils import predict_with_fpn_resnet34
from MaskRCNN_ResNet50_model_utils import predict_with_mask_rcnn_resnet50
from unet_utils import predict_with_unet

app = Flask(__name__)

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Charger le modèle
# model = tf.keras.models.load_model('../models/best_model_bce_v2.h5', custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient})

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