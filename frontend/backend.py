from flask import Flask, request, Response
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
import utils
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks



app = Flask(__name__)


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")


    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

loaded_model = get_model_instance_segmentation(2)
loaded_model.load_state_dict(torch.load('../model/fineTuned_torchMaskRCNN', map_location=torch.device('cpu')))
loaded_model.to("cpu")
loaded_model.eval() 



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    file = request.files['file']
    nparr = np.frombuffer(file.read(), np.uint8)

    eval_transform = get_transform(train=False)

    loaded_model.eval()
    with torch.no_grad():
        x = eval_transform(nparr)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to("cpu")
        predictions = loaded_model([x, ])
        pred = predictions[0]

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    # Filter predictions with confidence score greater than 0.9
    filter_mask = pred["scores"] > 0.9
    pred_labels = [f"face: {score:.3f}" for score in pred["scores"][filter_mask]]
    pred_boxes = pred["boxes"][filter_mask].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"][filter_mask] > 0.6).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    _, img_encoded = cv2.imencode('.jpg', output_image)
    return Response(img_encoded.tostring(), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)