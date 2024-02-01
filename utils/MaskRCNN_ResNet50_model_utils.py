import cv2
import numpy as np
from PIL import Image
from io import BytesIO

import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import transforms
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

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

def predict_with_mask_rcnn_resnet50(buffer):
    """
    Predict the mask with the with_mask_rcnn_resnet50 model
    
    Args:
        buffer (bytes): The image bytes
        
    Returns:
        bytes: The image bytes with the mask
    """
    # Load the model
    loaded_model = get_model_instance_segmentation(2)
    loaded_model.load_state_dict(torch.load('../models/trained_model.pth', map_location=torch.device('cpu')))
    loaded_model.to("cpu")

    # Use OpenCV to read the image and get its shape
    image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    eval_transform = get_transform(train=False)

    loaded_model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        x = x[:3, ...]
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

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(output_image.permute(1, 2, 0).numpy())

    # Create an in-memory bytes buffer for the image encoding
    image_buffer = BytesIO()
    pil_image.save(image_buffer, format='JPEG')

    # Get the encoded image bytes
    img_encoded = image_buffer.getvalue()

    return img_encoded