import sys
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, Response

import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import transforms
import torchvision.transforms.functional as F

sys.path.append('../utils/')
import utils
from FPN_model_utils import FaceModel
from MaskRCNN_ResNet50_model_utils import get_transform, get_model_instance_segmentation

app = Flask(__name__)

loaded_model = FaceModel("FPN", "resnet34", in_channels=3, out_classes=1)
loaded_model.load_state_dict(torch.load('../models/FPN_trained_model_v3.pth', map_location=torch.device('cpu')))
loaded_model.to("cpu")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400

    file = request.files['file']
    image_bytes = file.read()

    # Use OpenCV to read the image and get its shape
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    image = transforms.ToTensor()(image)
    eval_transform = get_transform(train=False)

    loaded_model.eval()
    with torch.no_grad():
        #x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        #x = x[:3, ...]
        #image = image[:3, ...]
        predictions = loaded_model(image)
        #pred = predictions[0]


    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    # Filter predictions with confidence score greater than 0.9
    #filter_mask = pred["scores"] > 0.9
    #pred_labels = [f"face: {score:.3f}" for score in pred["scores"][filter_mask]]
    #pred_boxes = pred["boxes"][filter_mask].long()
    #output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    #output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    pr_masks = predictions.sigmoid()
    #print(pr_masks.shape)
    masks = (pr_masks > 0.3).squeeze(1)

    output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(output_image.permute(1, 2, 0).numpy())

   # if pil_image.mode != 'RGB':
    #    pil_image = pil_image.convert('RGB')
    # Create an in-memory bytes buffer for the image encoding
    image_buffer = BytesIO()
    pil_image.save(image_buffer, format='JPEG')

    # Get the encoded image bytes
    img_encoded = image_buffer.getvalue()
    return Response(img_encoded, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)