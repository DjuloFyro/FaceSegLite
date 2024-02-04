import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms import v2 as T


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        
        self.block1 = conv_block(3, 16)
        self.block2 = conv_block(16, 32)
        self.block3 = conv_block(32, 64)
        self.block4 = conv_block(64, 128)
        self.block5 = conv_block(128, 256)
        
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.block6 = conv_block(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.block7 = conv_block(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.block8 = conv_block(64, 32)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.block9 = conv_block(32, 16)
        
        self.output = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x):
        block1 = self.block1(x)
        pool1 = F.max_pool2d(block1, 2)
        block2 = self.block2(pool1)
        pool2 = F.max_pool2d(block2, 2)
        block3 = self.block3(pool2)
        pool3 = F.max_pool2d(block3, 2)
        block4 = self.block4(pool3)
        pool4 = F.max_pool2d(block4, 2)
        block5 = self.block5(pool4)
        
        up6 = self.upconv4(block5)
        concat6 = torch.cat([up6, block4], dim=1)
        block6 = self.block6(concat6)
        up7 = self.upconv3(block6)
        concat7 = torch.cat([up7, block3], dim=1)
        block7 = self.block7(concat7)
        up8 = self.upconv2(block7)
        concat8 = torch.cat([up8, block2], dim=1)
        block8 = self.block8(concat8)
        up9 = self.upconv1(block8)
        concat9 = torch.cat([up9, block1], dim=1)
        block9 = self.block9(concat9)
        
        out = self.output(block9)
        
        return out


class FaceModel(pl.LightningModule):

    def __init__(self, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = UNet(num_classes=out_classes)

        # preprocessing parameteres for image
        self.register_buffer("std", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


def eval_transform(image):
  target_size = (256, 256)
  transforms = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Resize(target_size, antialias=True)])
  return transforms(image)

def predict_with_fpn_resnet34(buffer):
    """
    Predict the mask with the fpn_resnet34 model
    
    Args:
        buffer (bytes): The image bytes
    
    Returns:
        bytes: The image bytes with the mask
    """

    # Load the model
    loaded_model = FaceModel(in_channels=3, out_classes=1)
    loaded_model.load_state_dict(torch.load('../models/unet_model_small_v4.pth', map_location=torch.device('cpu')))
    loaded_model.to("cpu")

    # Use OpenCV to read the image and get its shape
    image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Sauvegardez les dimensions originales
    original_dimensions = image.shape[:2]

    image = eval_transform(image)

    loaded_model.eval()
    with torch.no_grad():
        predictions = loaded_model(image)

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    pr_masks = predictions.sigmoid()
    masks = (pr_masks > 0.5).squeeze(1)

    output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")

    output_image = cv2.resize(output_image.permute(1, 2, 0).numpy(), (original_dimensions[1], original_dimensions[0]))

    pil_image = Image.fromarray(output_image)

    image_buffer = BytesIO()
    pil_image.save(image_buffer, format='JPEG')

    img_encoded = image_buffer.getvalue()

    return img_encoded