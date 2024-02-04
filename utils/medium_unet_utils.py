from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
import torch

import cv2
import numpy as np
from PIL import Image
from io import BytesIO


class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)
    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)
    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return blockOutputs


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])
    def forward(self, x, encFeatures):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x
    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures
    

class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16), nbClasses=1, retainDim=True, outSize=(256, 256)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize    

    def forward(self, x):
      # grab the features from the encoder
      encFeatures = self.encoder(x)
      # pass the encoder features through decoder making sure that
      # their dimensions are suited for concatenation
      decFeatures = self.decoder(encFeatures[::-1][0],
        encFeatures[::-1][1:])
      # pass the decoder features through the regression head to
      # obtain the segmentation mask
      map = self.head(decFeatures)
      # check to see if we are retaining the original output
      # dimensions and if so, then resize the output to match them
      if self.retainDim:
        map = F.interpolate(map, self.outSize)
      # return the segmentation map
      return map
    

def predict_with_medium_unet(buffer):
    """
    Predict the mask with the with_mask_rcnn_resnet50 model
    
    Args:
        buffer (bytes): The image bytes
        
    Returns:
        bytes: The image bytes with the mask
    """
    THRESHOLD = 0.1

    # Load the model
    loaded_model = UNet(nbClasses=1, outSize=(256, 256))
    loaded_model.load_state_dict(torch.load('../models/unet_epoch_49.pt', map_location=torch.device('cpu')))
    loaded_model.eval()

    with torch.no_grad():
        # Use OpenCV to read the image and get its shape
        image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype("float32") / 255.0
        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (256, 256))
        orig = image.copy()

        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to("cpu")
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = loaded_model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > THRESHOLD).unsqueeze(0)
 
    # Convert the NumPy array to tensor for drawing
    orig = transforms.ToTensor()(orig)
    orig = (255.0 * (orig - orig.min()) / (orig.max() - orig.min())).to(torch.uint8)

    output_image = draw_segmentation_masks(orig, predMask, alpha=0.5, colors="blue")

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(output_image.permute(1, 2, 0).numpy())
    # Create an in-memory bytes buffer for the image encoding
    image_buffer = BytesIO()
    pil_image.save(image_buffer, format='JPEG')

    # Get the encoded image bytes
    img_encoded = image_buffer.getvalue()

    return img_encoded