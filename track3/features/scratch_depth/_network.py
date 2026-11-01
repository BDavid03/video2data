"""Original randomly initialized CNN, adapting the local coursework to dense regression.

Imported only inside authorized operations. No pretrained weights, model-zoo
constructors, teacher imports, external source trees or downloads are used here.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(),
        )


class CourseworkDepthCNN(nn.Module):
    """RGB [0,1], NCHW -> positive optical-axis depth, N1HW.

    Three max-pooling stages followed by bilinear upsampling and skip connections
    retain the pixel grid. The coursework classifier's Flatten/Dense head is
    replaced by a one-channel regression head. Every convolution starts random.
    """

    def __init__(self, base_channels=24, dropout=0.1):
        super().__init__()
        c = base_channels
        self.encoder1 = ConvBlock(3, c)
        self.encoder2 = ConvBlock(c, c * 2)
        self.encoder3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvBlock(c * 4, c * 8), nn.Dropout2d(dropout))
        self.decoder3 = ConvBlock(c * 12, c * 4)
        self.decoder2 = ConvBlock(c * 6, c * 2)
        self.decoder1 = ConvBlock(c * 3, c)
        self.depth_head = nn.Conv2d(c, 1, 1)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 8:
            raise ValueError("Expected RGB NCHW with both spatial dimensions at least eight")
        first = self.encoder1(image)
        second = self.encoder2(self.pool(first))
        third = self.encoder3(self.pool(second))
        features = self.bottleneck(self.pool(third))
        for skip, decoder in ((third, self.decoder3), (second, self.decoder2), (first, self.decoder1)):
            features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            features = decoder(torch.cat((features, skip), dim=1))
        return F.softplus(self.depth_head(features)) + 1e-3


def masked_log_depth_loss(prediction, target, valid):
    """MAE in log metres on valid labels; retains the teacher's absolute scale.

    Invalid target NaNs are replaced BEFORE log/arithmetic so they cannot poison
    backward. There is no scale alignment, evaluation metric or confidence loss.
    """
    mask = valid.bool() & torch.isfinite(target) & (target > 0)
    if not bool(mask.any()):
        raise ValueError("Training batch contains no usable depth labels")
    safe_target = torch.where(mask, target, torch.ones_like(target))
    error = (prediction.log() - safe_target.log()).abs()
    return error.masked_select(mask).mean()
