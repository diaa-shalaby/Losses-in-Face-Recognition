from utils import ImageClassificationBase
from torch import nn


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    
    if pool: 
        layers.append(nn.MaxPool2d(2))
        
    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Input: <batch_size - currently 32>
        self.conv1 = conv_block(in_channels, 64)                     # bs x 64 x 64 x 64
        self.conv2 = conv_block(64, 128, pool=True)                  # bs x 128 x 32 x 32 
        self.res1 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)                 # bs x 256 x 16 x 16 
        self.conv4 = conv_block(256, 512, pool=True)                 # bs x 512 x 8 x 8 
        self.res2 = nn.Sequential(conv_block(512, 512), 
                                  conv_block(512, 512))

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),     # bs x 512 x 1 x 1 
                                        nn.Flatten(),                # bs x 512  
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes)) # bs x num_classes

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
    
    