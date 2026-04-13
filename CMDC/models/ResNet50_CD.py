import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
# from base import BaseModel
# from utils.helpers import set_trainable
# from utils.losses import *
from models.decoders import *
from models.encoder import Encoder
# from utils.losses import CE_loss

class ResNet50_CD(nn.Module):
    def __init__(self, num_classes, pretrained=None):
        super(ResNet50_CD, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        # create the model
        self.encoder = Encoder(pretrained=pretrained)
        upscale = 8
        num_out_ch = 2048
        decoder_in_ch = num_out_ch // 4
        self.decoder = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)
        self.sdecoder =  MainDecoder(upscale, decoder_in_ch, num_classes=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(2048,1024)

    def forward(self, x, mask=None, return_features=False):
        if return_features:  # return change predictions and features
            features,a,b = self.encoder(x[0], x[1])
            if mask != None:
                z1 = x[0]*mask
                z2 = x[1]*mask
                zfeatures,za,zb = self.encoder(z1, z2)
                return za,zb
            # a = self.sdecoder(a)
            # b = self.sdecoder(b)
            return self.decoder(features), features,a,b
        else:
            features,a,b = self.encoder(x[0], x[1])
            if mask != None:
                z1 = x[0]*mask
                z2 = x[1]*mask
                zfeatures,za,zb = self.encoder(z1, z2)
                za = self.avgpool(za)
                za = torch.flatten(za,1)
                za = self.linear(za)
                zb = self.avgpool(zb)
                zb = torch.flatten(zb,1)
                zb = self.linear(zb)
                return za,zb
            # a = self.sdecoder(a)
            # b = self.sdecoder(b)
            return self.decoder(features),a,b

    def pretrained_parameters(self):
        if self.pretrained:
            return list(self.encoder.get_backbone_params())
        else:
            return []

    def new_parameters(self):
        if self.pretrained:
            pretrained_ids = [id(p) for p in self.encoder.get_backbone_params()]
            return [p for p in self.parameters() if id(p) not in pretrained_ids]
        else:
            return list(self.parameters())

# net = ResNet50_CD(num_classes=2, pretrained=None)
# A = torch.rand([4,3,256,256])
# B = torch.rand([4,3,256,256])
# out = net(A,B)
# print(out.shape)




