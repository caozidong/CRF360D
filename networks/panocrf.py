import torch
import torch.nn as nn
import torch.nn.functional as F

from .pano_layers import *

from argparse import Namespace
from .models import register
########################################################################################################################

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class EfficientNetEncoder(nn.Module):
    def __init__(self):
        super(EfficientNetEncoder, self).__init__()
        basemodel_name = 'tf_efficientnet_b5_ap'
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == 'blocks':
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features

class PanoDepth(nn.Module):
   
    def __init__(self, args):
        super().__init__()

        self.depth = args.depth 
        self.rotation = args.rotation
        self.localconv = args.localconv
        self.interact = args.interact
        
        self.encoder = EfficientNetEncoder()

        self.conv3 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0)

        self.up_3 = Up(176 + 1024, 512)
        self.up_2 = Up(64 + 512, 256)
        self.up_1 = Up(40 + 256, 128)

        win = 4
        crf_dims = [128, 256, 512, 1024]
        self.layer_3 = PanoLayer(embed_dim=crf_dims[3], window_size=win, num_heads=64, depth = self.depth, rotation=self.rotation, localconv=self.localconv, interact=self.interact)
        self.layer_2 = PanoLayer(embed_dim=crf_dims[2], window_size=win, num_heads=32, depth = self.depth, rotation=self.rotation, localconv=self.localconv, interact=self.interact)
        self.layer_1 = PanoLayer(embed_dim=crf_dims[1], window_size=win, num_heads=16, depth = self.depth, rotation=self.rotation, localconv=self.localconv, interact=self.interact)
        self.layer_0 = PanoLayer(embed_dim=crf_dims[0], window_size=win, num_heads=8, depth = self.depth, rotation=self.rotation, localconv=self.localconv, interact=self.interact)

        self.disp_head = DispHead(input_dim=crf_dims[0])
        
        self.min_depth = args.min_depth
        self.max_depth = args.max_depth

        self.relu = nn.ReLU(False)

    def forward(self, imgs, others):

        feats = self.encoder(imgs)
        x0, x1, x2, x3 = feats[5], feats[6], feats[8], feats[11]
        x3 = self.conv3(x3)

        e3 = self.layer_3(x3)

        f2 = self.up_3(e3, x2)
        e2 = self.layer_2(f2)

        f1 = self.up_2(e2, x1)
        e1 = self.layer_1(f1)
    
        f0 = self.up_1(e1, x0)
        e0 = self.layer_0(f0)

        d = self.disp_head(e0, 4)

        depth = d * self.max_depth

        outputs = {}
        outputs["pred_depth"] = depth 
        return outputs


class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        self.conv = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

@register('panocrf')
def make_newcrf(encoder='efficientnet', min_depth=0.1, max_depth=10.0, depth=2, rotation=True, \
                localconv=True, interact=True):

    args = Namespace()
    args.encoder = encoder
    args.min_depth = min_depth
    args.max_depth = max_depth
    args.depth = depth
    args.rotation = rotation
    args.localconv = localconv
    args.interact = interact

    return PanoDepth(args)