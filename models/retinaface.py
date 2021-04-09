import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models.resnet as resnet
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils import mkldnn as mkldnn_utils

from models.mobilev1 import MobileNetV1 as MobileNetV1
from models.mobilev1 import FPN as FPN
from models.mobilev1 import SSH as SSH
from models.detnas.network import ShuffleNetV2DetNAS

from torch.quantization import QuantStub, DeQuantStub

import timm


class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)
        # self.output_act = nn.LogSoftmax(dim=-1)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        # b, h, w, c = out.shape
        # out = out.view(b, h, w, self.num_anchors, 2)
        # out = self.output_act(out)
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, phase = 'train', net = 'mnet0.25', return_layers = {'stage1': 1, 'stage2': 2, 'stage3': 3}):
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        in_channels_list = [ 64, 128, 256]
        if net == 'mnet0.25':
            backbone = MobileNetV1()
            if True:
                checkpoint = torch.load("model_best.pth.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
            self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        elif net == 'detnas':
            backbone = ShuffleNetV2DetNAS(model_size='VOC_RetinaNet_300M')
            checkpoint = torch.load("VOC_RetinaNet_300M.pkl", map_location=torch.device('cpu'))
            backbone.load_state_dict(checkpoint)
            return_layers = {'6': 1, '9': 2, '16': 3}
            backbone = backbone.features
            self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
            in_channels_list = [ 160, 320, 640 ]
        elif net == 'mnet2':
            self.body = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True, out_indices=(0,1,2))
            in_channels_list = [ 16, 24, 32 ]

        out_channels = in_channels_list[0]
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(inchannels=out_channels)
        self.BboxHead = self._make_bbox_head(inchannels=out_channels)
        self.LandmarkHead = self._make_landmark_head(inchannels=out_channels)
        self.lin = nn.Conv2d(3, 16, kernel_size=(3,3), stride=(2,2), padding=1)

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def activate_mkldnn(self, inplace=True):
         # convert to mkldnn
         self.body.eval()
         self.ssh1.eval()
         self.ssh2.eval()
         self.ssh3.eval()
         self.ClassHead.eval()
         self.BboxHead.eval()
         self.LandmarkHead.eval()
         if inplace:
             self.body = mkldnn_utils.to_mkldnn(self.body)
             self.ssh1 = mkldnn_utils.to_mkldnn(self.ssh1)
             self.ssh2 = mkldnn_utils.to_mkldnn(self.ssh2)
             self.ssh3 = mkldnn_utils.to_mkldnn(self.ssh3)
             self.lin = mkldnn_utils.to_mkldnn(self.lin)
             #self.ClassHead = mkldnn_utils.to_mkldnn(self.ClassHead)
             #self.BboxHead = mkldnn_utils.to_mkldnn(self.BboxHead)
             #self.LandmarkHead = mkldnn_utils.to_mkldnn(self.LandmarkHead)
         else:
             self.body_mkl = mkldnn_utils.to_mkldnn(self.body)
             self.ssh1_mkl = mkldnn_utils.to_mkldnn(self.ssh1)
             self.ssh2_mkl = mkldnn_utils.to_mkldnn(self.ssh2)
             self.ssh3_mkl = mkldnn_utils.to_mkldnn(self.ssh3)
             self.ClassHead_mkl = mkldnn_utils.to_mkldnn(self.ClassHead)
             self.BboxHead_mkl = mkldnn_utils.to_mkldnn(self.BboxHead)
             self.LandmarkHead_mkl = mkldnn_utils.to_mkldnn(self.LandmarkHead)

             return self.body_mkl, self.ssh1_mkl, self.ssh2_mkl, self.ssh3_mkl,\
                    self.ClassHead_mkl, self.BboxHead_mkl, self.LandmarkHead_mkl

    def forward(self,inputs):
        out = self.body(self.lin(inputs))
        if inputs.is_mkldnn:
            for k, v in out.items():
                out[k] = v.to_dense()

        # FPN
        fpn = self.fpn(out)

        if inputs.is_mkldnn:
            for i in range(len(fpn)):
                fpn[i] = fpn[i].to_mkldnn()

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            #bbox_regressions = self.dequant(bbox_regressions)
            #classifications = self.dequant(classifications)
            #ldm_regressions = self.dequant(ldm_regressions)
            output = (bbox_regressions, F.softmax(classifications, dim=-1).select(2,1), ldm_regressions)
        return output
