import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from helper import collate_fn, draw_box
from torchvision import models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from unet_parts import *

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# road segmentation
class FCNHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        """in_channels: dim of input feature map after fusion
           """
        self.inter_channels = in_channels // 4
        self.layers = [
            nn.Conv2d(in_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(self.inter_channels, num_classes, 1)
        ]

        super(FCNHead, self).__init__(*self.layers)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Road_Layout(nn.Module):
  def __init__(self, backbone, classifier):
        super(Road_Layout, self).__init__()
        #images, targets = self.transform(images, targets)
        self.backbone = backbone
        self.classifier = classifier
 

  def forward(self, images, targets=None):
        feats= []
        for view in range(6):
            feats.append(self.backbone(images[:,view,:])["out"])
            #concatenate feature map from all views
        #fused_feature = torch.cat(feats,dim = 1) #(batch_size, 6*fused channels, H, W )
        fused_feature = torch.mean(torch.stack(feats),dim = 0)
        x = self.classifier(fused_feature) #(batch size, num_classes, H,W)
        x = F.interpolate(x, size=(800,800), mode='bilinear', align_corners=False)

        return x #(batch_size, num_classes, 800,800)


# object detection
class Fusion_Layer(nn.Module):
    """Model to generate  800 * 800 size road map / Convert to Bird Eye View;
       road_model_feat is the feature map output from the road_model, assumed to have (h,w) the same as backbone output feat dim;
       mean?? project with camera intrinsics??"""
    def __init__(self, backbone, feature_channels, road_model_feat = None):
        super(Fusion_Layer, self).__init__()    
        tot_channels = feature_channels 
        self.road_model_feat = road_model_feat
        if road_model_feat is not None:
            tot_channels += road_model_feat.size()[1]
        
        #for mapping back to 800*800
        self.backbone = backbone
        self.relu =  nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(tot_channels, tot_channels, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(tot_channels)
        self.deconv2 = nn.ConvTranspose2d(tot_channels, tot_channels//4, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(tot_channels//4)
        self.deconv3 = nn.ConvTranspose2d(tot_channels//4, tot_channels//16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(tot_channels//16)
        self.deconv4 = nn.ConvTranspose2d(tot_channels//16, tot_channels//16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(tot_channels//16)
        self.deconv5 = nn.ConvTranspose2d(tot_channels//16, 3, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(3)

    def forward(self, images):
        feats= []
        for view in range(6):
            feats.append(self.backbone(images[:,view,:])["out"])
        fused_feature = torch.mean(torch.stack(feats),dim = 0)
        # fused_feature = torch.cat(feats,dim = 1)
        if self.road_model_feat is not None:
            fused_feature = toch.cat([fused_feature,self.road_model_feat],dim = 1)

        #transform into (batch_size, channels, 800,800)
        x1 = self.bn1(self.relu(self.deconv1(fused_feature))) # (H,W) -> 2*(H,W)
        x2 = self.bn2(self.relu(self.deconv2(x1)))
        x3 = self.bn3(self.relu(self.deconv3(x2)))
        x4 = self.bn4(self.relu(self.deconv4(x3)))
        x5 = self.bn5(self.relu(self.deconv5(x4)))

        return x5

class Box_Model(nn.Module):
    def __init__(self, fuse_layer, detect_model):
        super(Box_Model, self).__init__()
        self.fuse_layer = fuse_layer
        self.detect_model  = detect_model

    def transform_target(self,targets):
        res = [] #targets should be a list of dictionaries with key "boxes" and "labels"
        for t in targets:
            N = t['category'].size(0) #number of boxes
            bbox = torch.zeros(N,4)
            for n in range(N):
                nth_box = 10*t['bounding_box'][n] #shape (2 ,4) multiply 10 because the original value is in meters!
                xmax = torch.max(nth_box[0,:]) 
                xmin = torch.min(nth_box[0,:])
                ymax = torch.max(nth_box[1,:]) 
                ymin = torch.min(nth_box[1,:])
                bbox[n] = torch.tensor([xmin,ymin,xmax,ymax]) + 400
            res.append({"boxes":bbox.to(device),"labels":t['category'].to(device)})
        return res 

    def get_output(self,preds):
        pred_box = []
        pred_label = []
        for p in preds: 
            nbox = p['boxes'].size(0)
            res_box = torch.zeros(nbox, 2, 4)
            res_label = torch.zeros(nbox)
            for n in range(nbox):
                xmin, ymin, xmax, ymax = p['boxes'][n] #in pixel level
                res_box[n] = torch.tensor([[xmax, xmax, xmin, xmin],[ymax,ymin,ymax, ymin]])
                res_label[n] = p['labels'][n]
            res_box = (res_box - 400)/10 #the unit should be meter instead of pixels
            pred_box.append(res_box)
            pred_label.append(res_label)
        return {"boxes":tuple(pred_box), "labels":tuple(pred_label)}


    def forward(self, images, targets  = None):

        top_down = self.fuse_layer(images) #(batch_size, 3, 800, 800)
        top_down  = [i for i in top_down]
        if self.training:
            self.detect_model.train() 
            targets = self.transform_target(targets)
            output = self.detect_model(top_down,targets) #loss_dict
        else:
            preds = self.detect_model(top_down)#list of dictionary of keys 'boxes', 'labels', 'scores'
            #need to transform predicted boxes coordinates, a torch tensor of size (num_boxes, 2, 4)
            output = self.get_output(preds)

        return output #at eval mode, output is a dictionary: "boxes": a tuple of tensors of size (num_boxes, 2, 4), "labels": a tuple of tensor (num_boxes)

#backbone class from pirl jigsaw
class Representation_Generator(nn.Module):
    """Class that returns features for original image and fused features for image patches;
       its backbone is what we used for downstream task. """

    def __init__(self):
        super(Representation_Generator, self).__init__()
        #self.backbone = torch.nn.Sequential(*list(resnet50().children())[:-2]) 
        self.backbone = torchvision.models.segmentation.fcn_resnet50(pretrained=False).backbone
        #Should we add pyramid network in it as in faster_rcnn ???
        self.pool = nn.AdaptiveAvgPool2d((1,1)) #pool the spatial dimension to be 1*1
        self.head_f = nn.Linear(2048, 128) 
        self.head_g = nn.Linear(9*2048,128)

    def forward(self, images, patches = None):
        image_feat = self.pool(self.backbone(images)['out'])
        image_feat = image_feat.view(-1,2048) #batch size, 2048
        image_feat = self.head_f(image_feat) #batch size, 128

        if patches is not None:
            patches_feat = []
            for i, patch in enumerate(patches):
                
                patch_feat = self.pool(self.backbone(patch)['out']) #batch size, 2048, 1,1
                patch_feat = patch_feat.view(-1,2048) # batch_size,2048
                patches_feat.append(patch_feat)
         
            patches_feat = torch.cat(patches_feat, axis = 1) #batch size, 2048*9, 
            patches_feat = self.head_g(patches_feat)   #batch size, 128  

            return image_feat, patches_feat
        else:
            return image_feat


#phrase pirl backbone for object detection: add FPN and freeze BN
def resnet_fpn_backbone(backbone, freeze_bn = True):

    # copied the behaviour from faster_rcnn, freeze layer1
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = 2048 // 8 #64 is resnet's inplanes
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    bb_fpn = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
   
    def set_bn_eval(m): 
        
        #freeze the batchnorm2d layer: in object detection we are using small batch size, so we don't want to track batch statistics cause they are poor
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.eval()
    
    if freeze_bn:
        return bb_fpn.apply(set_bn_eval)
    else:
        return bb_fpn