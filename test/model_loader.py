
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
# import ...
from models_final import *
from torchvision import models

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((198,198)),
                 torchvision.transforms.ToTensor(), 
                 torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                  std = [0.229, 0.224, 0.225])])
    return transform
# For road map task
def get_transform_task2(): 
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((198,198)),
                 torchvision.transforms.ToTensor(), 
                 torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                  std = [0.229, 0.224, 0.225])])
    return transform

class ModelLoader():

    def __init__(self, model_file='models_final.py'):
        #  1. create the model object
        #  2. load your state_dict
        #  3. call cuda()
        # road segmentation
        rep_net1 = Representation_Generator().to(device)
        rep_net2 = Representation_Generator().to(device)
        rep_net1.load_state_dict(torch.load('rep_net3.pth'))
        rep_net2.load_state_dict(torch.load('rep_net3.pth'))

        #allow reconstruction of topdown view and object detection to use separate backbone
        self.reconstruct_backbone = rep_net1.backbone 
        self.backbone = models.segmentation.fcn_resnet50(num_classes = 2, pretrained=False).backbone
        self.detection_backbone = resnet_fpn_backbone(rep_net2.backbone) 

        self.classifier = UNet(2048, 2)
        self.road_model = Road_Layout(self.backbone, self.classifier).to(device)

        # object detection
        self.num_classes = 10
        self.anchor_generator = AnchorGenerator( ((8,),(16,),(32,), (64,), (128,)),
                                        ((0.5, 1.0, 2.0),) *5)
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], #cooresponds to 4 layers in resnet
                                                        output_size=7,
                                                        sampling_ratio=2)

        ## put the pieces together inside a FasterRCNN model, input needs to be (batch_size, channels, H, W)
        self.detect_model = FasterRCNN(self.detection_backbone,
                        num_classes = self.num_classes,
                        rpn_anchor_generator = self.anchor_generator,
                        box_roi_pool = self.roi_pooler)

        self.FL = Fusion_Layer(self.reconstruct_backbone, 2048, road_model_feat = None)
        self.box_model = Box_Model(self.FL, self.detect_model).to(device)

        #load model state_dict
        self.road_model.load_state_dict(torch.load('road_map_unet.pth'))
        self.road_model.eval()

        self.box_model.load_state_dict(torch.load('box_detect_SSL.pth'))
        self.box_model.eval()

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        with torch.no_grad():
            output = self.box_model(samples)
            pred_boxes = output['boxes']
            if pred_boxes[0].numel() == 0:
                return tuple(torch.zeros(len(samples),1,2,4))
        return pred_boxes

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # return a cuda tensor with size [batch_size, 800, 800] 
        with torch.no_grad():
            output = self.road_model(samples)
            
        return torch.argmax(output, dim=1)
