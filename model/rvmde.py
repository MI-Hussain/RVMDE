import torch
import torch.nn as nn
import torch.nn.functional as F
from model.radar_retinanet import ResNet102, ResNet_radar

"""
Parts of the code is borrowed from https://github.com/lochenchou/DORN_radar
If you use this code for your research please cite him as well.

For further details please visit https://github.com/lochenchou/DORN_radar

"""

class RVMDE(torch.nn.Module): #RVMDE

    def __init__(self, ord_num=80, input_size=(350, 800), kernel_size=16, pyramid=[6, 12, 18], pretrained=True):
        super().__init__()
        assert len(input_size) == 2
        assert isinstance(kernel_size, int)
        self.size = input_size
        self.ord_num = ord_num
        self.retinanet = ResNet102(pretrained=pretrained)
        self.resnet_radar = ResNet_radar(pretrained=pretrained)
        
        self.conv_radar = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1280, 2048, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, ord_num * 2, kernel_size=1)
        )
        self.ord_regression_layer = OrdinalRegressionLayer()
        
    def forward(self, image, radar):

        N, C, H, W = image.shape
        img_feat = self.retinanet(image)
        radar_feat = self.resnet_radar(radar)
        radar_feat = self.conv_radar(radar_feat)
        feat = torch.cat((img_feat, radar_feat), dim=1)
        feat = self.concat_process(feat)
        feat = F.interpolate(feat, size=self.size, mode="bilinear", align_corners=True)
        prob = self.ord_regression_layer(feat) 
        
        label = torch.sum((prob >= 0.5), dim=1).view(-1, 1, H, W)
    
        return prob, label



class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):

        N, C, H, W = x.size() 
        ord_num = C // 2
        
        label_0 = x[:, 0::2, :, :].clone().view(N, 1, ord_num, H, W)
        label_1 = x[:, 1::2, :, :].clone().view(N, 1, ord_num, H, W)  

        label = torch.cat((label_0, label_1), dim=1) 
        label = torch.clamp(label, min=1e-8, max=1e8) 

        label_ord = torch.nn.functional.softmax(label, dim=1)
        prob = label_ord[:,1,:,:,:].clone() 
        return prob
