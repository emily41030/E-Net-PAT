import torch
import torch.nn as nn
import utils
import torchvision.models as models
import os
import urllib.request
from collections import namedtuple


class Discriminator(nn.Module):
    def __init__(self, init_weights=True):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 32, kernel_size=(3, 3),
                      stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 64, kernel_size=(3, 3),
                      stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(True),
            nn.Conv2d(128, 128, kernel_size=(3, 3),
                      stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(True),
            nn.Conv2d(128, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(True),
            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(True)
        )
        self.f_2 = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.f_2(x)
        m = nn.Sigmoid()
        x = m(x)
        return x


class Net_LReLU(nn.Module):
    def __init__(self, input_channels, channels, kernel_size, padding):
        super(Net_LReLU, self).__init__()

        self.model = torch.nn.Sequential(
            nn.Conv2d(input_channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            # ====================resi_1===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_2===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_3===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_4===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_5===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_6===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_7===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_8===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_9===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_10===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # '''==============================================='''
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, input_channels,
                      kernel_size=kernel_size, padding=padding)
        )

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)

    def forward(self, x):
        out = self.model(x)
        return out


class Net(nn.Module):
    def __init__(self, input_channels, channels, kernel_size, padding):
        super(Net, self).__init__()

        self.model = torch.nn.Sequential(
            nn.Conv2d(input_channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            # ====================resi_1===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_2===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_3===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_4===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_5===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_6===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_7===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_8===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_9===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # ====================resi_10===================='''
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            # '''==============================================='''
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(channels, input_channels,
                      kernel_size=kernel_size, padding=padding)
        )

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)

    def forward(self, x):
        out = self.model(x)
        return out

class LossNet(nn.Module):
    def creat_loss_Net(self):
        self.vgg19 = models.vgg19(pretrained=True).features
        ################# VGG #################
        self.VGG_m2_model = nn.Sequential(
            *list(self.vgg19.children())[:5])
        self.VGG_m5_model = nn.Sequential(
            *list(self.vgg19.children())[:10])
        for param in self.VGG_m2_model.parameters():
            param.requires_grad = False
        for param in self.VGG_m5_model.parameters():
            param.requires_grad = False

        ################ Texture #############
        self.conv1_1 = nn.Sequential(
            *list(self.vgg19.children())[:1])
        self.conv2_1 = nn.Sequential(
            *list(self.vgg19.children())[:6])
        self.conv3_1 = nn.Sequential(
            *list(self.vgg19.children())[:11])
        for param in self.conv1_1.parameters():
            param.requires_grad = False
        for param in self.conv2_1.parameters():
            param.requires_grad = False
        for param in self.conv3_1.parameters():
            param.requires_grad = False

        if self.gpu_mode:
            self.conv1_1.cuda()
            self.conv2_1.cuda()
            self.conv3_1.cuda()
            self.VGG_m2_model.cuda()
            self.VGG_m5_model.cuda()
            # self.T_model.cuda()
            self.loss = nn.MSELoss().cuda()
        else:
            self.loss = nn.MSELoss()

        if self.gpu_mode:
            self.normalization_mean = torch.tensor(
                [0.485, 0.456, 0.406]).cuda()
            self.normalization_std = torch.tensor(
                [0.229, 0.224, 0.225]).cuda()
        else:
            self.normalization_mean = torch.tensor([0.485, 0.456, 0.406])
            self.normalization_std = torch.tensor([0.229, 0.224, 0.225])

        return self


#####################  VGG19 model #########################
# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                                 <-conv1_1
#   (1): ReLU(inplace)
#   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU(inplace)
#   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                               <-conv2_1
#   (6): ReLU(inplace)
#   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU(inplace)
#   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                             <-conv3_1
#   (11): ReLU(inplace)
#   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU(inplace)
#   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU(inplace)
#   (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (17): ReLU(inplace)
#   (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (20): ReLU(inplace)
#   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (22): ReLU(inplace)
#   (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (24): ReLU(inplace)
#   (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (26): ReLU(inplace)
#   (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (29): ReLU(inplace)
#   (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (31): ReLU(inplace)
#   (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (33): ReLU(inplace)
#   (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (35): ReLU(inplace)
#   (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# )
