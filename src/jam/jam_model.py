from torchvision.models.resnet import resnet50
import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as f


class ResNetConv_512(nn.Module):
    def __init__(self, org_model):
        super(ResNetConv_512, self).__init__()
        self.features = nn.Sequential(*list(org_model.children())[:-2])

    def forward(self, x):
        return self.features(x)


class JAM_TFR(nn.Module):
    def __init__(self, in_ch, out_frames, degree, modes, train=True):
        super().__init__()
        self.degree = degree
        self.modes = modes
        self.out_pts = ((degree + 1) * 2 + 1) * self.modes
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_ch, self.resnet.conv1.out_channels, kernel_size=self.resnet.conv1.kernel_size,
                                      stride=self.resnet.conv1.stride, padding=self.resnet.conv1.padding, bias=False)

        self.resnet_strip = ResNetConv_512(self.resnet)

        print("Num modes: ", self.modes)

        # Agent state embedding (x, y, vel, acc, yawr)
        self.state_fc = nn.Linear(in_features=5, out_features=64)
        self.enc_lstm = nn.LSTM(64, 64, batch_first=True)

        self.enc_lstm_fc = nn.Linear(in_features=64, out_features=64)

        self.dec_lstm = nn.LSTM(128, 128, batch_first=True)


    def forward(self, x):
        out = self.resnet_strip(x)
        state_tensor = torch.zeros(out.size(0), 64, out.size(2), out.size(3))
