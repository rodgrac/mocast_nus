from torchvision.models.resnet import resnet50
import torch
import math
from torch import nn, optim
from torch.nn.init import kaiming_normal_
import numpy as np
from orthnet import Legendre_Normalized
from orthnet.backend import NumpyBackend
import torch.nn.functional as F
import torch_dct as dct
import torch.fft


# Multimodal Regression | PolyFit | MetaLR
class MOCAST4_METALR(nn.Module):
    def __init__(self, in_ch, out_frames, degree, modes):
        super().__init__()
        self.degree = degree
        self.modes = modes
        self.out_pts = ((degree + 1) * 2) * self.modes
        self.basis_norm = False
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_ch, self.resnet.conv1.out_channels, kernel_size=self.resnet.conv1.kernel_size,
                                      stride=self.resnet.conv1.stride, padding=self.resnet.conv1.padding, bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=448)

        print("Num modes: ", self.modes)

        # Agent state embedding (x, y, vel, acc, yawr)
        self.state_fc = nn.Linear(in_features=5, out_features=64)
        self.enc_lstm = nn.LSTM(64, 64, batch_first=True)
        self.enc_lstm_fc = nn.Linear(in_features=64, out_features=64)

        self.cls_fc = nn.Linear(in_features=512, out_features=modes)

        self.final_fc1 = nn.Linear(in_features=512, out_features=256)
        # self.final_fc2 = nn.Linear(in_features=256, out_features=256)
        self.l_relu = nn.ReLU()

        self.t_n = np.arange(-6, out_frames + 1, dtype=np.float32)
        self.t_n = self.t_n / self.t_n[-1]

        self.final_fc2 = nn.Linear(in_features=256, out_features=self.out_pts)

        self.inner_parameters = [
            {'params': self.final_fc1.parameters()},
            {'params': self.final_fc2.parameters()},
            # {'params': self.final_fc3.parameters()}
        ]

        self.outer_parameters = [
            {'params': self.cls_fc.parameters()},
            {'params': self.final_fc1.parameters()},
            {'params': self.final_fc2.parameters()},
            # {'params': self.final_fc3.parameters()}
        ]

        # Legendre Orthogonal basis matrix
        self.tmat = torch.from_numpy(Legendre_Normalized(np.expand_dims(self.t_n, 1), degree).tensor).T

        self.sm = nn.Softmax(dim=1)

    def forward(self, x, device, state=None, state_len=None, out_type=3):
        self.tmat = self.tmat.to(device)
        enc_h_s = torch.zeros(1, x.size(0), 64).to(device)
        enc_c_s = torch.zeros(1, x.size(0), 64).to(device)
        if out_type != 3:
            enc_h_s = nn.init.xavier_normal_(enc_h_s)
            enc_c_s = nn.init.xavier_normal_(enc_c_s)

        # Variable length state LSTM
        state = self.state_fc(state.float())
        state_len = torch.clamp(state_len, min=1).type(torch.LongTensor)

        state = torch.nn.utils.rnn.pack_padded_sequence(state, state_len.to('cpu'), batch_first=True,
                                                        enforce_sorted=False).float()
        self.enc_lstm.flatten_parameters()
        out, _ = self.enc_lstm(state, (enc_h_s, enc_c_s))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.enc_lstm_fc(out[torch.arange(out.size(0)), state_len - 1, :])

        # Concatenate state encoding with resnet encoding
        out = torch.cat((self.resnet(x), out), dim=1)

        conf = self.cls_fc(out)

        out = self.final_fc1(out)
        out = self.l_relu(out)

        # out = self.final_fc2(out)
        # out = self.l_relu(out)

        out = self.final_fc2(out)
        # out = self.decoder(out, dec_params)
        out = out.view(x.size(0), self.modes, -1).clone()

        # conf = out[:, :, -1]
        # out = out[:, :, :(self.degree + 1) * 2].clone()

        # out_type: 0 (history); 1 (future); 2 (both_train); 3 (both_eval)
        if out_type == 0:
            out_x = torch.matmul(out[:, :, :self.degree + 1], self.tmat[:, :7])
            out_y = torch.matmul(out[:, :, self.degree + 1:], self.tmat[:, :7])
        elif out_type == 1:
            out_x = torch.matmul(out[:, :, :self.degree + 1], self.tmat[:, 7:])
            out_y = torch.matmul(out[:, :, self.degree + 1:], self.tmat[:, 7:])
        else:
            out_x = torch.matmul(out[:, :, :self.degree + 1], self.tmat)
            out_y = torch.matmul(out[:, :, self.degree + 1:], self.tmat)

        if out_type == 3:
            # Testing
            # Pick top N modes
            (_, top_idx) = torch.topk(conf, 10)
            out_x = torch.gather(out_x, 1, top_idx.unsqueeze(dim=-1).repeat(1, 1, out_x.size(2)))
            out_y = torch.gather(out_y, 1, top_idx.unsqueeze(dim=-1).repeat(1, 1, out_y.size(2)))
            conf = torch.gather(conf, 1, top_idx)
            return torch.stack((out_x, out_y), dim=3).detach(), conf.detach()
        else:
            # Training
            return torch.stack((out_x, out_y), dim=3), conf
