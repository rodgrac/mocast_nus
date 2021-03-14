from torchvision.models.resnet import resnet50
import torch
from torch import nn, optim
import numpy as np
from orthnet import Legendre_Normalized
import torch.nn.functional as f


class ResNetConv_512(nn.Module):
    def __init__(self, org_model):
        super(ResNetConv_512, self).__init__()
        self.features = nn.Sequential(*list(org_model.children())[:-3])

    def forward(self, x):
        return self.features(x)



class JAM_TFR(nn.Module):
    def __init__(self, in_ch, out_frames, degree, modes):
        super().__init__()
        self.degree = degree
        self.modes = modes
        self.basis_norm = False
        self.out_pts = ((degree + 1) * 2) * self.modes
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_ch, self.resnet.conv1.out_channels, kernel_size=self.resnet.conv1.kernel_size,
                                      stride=self.resnet.conv1.stride, padding=self.resnet.conv1.padding, bias=False)

        self.resnet_strip = ResNetConv_512(self.resnet)

        print("Num modes: ", self.modes)

        # Agent state embedding (x, y, vel, acc, yawr)
        self.state_fc = nn.Linear(in_features=5, out_features=64)
        self.enc_lstm = nn.LSTM(64, 64, batch_first=True)
        self.enc_lstm_fc = nn.Linear(in_features=64, out_features=64)

        self.enc_conv1 = nn.Conv2d(1024+64, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_ln1 = nn.LayerNorm([512, 14, 14])
        self.enc_act = nn.ReLU(inplace=True)

        self.enc_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc_ln2 = nn.LayerNorm([512, 7, 7])
        self.enc_avg_pool = nn.AvgPool2d(7)

        self.cls_fc = nn.Linear(in_features=512+64, out_features=modes)

        self.dec_fc1 = nn.Linear(in_features=512+64, out_features=256)
        self.l_relu = nn.ReLU()

        self.t_n = np.arange(-6, out_frames + 1, dtype=np.float32)

        if not self.basis_norm:
            self.t_n = self.t_n / self.t_n[-1]

        self.dec_fc2 = nn.Linear(in_features=256, out_features=self.out_pts)

        # Legendre Orthogonal basis matrix
        self.tmat = torch.from_numpy(Legendre_Normalized(np.expand_dims(self.t_n, 1), degree).tensor).T

        self.sm = nn.Softmax(dim=1)

    # Variable length state LSTM
    def state_lstm(self, state, state_len, device):
        enc_h_s = torch.zeros(1, state.size(0), 64).to(device)
        enc_c_s = torch.zeros(1, state.size(0), 64).to(device)
        # if not self.sm:
        #     enc_h_s = nn.init.xavier_normal_(enc_h_s)
        #     enc_c_s = nn.init.xavier_normal_(enc_c_s)
        state = self.state_fc(state.float())
        state_len = torch.clamp(state_len, min=1).type(torch.LongTensor)
        state = torch.nn.utils.rnn.pack_padded_sequence(state, state_len.to('cpu'), batch_first=True,
                                                        enforce_sorted=False).float()
        self.enc_lstm.flatten_parameters()
        lstm_out, _ = self.enc_lstm(state, (enc_h_s, enc_c_s))
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return self.enc_lstm_fc(lstm_out[torch.arange(lstm_out.size(0)), state_len - 1, :])

    def forward(self, x, device, ego_state, ego_state_len, agents_state, agents_state_len, agents_grid_pos, out_type=2):
        self.tmat = self.tmat.to(device)
        agents_grid_pos = agents_grid_pos.type(torch.LongTensor)
        cnn_tensor = self.resnet_strip(x)
        state_tensor = torch.zeros(cnn_tensor.size(0), cnn_tensor.size(2), cnn_tensor.size(3), 64).to(device)

        # Ego State LSTM
        ego_state = self.state_lstm(ego_state, ego_state_len, device)

        if len(agents_state.size()) < 4:
            agents_state = agents_state.unsqueeze(0)
        if len(agents_state_len.size()) < 2:
            agents_state_len = agents_state_len.unsqueeze(0)
        if len(agents_grid_pos.size()) < 3:
            agents_grid_pos = agents_grid_pos.unsqueeze(0)
        # Agents State LSTM
        if agents_state_len.numel():
            for ag in torch.arange(agents_state.size(1)):
                agent_state = self.state_lstm(agents_state[:, ag, :, :], agents_state_len[:, ag], device)
                state_tensor[torch.arange(x.size(0)), agents_grid_pos[:, ag, 0], agents_grid_pos[:, ag, 1], :] += agent_state

        out = torch.cat((cnn_tensor, state_tensor.permute(0, 3, 1, 2)), dim=1)

        out = self.enc_conv1(out)
        out = self.enc_ln1(out)
        out = self.enc_act(out)

        out = self.enc_conv2(out)
        out = self.enc_ln2(out)
        out = self.enc_act(out)

        out = self.enc_avg_pool(out).view(out.size(0), -1)
        out = torch.cat((out, ego_state), dim=1)

        conf = self.cls_fc(out)

        out = self.dec_fc1(out)
        out = self.l_relu(out)

        out = self.dec_fc2(out)
        out = out.view(x.size(0), self.modes, -1)

        out = out[:, :, :(self.degree + 1) * 2]

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
