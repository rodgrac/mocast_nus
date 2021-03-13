from torchvision.models.resnet import resnet50
import torch
from torch import nn, optim
import numpy as np
from orthnet import Legendre, Legendre_Normalized
from orthnet.backend import NumpyBackend
import torch.nn.functional as f
import torch_dct as dct
import torch.fft


class MOCAST_0(nn.Module):
    def __init__(self, in_ch, out_frames, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.out_frames = out_frames
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_ch, self.resnet.conv1.out_channels, kernel_size=self.resnet.conv1.kernel_size,
                                      stride=self.resnet.conv1.stride, padding=self.resnet.conv1.padding, bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=128)

        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.final_fc = nn.Linear(in_features=128 * out_frames, out_features=2 * out_frames)

    def forward(self, x, device):
        lstm_x = torch.zeros(x.size(0), self.out_frames, 128).to(device)
        h_s = torch.zeros(1, x.size(0), 128).to(device)
        c_s = torch.zeros(1, x.size(0), 128).to(device)

        lstm_x[:, 0, :] = self.resnet(x)
        # if out.shape[1] < self.batch_size:
        #     self.lstm_x = self.lstm_x[:out.shape[1], :, :]
        #     self.c_s = self.c_s[:, :out.shape[1], :]

        out, _ = self.lstm(lstm_x, (h_s, c_s))

        out = torch.flatten(out, start_dim=1)

        out = self.final_fc(out)

        return out


class MOCAST_1(nn.Module):
    def __init__(self, in_ch, out_frames, degree, batch_size=32, train=True):
        super().__init__()
        self.batch_size = batch_size
        self.degree = degree
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_ch, self.resnet.conv1.out_channels, kernel_size=self.resnet.conv1.kernel_size,
                                      stride=self.resnet.conv1.stride, padding=self.resnet.conv1.padding, bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=96)

        self.enc_lstm = nn.LSTM(5, 32, batch_first=True)

        self.final_fc = nn.Linear(in_features=128, out_features=(degree + 1) * 2)

        if train:
            t_n = np.array(range(-4, out_frames + 1), dtype=np.float32) / out_frames
        else:
            t_n = np.array(range(1, out_frames + 1), dtype=np.float32) / out_frames
        self.tmat = torch.from_numpy(np.vstack([t_n ** i for i in range(degree, -1, -1)]))

    def forward(self, x, device, state=None):
        enc_h_s = torch.zeros(1, x.size(0), 32).to(device)
        enc_c_s = torch.zeros(1, x.size(0), 32).to(device)

        out, _ = self.enc_lstm(state, (enc_h_s, enc_c_s))
        out = torch.cat((self.resnet(x), out[:, -1, :]), dim=1)

        out = self.final_fc(out)

        self.tmat = self.tmat.to(device)
        out_x = torch.matmul(out[:, :self.degree + 1], self.tmat)
        out_y = torch.matmul(out[:, self.degree + 1:], self.tmat)
        return torch.stack((out_x, out_y), dim=2)


class MOCAST_2(nn.Module):
    def __init__(self, in_ch, out_frames, degree, batch_size=32, train=True):
        super().__init__()
        self.batch_size = batch_size
        self.degree = degree
        self.out_frames = out_frames
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_ch, self.resnet.conv1.out_channels, kernel_size=self.resnet.conv1.kernel_size,
                                      stride=self.resnet.conv1.stride, padding=self.resnet.conv1.padding, bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=128)

        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.final_fc = nn.Linear(in_features=128 * out_frames, out_features=(degree + 1) * 2)

        if train:
            t_n = np.array(range(-4, out_frames + 1), dtype=np.float32) / out_frames
        else:
            t_n = np.array(range(1, out_frames + 1), dtype=np.float32) / out_frames
        self.tmat = torch.from_numpy(np.vstack([t_n ** i for i in range(degree, -1, -1)]))

    def forward(self, x, device, state=None):
        lstm_x = torch.zeros(x.size(0), self.out_frames, 128).to(device)
        h_s = torch.zeros(1, x.size(0), 128).to(device)
        c_s = torch.zeros(1, x.size(0), 128).to(device)

        lstm_x[:, 0, :] = self.resnet(x).to(device)
        out, _ = self.lstm(lstm_x, (h_s, c_s))
        out = torch.flatten(out, start_dim=1)
        out = self.final_fc(out)

        self.tmat = self.tmat.to(device)
        out_x = torch.matmul(out[:, :self.degree + 1], self.tmat)
        out_y = torch.matmul(out[:, self.degree + 1:], self.tmat)
        return torch.stack((out_x, out_y), dim=2)


class MOCAST_3(nn.Module):
    def __init__(self, in_ch, out_frames, degree, batch_size=32, train=True):
        super().__init__()
        self.batch_size = batch_size
        self.degree = degree
        self.out_frames = out_frames
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_ch, self.resnet.conv1.out_channels, kernel_size=self.resnet.conv1.kernel_size,
                                      stride=self.resnet.conv1.stride, padding=self.resnet.conv1.padding, bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=96)

        self.enc_lstm = nn.LSTM(5, 32, 2, batch_first=True)

        self.dec_lstm = nn.LSTM(128, 128, 2, batch_first=True)
        self.final_fc = nn.Linear(in_features=128 * out_frames, out_features=(degree + 1) * 2)

        if train:
            t_n = np.array(range(-4, out_frames + 1), dtype=np.float32) / out_frames
        else:
            t_n = np.array(range(1, out_frames + 1), dtype=np.float32) / out_frames
        self.tmat = torch.from_numpy(np.vstack([t_n ** i for i in range(degree, -1, -1)]))

    def forward(self, x, device, state=None):
        enc_h_s = torch.zeros(2, x.size(0), 32).to(device)
        enc_c_s = torch.zeros(2, x.size(0), 32).to(device)

        dec_lstm_x = torch.zeros(x.size(0), self.out_frames, 128).to(device)
        dec_h_s = torch.zeros(2, x.size(0), 128).to(device)
        dec_c_s = torch.zeros(2, x.size(0), 128).to(device)

        out, _ = self.enc_lstm(state, (enc_h_s, enc_c_s))
        out = torch.cat((self.resnet(x), out[:, -1, :]), dim=1).to(device)

        dec_lstm_x[:, 0, :] = out
        out, _ = self.dec_lstm(dec_lstm_x, (dec_h_s, dec_c_s))
        out = torch.flatten(out, start_dim=1)
        out = self.final_fc(out)

        self.tmat = self.tmat.to(device)
        out_x = torch.matmul(out[:, :self.degree + 1], self.tmat)
        out_y = torch.matmul(out[:, self.degree + 1:], self.tmat)
        return torch.stack((out_x, out_y), dim=2)


# Multimodal Regression | PolyFit
class MOCAST_4(nn.Module):
    def __init__(self, in_ch, out_frames, degree, modes, dec='ortho'):
        super().__init__()
        self.degree = degree
        self.modes = modes
        self.dec = dec
        self.basis_norm = False
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_ch, self.resnet.conv1.out_channels, kernel_size=self.resnet.conv1.kernel_size,
                                      stride=self.resnet.conv1.stride, padding=self.resnet.conv1.padding, bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=448)

        print("Num modes: {}, Decoder: {}".format(self.modes, self.dec))

        self.state_fc = nn.Linear(in_features=5, out_features=64)
        self.enc_lstm = nn.LSTM(64, 64, batch_first=True)

        self.enc_lstm_fc = nn.Linear(in_features=64, out_features=64)
        self.enc_cat_fc = nn.Linear(in_features=512, out_features=512)

        self.cls_fc = nn.Linear(in_features=512, out_features=modes)

        self.dec_fc1 = nn.Linear(in_features=512, out_features=256)
        self.l_relu = nn.ReLU()

        self.t_n = np.arange(-6, out_frames + 1, dtype=np.float32)

        if not self.basis_norm:
            self.t_n = self.t_n / self.t_n[-1]

        if self.dec in ['dct', 'fftc']:
            self.dec_fc2 = nn.Linear(in_features=256, out_features=((self.t_n.shape[0] * 2) * self.modes))
        elif self.dec == 'polytr':
            self.t_n = torch.from_numpy(self.t_n)
            self.dec_fc2 = nn.Linear(in_features=256, out_features=((degree + 1) * 2) * self.modes + 1)
        else:
            self.dec_fc2 = nn.Linear(in_features=256, out_features=((degree + 1) * 2) * self.modes)
            if self.dec == 'poly':
                self.tmat = torch.from_numpy(np.vstack([self.t_n ** i for i in range(degree, -1, -1)]))
            elif self.dec == 'ortho':
                self.tmat = torch.from_numpy(Legendre_Normalized(np.expand_dims(self.t_n, 1), degree).tensor).T

        self.sm = nn.Softmax(dim=1)

        if self.basis_norm:
            tmat_max, _ = torch.max(self.tmat, dim=1, keepdim=True)
            tmat_min, _ = torch.min(self.tmat, dim=1, keepdim=True)
            self.tmat = ((self.tmat - tmat_min) / (tmat_max - tmat_min))
            self.tmat = self.tmat - self.tmat[:, 2].unsqueeze(1)

    def forward(self, x, device, state=None, state_len=None, out_type=2):
        enc_h_s = torch.zeros(1, x.size(0), 64).to(device)
        enc_c_s = torch.zeros(1, x.size(0), 64).to(device)
        if not self.sm:
            enc_h_s = nn.init.xavier_normal_(enc_h_s)
            enc_c_s = nn.init.xavier_normal_(enc_c_s)

        state = self.state_fc(state.float())
        state_len = torch.clamp(state_len, min=1).type(torch.LongTensor)

        state = torch.nn.utils.rnn.pack_padded_sequence(state, state_len.to('cpu'), batch_first=True,
                                                        enforce_sorted=False).float()
        out, _ = self.enc_lstm(state, (enc_h_s, enc_c_s))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.enc_lstm_fc(out[torch.arange(out.size(0)), state_len - 1, :])

        out = torch.cat((self.resnet(x), out), dim=1)

        # out = torch.cat(self.modes * [out], dim=1).view(-1, self.modes, 128)

        out = self.enc_cat_fc(out)
        out = self.l_relu(out)

        conf = self.cls_fc(out)

        out = self.dec_fc1(out)
        out = self.l_relu(out)

        out = self.dec_fc2(out)

        if self.dec == 'polytr':
            eval_pt = out[:, -1]
            out = out[:, :-1]

        out = out.view(x.size(0), self.modes, -1)

        if self.dec in ['poly', 'ortho']:
            self.tmat = self.tmat.to(device)
            if out_type == 0:
                out_x = torch.matmul(out[:, :, :self.degree + 1], self.tmat[:, :7])
                out_y = torch.matmul(out[:, :, self.degree + 1:], self.tmat[:, :7])
            else:
                out_x = torch.matmul(out[:, :, :self.degree + 1], self.tmat)
                out_y = torch.matmul(out[:, :, self.degree + 1:], self.tmat)
        elif self.dec == 'polytr':
            t_n_offset = self.t_n.unsqueeze(0).repeat(x.size(0), 1) - eval_pt.unsqueeze(-1)
            self.tmat = torch.stack([t_n_offset ** i for i in range(self.degree, -1, -1)], dim=1)
            out_x = torch.einsum('bmd,bds->bms', out[:, :, :self.degree + 1], self.tmat)
            out_y = torch.einsum('bmd,bds->bms', out[:, :, self.degree + 1:], self.tmat)
        elif self.dec == 'dct':
            out_x = dct.idct(out[:, :, :self.t_n.shape[0]])
            out_y = dct.idct(out[:, :, self.t_n.shape[0]:])
        elif self.dec == 'fftc':
            out = out.view(x.size(0), self.modes, -1, 2)
            out = torch.ifft(out, 1, normalized=True)
            out_x, out_y = out[:, :, :, 0], out[:, :, :, 1]

        if out_type == 2:
            return torch.stack((out_x, out_y), dim=3), conf
        elif out_type == 3:
            (_, top_idx) = torch.topk(conf, 10)
            out_x = torch.gather(out_x, 1, top_idx.unsqueeze(dim=-1).repeat(1, 1, out_x.size(2)))
            out_y = torch.gather(out_y, 1, top_idx.unsqueeze(dim=-1).repeat(1, 1, out_y.size(2)))
            conf = torch.gather(conf, 1, top_idx)
            return torch.stack((out_x, out_y), dim=3).detach(), conf.detach()


# Multimodal Regression | DecLSTM | PolyFit
class MOCAST_5(nn.Module):
    def __init__(self, in_ch, out_frames, degree, modes, batch_size=32, train=True):
        super().__init__()
        self.batch_size = batch_size
        self.degree = degree
        self.modes = modes
        self.out_frames = out_frames
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_ch, self.resnet.conv1.out_channels, kernel_size=self.resnet.conv1.kernel_size,
                                      stride=self.resnet.conv1.stride, padding=self.resnet.conv1.padding, bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=448)

        print("Num modes: ", self.modes)

        self.state_fc = nn.Linear(5, 64)
        self.enc_lstm = nn.LSTM(64, 64, batch_first=True)

        self.enc_lstm_fc = nn.Linear(in_features=64, out_features=64)

        self.final_fc1 = nn.Linear(in_features=512, out_features=128)
        self.l_relu = nn.ReLU()

        self.dec_lstm = nn.LSTM(128, 128, batch_first=True)

        self.t_n = np.array(range(-6, out_frames + 1), dtype=np.float32)

        self.final_fc2 = nn.Linear(in_features=128 * out_frames, out_features=(self.t_n.shape[0] * 2 + 1) * self.modes)

        if train:
            # t_n = np.array(range(-6, out_frames + 1), dtype=np.float32) / out_frames
            self.sm = None
        else:
            # t_n = np.array(range(1, out_frames + 1), dtype=np.float32) / out_frames
            self.sm = nn.Softmax(dim=1)
        # self.tmat = torch.from_numpy(np.vstack([t_n ** i for i in range(degree, -1, -1)]))
        # self.tmat = torch.from_numpy(Legendre_Normalized(np.expand_dims(t_n, 1), degree).tensor).T

    def forward(self, x, device, state=None, state_len=None):
        enc_h_s = torch.zeros(1, x.size(0), 64).to(device)
        enc_c_s = torch.zeros(1, x.size(0), 64).to(device)
        enc_h_s = nn.init.xavier_normal_(enc_h_s)
        enc_c_s = nn.init.xavier_normal_(enc_c_s)

        dec_lstm_x = torch.zeros(x.size(0), self.out_frames, 128).to(device)
        dec_h_s = torch.zeros(1, x.size(0), 128).to(device)
        dec_c_s = torch.zeros(1, x.size(0), 128).to(device)
        dec_h_s = nn.init.xavier_normal_(dec_h_s)
        dec_c_s = nn.init.xavier_normal_(dec_c_s)

        state = self.state_fc(state.float())
        state_len = torch.clamp(state_len, min=1).type(torch.LongTensor)

        state = torch.nn.utils.rnn.pack_padded_sequence(state, state_len.to('cpu'), batch_first=True,
                                                        enforce_sorted=False).float()
        out, _ = self.enc_lstm(state, (enc_h_s, enc_c_s))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.enc_lstm_fc(out[torch.arange(out.size(0)), state_len - 1, :])

        out = torch.cat((self.resnet(x), out), dim=1)

        # out = torch.cat(self.modes * [out], dim=1).view(-1, self.modes, 128)

        out = self.final_fc1(out)
        out = self.l_relu(out)

        dec_lstm_x[:, 0, :] = out
        out, _ = self.dec_lstm(dec_lstm_x, (dec_h_s, dec_c_s))
        out = torch.flatten(out, start_dim=1)

        out = self.final_fc2(out).view(x.size(0), self.modes, -1)

        # self.tmat = self.tmat.to(device)
        # out_x = torch.matmul(out[:, :, :self.degree + 1], self.tmat)
        # out_y = torch.matmul(out[:, :, self.degree + 1:-1], self.tmat)
        out_x = dct.idct(out[:, :, :self.t_n.shape[0]])
        out_y = dct.idct(out[:, :, self.t_n.shape[0]:-1])

        if self.sm:
            (_, top_idx) = torch.topk(out[:, :, -1], 10)
            out_x = torch.gather(out_x, 1, top_idx.unsqueeze(dim=-1).repeat(1, 1, out_x.size(2)))
            out_y = torch.gather(out_y, 1, top_idx.unsqueeze(dim=-1).repeat(1, 1, out_y.size(2)))
            out = torch.gather(out, 1, top_idx.unsqueeze(dim=-1).repeat(1, 1, out.size(2)))

            return torch.stack((out_x, out_y), dim=3), self.sm(out[:, :, -1])
        else:
            return torch.stack((out_x, out_y), dim=3), out[:, :, -1]


# Multimodal Regression | DecLSTM
class MOCAST_6(nn.Module):
    def __init__(self, in_ch, out_frames, modes, batch_size=32, train=True):
        super().__init__()
        self.batch_size = batch_size
        self.modes = modes
        self.out_frames = out_frames
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_ch, self.resnet.conv1.out_channels, kernel_size=self.resnet.conv1.kernel_size,
                                      stride=self.resnet.conv1.stride, padding=self.resnet.conv1.padding, bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=256)

        self.enc_lstm = nn.LSTM(5, 32, batch_first=True)

        self.final_fc1 = nn.Linear(in_features=288, out_features=128)
        self.l_relu = nn.ReLU()

        self.dec_lstm = nn.LSTMCell(128, 128)

        self.final_fc2 = nn.Linear(in_features=128 * out_frames, out_features=((out_frames + 5) * 2 + 1) * self.modes)

        if train:
            self.sm = None
        else:
            self.sm = nn.Softmax(dim=1)

    def forward(self, x, device, state=None):
        enc_h_s = torch.zeros(1, x.size(0), 32).to(device)
        enc_c_s = torch.zeros(1, x.size(0), 32).to(device)

        # dec_lstm_x = torch.zeros(x.size(0), self.out_frames, 256).to(device)
        dec_h_s = torch.zeros(x.size(0), 128).to(device)
        dec_c_s = torch.zeros(x.size(0), 128).to(device)

        out, _ = self.enc_lstm(state, (enc_h_s, enc_c_s))
        out = torch.cat((self.resnet(x), out[:, -1, :]), dim=1)

        # out = torch.cat(self.modes * [out], dim=1).view(-1, self.modes, 128)

        out = self.final_fc1(out)
        out = self.l_relu(out)

        dec_out = []
        for i in range(self.out_frames):
            dec_h_s, dec_c_s = self.dec_lstm(out, (dec_h_s, dec_c_s))
            out = dec_h_s
            dec_out.append(dec_h_s)

        out = torch.stack(dec_out, dim=2)
        out = torch.flatten(out, start_dim=1)

        out = self.final_fc2(out).view(x.size(0), self.modes, -1)

        out_cord = out[:, :, :-1].reshape((x.size(0), self.modes, -1, 2))

        if self.sm:
            return out_cord[:, :, 5:, :], self.sm(out[:, :, -1])
        else:
            return out_cord, out[:, :, -1]
