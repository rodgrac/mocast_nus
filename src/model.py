from torchvision.models.resnet import resnet50
import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as f


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
    def __init__(self, in_ch, out_frames, degree, modes, batch_size=32, train=True):
        super().__init__()
        self.batch_size = batch_size
        self.degree = degree
        self.modes = modes
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_ch, self.resnet.conv1.out_channels, kernel_size=self.resnet.conv1.kernel_size,
                                      stride=self.resnet.conv1.stride, padding=self.resnet.conv1.padding, bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=96)

        print("Num modes: ", self.modes)

        self.enc_lstm = nn.LSTM(3, 32, batch_first=True)

        self.final_fc1 = nn.Linear(in_features=128, out_features=256)
        self.l_relu = nn.ReLU()

        self.final_fc2 = nn.Linear(in_features=256, out_features=((degree + 1) * 2 + 1) * self.modes)

        if train:
            t_n = np.array(range(-2, out_frames + 1), dtype=np.float32) / out_frames
            self.sm = None
        else:
            t_n = np.array(range(1, out_frames + 1), dtype=np.float32) / out_frames
            self.sm = nn.Softmax(dim=1)
        self.tmat = torch.from_numpy(np.vstack([t_n ** i for i in range(degree, -1, -1)]))

    def forward(self, x, device, state=None, state_len=None):
        enc_h_s = torch.zeros(1, x.size(0), 32).to(device)
        enc_c_s = torch.zeros(1, x.size(0), 32).to(device)

        state_len = torch.clamp(state_len, min=1)
        state = torch.nn.utils.rnn.pack_padded_sequence(state, state_len.to('cpu'), batch_first=True, enforce_sorted=False).float()
        out, _ = self.enc_lstm(state, (enc_h_s, enc_c_s))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = torch.cat((self.resnet(x), out[torch.arange(out.size(0)), state_len - 1, :]), dim=1)

        # out = torch.cat(self.modes * [out], dim=1).view(-1, self.modes, 128)

        out = self.final_fc1(out)
        out = self.l_relu(out)

        out = self.final_fc2(out).view(x.size(0), self.modes, -1)

        self.tmat = self.tmat.to(device)
        out_x = torch.matmul(out[:, :, :self.degree + 1], self.tmat)
        out_y = torch.matmul(out[:, :, self.degree + 1:-1], self.tmat)

        if self.sm:
            (_, top_idx) = torch.topk(out[:, :, -1], 3)
            out_x = torch.gather(out_x, 1, top_idx.unsqueeze(dim=-1).repeat(1, 1, out_x.size(2)))
            out_y = torch.gather(out_y, 1, top_idx.unsqueeze(dim=-1).repeat(1, 1, out_y.size(2)))
            out = torch.gather(out, 1, top_idx.unsqueeze(dim=-1).repeat(1, 1, out.size(2)))

            return torch.stack((out_x, out_y), dim=3), self.sm(out[:, :, -1])
        else:
            return torch.stack((out_x, out_y), dim=3), out[:, :, -1]


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
        self.resnet.fc = nn.Linear(in_features=2048, out_features=256)

        print("Num modes: ", self.modes)

        self.enc_lstm = nn.LSTM(5, 32, batch_first=True)

        self.final_fc1 = nn.Linear(in_features=288, out_features=128)
        self.l_relu = nn.ReLU()

        self.dec_lstm = nn.LSTMCell(128, 128)

        self.final_fc2 = nn.Linear(in_features=128 * out_frames, out_features=((degree + 1) * 2 + 1) * self.modes)

        if train:
            t_n = np.array(range(-4, out_frames + 1), dtype=np.float32) / out_frames
            self.sm = None
        else:
            t_n = np.array(range(1, out_frames + 1), dtype=np.float32) / out_frames
            self.sm = nn.Softmax(dim=1)
        self.tmat = torch.from_numpy(np.vstack([t_n ** i for i in range(degree, -1, -1)]))

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

        self.tmat = self.tmat.to(device)
        out_x = torch.matmul(out[:, :, :self.degree + 1], self.tmat)
        out_y = torch.matmul(out[:, :, self.degree + 1:-1], self.tmat)

        if self.sm:
            (_, top_idx) = torch.topk(out[:, :, -1], 3)
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
