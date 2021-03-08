import os
import sys
import torch
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt

from model import MOCAST_4
from nusc_dataloader import NuScenes_HDF
from utils import save_model_dict

# Prints whole tensor for debug
# torch.set_printoptions(profile="full")


# Returns closest mode to GT
def find_closest_traj(pred, gt):
    ade = torch.sum((gt.unsqueeze(1) - pred) ** 2, dim=-1) ** 0.5
    ade = torch.mean(ade, dim=-1)
    return torch.argmin(ade, dim=-1)


def forward_mm(data, model, device, criterion):
    inputs = data["image"].to(device)

    agent_seq_len = torch.sum(data["mask_past"], dim=1).to(device)
    history_window = torch.flip(data["agent_past"], [1]).to(device)
    history_mask = torch.flip(data['mask_past'], [1]).to(device)

    targets = data["agent_future"].to(device)
    targets = torch.cat((history_window, targets), dim=1)
    target_mask = torch.cat((history_mask, data['mask_future'].to(device)), dim=1)
    # Forward pass
    outputs, scores = model(inputs, device, data["agent_state"].to(device), agent_seq_len)

    labels = find_closest_traj(outputs, targets)

    loss_reg = criterion[0](outputs[torch.arange(outputs.size(0)), labels, :, :], targets)
    loss_cls = criterion[1](scores, labels)
    loss = loss_reg + loss_cls
    # not all the output steps are valid, but we can filter them out from the loss using availabilities

    loss = loss * torch.unsqueeze(target_mask, 2)

    loss = loss.mean()
    return loss, outputs


def train(model, train_dataloader, device, criterion, model_out_dir_):
    # ==== TRAIN LOOP
    epochs = 15
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=epochs,
                                                steps_per_epoch=len(train_dataloader))
    epoch_mean_loss = []
    for epoch in range(epochs):
        losses_train = []
        progress_bar = tqdm(train_dataloader)
        for data in progress_bar:
            model.train()
            torch.set_grad_enabled(True)

            loss, _ = forward_mm(data, model, device, criterion)

            optimizer.zero_grad()
            # Backward pass
            loss.backward()

            # nn.utils.clip_grad_value_(model.parameters(), 0.1)

            optimizer.step()
            sched.step()

            losses_train.append(loss.cpu().detach().item())
            progress_bar.set_description(
                f"Epoch: {epoch + 1}/{epochs} loss: {loss.cpu().detach().item()} loss(avg): {np.mean(losses_train)}")

        epoch_mean_loss.append(np.mean(losses_train))

        if (epoch + 1) % 5 == 0:
            save_model_dict(model, model_out_dir_, epoch + 1)

    return epoch_mean_loss


if __name__ == '__main__':
    torch.cuda.empty_cache()
    in_ch = 3
    out_pts = 12
    poly_deg = 5
    num_modes = 10
    batch_size = 16
    model_out_dir_root = '/scratch/rodney/models/nuScenes'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    train_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-trainval-train.h5', transform)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MOCAST_4(in_ch, out_pts, poly_deg, num_modes, dec='polytr').to(device)

    model_out_dir = os.path.join(model_out_dir_root,
                                     model.__class__.__name__ + time.strftime("_%m_%d_%Y_%H_%M_%S", time.localtime()))

    # Create model out directory
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    criterion_reg = nn.MSELoss(reduction="none")
    criterion_cls = nn.CrossEntropyLoss()

    # Training
    losses_train = train(model, train_dl, device, [criterion_reg, criterion_cls], model_out_dir)

    time_string = time.strftime("_%m_%d_%Y_%H_%M_%S", time.localtime())
    torch.save(model.state_dict(), '../models/' + model.__class__.__name__ + time_string + '.pth')
    print("Saved model as ../models/" + model.__class__.__name__ + time_string + '.pth')

    train_ds.close_hf()

    plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
    plt.legend()
    plt.show()
