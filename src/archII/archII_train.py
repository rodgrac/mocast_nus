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

from archII.archII_model import STSE_MHA
from nusc_dataloader import NuScenes_HDF
from utils import save_model_dict, find_closest_traj
from archII.archII_eval import evaluate


# Prints whole tensor for debug
# torch.set_printoptions(profile="full")

def forward_mm(data, model, device, criterion):
    inputs = data["image"].to(device)

    ta_seq_len = torch.sum(data["ta_mask_past"][:, :5], dim=1).to(device)
    ta_past = torch.flip(data["ta_past"][:, :hist_pts+1, :].to(device), [1])
    ta_mask_past = torch.flip(data['ta_mask_past'][:, :hist_pts+1].to(device), [1])

    ta_coord = data["ta_future"].to(device)
    ta_mask = data['ta_mask_future'].to(device)
    ta_coord = torch.cat((ta_past, ta_coord), dim=1)
    ta_mask = torch.cat((ta_mask_past, ta_mask), dim=1)

    # Forward pass
    outputs, scores = model(inputs, device, data["ta_state_hist"].to(device), ta_seq_len, data["sa_state_hist"].to(device),
                            data['sa_sthist_len'].to(device), data['sa_grid_pos'].to(device), out_type=2)

    labels = find_closest_traj(outputs, ta_coord, ta_mask)

    loss_reg = criterion[0](outputs[torch.arange(outputs.size(0)), labels, :, :], ta_coord)
    loss_cls = criterion[1](scores, labels)
    loss = loss_reg + loss_cls

    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss = loss * torch.unsqueeze(ta_mask, 2)

    loss = loss.mean()
    return loss, outputs


def train(model, train_dataloader, device, criterion, optim):
    # ==== TRAIN LOOP
    log_fr = 100

    model.train()
    torch.set_grad_enabled(True)

    epoch_train_loss = []
    progress_bar = tqdm(train_dataloader)
    for it, data in enumerate(progress_bar):
        loss, _ = forward_mm(data, model, device, criterion)

        optim[0].zero_grad()
        # Backward pass
        loss.backward()

        # nn.utils.clip_grad_value_(model.parameters(), 0.1)

        optim[0].step()
        optim[1].step()

        epoch_train_loss.append(loss.detach().cpu().numpy())

        if it % log_fr == 0:
            print("Epoch: {}/{} It: {}:Epoch train loss(avg): {:.4f}".format(epoch + 1, epochs, it, np.mean(epoch_train_loss)))

    return epoch_train_loss


if __name__ == '__main__':
    torch.cuda.empty_cache()
    in_ch = 3
    hist_pts = 12
    fut_pts = 12
    poly_deg = 7
    num_modes = 10
    epochs = 15
    batch_size = 16

    model_out_dir_root = '/scratch/rodney/models/nuScenes'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    train_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-archII-v1.0-trainval-train.h5', transform)
    val_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-archII-v1.0-trainval-val.h5', transform)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=batch_size)
    val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=batch_size)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = STSE_MHA(in_ch, hist_pts, fut_pts, poly_deg, num_modes, dec='ortho').to(device)

    model_out_dir = os.path.join(model_out_dir_root,
                                 model.__class__.__name__ + time.strftime("_%m_%d_%Y_%H_%M_%S", time.localtime()))
    print("Model out directory:", model_out_dir)

    criterion_reg = nn.MSELoss(reduction="none")
    criterion_cls = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=epochs,
                                                steps_per_epoch=len(train_dl))

    train_losses = []
    val_losses = []
    # Training
    for epoch in range(epochs):
        # Training
        train_loss = train(model, train_dl, device, [criterion_reg, criterion_cls], [optimizer, sched])
        train_losses.append(np.mean(train_loss))
        if (epoch+1) > 10:
            save_model_dict(model, model_out_dir, epoch + 1)
        # Validation
        _, _, _, _, val_loss = evaluate(model, val_dl, device, [criterion_reg, criterion_cls])
        print("Epoch {}/{} Val loss: {:.4f}".format(epoch + 1, epochs, np.mean(val_loss)))
        val_losses.append(np.mean(val_loss))

    train_ds.close_hf()
    val_ds.close_hf()

    plt.plot(np.arange(len(train_losses)), train_losses, 'b', label="train loss")
    plt.plot(np.arange(len(val_losses)), val_losses, 'r', label="val loss")
    plt.legend()
    plt.savefig(model_out_dir + '/loss.png')
    plt.show()
