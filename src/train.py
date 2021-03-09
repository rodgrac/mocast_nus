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


def train(model, train_dataloader, device, criterion):
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

    return epoch_mean_loss


torch.cuda.empty_cache()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

train_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-trainval-train.h5', transform)

train_dl = DataLoader(train_ds, shuffle=True, batch_size=16, num_workers=16)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model = MOCAST_4(3, 12, 5, 10, 16, dec='polytr').to(device)

criterion_reg = nn.MSELoss(reduction="none")
criterion_cls = nn.CrossEntropyLoss()

# Training
losses_train = train(model, train_dl, device, [criterion_reg, criterion_cls])

time_string = time.strftime("_%m_%d_%Y_%H_%M_%S", time.localtime())
torch.save(model.state_dict(), '../models/' + model.__class__.__name__ + time_string + '.pth')
print("Saved model as ../models/" + model.__class__.__name__ + time_string + '.pth')

train_ds.close_hf()

plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
plt.legend()
plt.show()
