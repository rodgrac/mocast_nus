import os
import sys
import torch
import time
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt

from meta_lr.meta_model import MOCAST4_METALR
from nusc_dataloader import NuScenes_HDF


# Prints whole tensor for debug
# torch.set_printoptions(profile="full")

def clone_model_param(model):
    new_param = {}
    for name, params in model.named_parameters():
        new_param[name] = params.clone()

    return new_param

def update_param_data(model, new_params):
    for name, params in model.named_parameters():
        params.data.copy_(new_params[name])


def find_closest_traj(pred, gt):
    gt = np.expand_dims(gt, 1)
    ade = np.sum((gt - pred) ** 2, axis=-1) ** 0.5
    ade = np.mean(ade, axis=-1)
    labels = torch.from_numpy(np.argmin(ade, axis=-1))
    return labels


def forward_mm(data, model, device, criterion):
    inputs = data["image"].to(device)

    agent_seq_len = torch.sum(data["mask_past"], dim=1).to(device)
    history_window = torch.flip(data["agent_past"], [1]).to(device)
    history_mask = torch.flip(data['mask_past'], [1]).to(device)

    targets = data["agent_future"].to(device)
    target_mask = data['mask_future'].to(device)

    # Inner Loop
    for l in range(10):
        outputs_h, scores_h = model(inputs, device, data["agent_state"].to(device), agent_seq_len, hist=True)
        labels_h = find_closest_traj(outputs_h.cpu().detach().numpy(), history_window.cpu().detach().numpy()).to(device)
        loss_reg_h = criterion[0](outputs_h[torch.arange(outputs_h.size(0)), labels_h, :, :], history_window)
        loss_cls_h = criterion[1](scores_h, labels_h)
        loss_h = loss_reg_h + loss_cls_h
        # not all the output steps are valid, but we can filter them out from the loss using availabilities
        loss_h = loss_h * torch.unsqueeze(history_mask, 2)
        loss_h = loss_h.mean()

        grads = torch.autograd.grad(loss_h, model.parameters(), create_graph=True)

        with torch.no_grad():
            for i, p in enumerate(model.parameters()):
                new_p = p - 3e-4 * grads[i]
                p.copy_(new_p)
        print('Inner loop Loss: {:.4f}'.format(loss_h))

    outputs, scores = model(inputs, device, data["agent_state"].to(device), agent_seq_len, hist=False)
    labels = find_closest_traj(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy()).to(device)
    loss_reg = criterion[0](outputs[torch.arange(outputs.size(0)), labels, :, :], targets)
    loss_cls = criterion[1](scores, labels)
    loss = loss_reg + loss_cls
    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss = loss * torch.unsqueeze(target_mask, 2)
    loss = loss.mean()
    return loss, outputs


def train(model, train_ds, device, criterion):
    # ==== TRAIN LOOP
    epochs = 10000
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=epochs,
                                                steps_per_epoch=1)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=4, num_workers=4)
    tr_it = iter(train_dl)

    epoch_mean_loss = []
    for epoch in range(epochs):
        try:
            data = next(tr_it)
        except StopIteration:
            train_dl = DataLoader(train_ds, shuffle=True, batch_size=4, num_workers=4)
            tr_it = iter(train_dl)
            data = next(tr_it)

        model.train()
        torch.set_grad_enabled(True)

        optimizer.zero_grad()

        # init_params = clone_model_param(model)

        with torch.enable_grad():
            loss, _ = forward_mm(data, model, device, criterion)

        # update_param_data(model, init_params)

        # Backward pass
        loss.backward()

        # nn.utils.clip_grad_value_(model.parameters(), 0.1)

        optimizer.step()
        sched.step()

        epoch_mean_loss.append(loss.cpu().detach().item())
        print("Epoch: {}/{} Outer loss: {} loss(avg): {}".format(epoch + 1, epochs, loss.cpu().detach().item(),
                                                                 np.mean(epoch_mean_loss)))

    return epoch_mean_loss


torch.cuda.empty_cache()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])

train_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-trainval-train.h5', transform)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MOCAST4_METALR(3, 12, 5, 10, 16).to(device)

# if torch.cuda.device_count() > 1:
#     # print("Using", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)


criterion_reg = nn.MSELoss(reduction="none")
criterion_cls = nn.CrossEntropyLoss()

# Training
losses_train = train(model, train_ds, device, [criterion_reg, criterion_cls])

time_string = time.strftime("_%m_%d_%Y_%H_%M_%S", time.localtime())
torch.save(model.state_dict(), '../../models/' + model.__class__.__name__ + time_string + '.pth')
print("Saved model as ../models/" + model.__class__.__name__ + time_string + '.pth')

train_ds.close_hf()

plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
plt.legend()
plt.show()
