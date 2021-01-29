import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import *
from process_ds import *
from utils import *


def find_closest_traj(pred, gt):
    gt = np.expand_dims(gt, 1)
    ade = np.sum((gt - pred) ** 2, axis=-1) ** 0.5
    ade = np.mean(ade, axis=-1)
    labels = torch.from_numpy(np.argmin(ade, axis=-1))
    return labels


def forward_mm(data, model, device, criterion):
    inputs = data["image"].to(device)

    targets = data["agent_future"].to(device)

    # Forward pass
    outputs, scores = model(inputs, device, data["agent_state"].to(device))

    labels = find_closest_traj(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy()).to(device)

    loss_reg = criterion[0](outputs[torch.arange(outputs.size(0)), labels, :, :], targets)
    loss_cls = criterion[1](scores, labels)
    loss = loss_reg + loss_cls
    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    # loss = loss * target_availabilities
    loss = loss.mean()
    return loss, outputs


def train(model, train_dataloader, device, criterion):
    # ==== TRAIN LOOP
    tr_it = iter(train_dataloader)
    progress_bar = tqdm(range(1000))
    losses_train = []

    optimizer = optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=1000,
                                                steps_per_epoch=len(train_dataloader))

    for _ in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)
        model.train()
        torch.set_grad_enabled(True)
        loss, _ = forward_mm(data, model, device, criterion)

        # Backward pass
        loss.backward()

        nn.utils.clip_grad_value_(model.parameters(), 0.1)

        optimizer.step()
        optimizer.zero_grad()

        sched.step()

        losses_train.append(loss.item())
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

    return losses_train


torch.cuda.empty_cache()

train_ds = load_obj('../datasets/nuScenes/processed/nuscenes-mini-p.pkl')

train_dl = DataLoader(train_ds, shuffle=True, batch_size=16, num_workers=16)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = MOCAST_4(3, 10, 5, 3, 16).to(device)

criterion_reg = nn.MSELoss(reduction="none")
criterion_cls = nn.CrossEntropyLoss()

# Training
losses_train = train(model, train_dl, device, [criterion_reg, criterion_cls])
