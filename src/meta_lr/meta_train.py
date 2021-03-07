import os
import sys
import torch
import time
import cv2
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import dump_model_graph
from utils import save_model_dict

from meta_lr.meta_model import MOCAST4_METALR
from nusc_dataloader import NuScenes_HDF


# Prints whole tensor for debug
# torch.set_printoptions(profile="full")

def clone_model_param(model):
    new_param = {}
    for name, params in model.named_parameters():
        new_param[name] = params.clone()

    return new_param


def reset_param_data(model, new_params):
    for name, params in model.named_parameters():
        params.data.copy_(new_params[name].data)


def get_batch_sample(data, ind):
    sample = {}
    for k, v in data.items():
        sample[k] = v[ind]
        if k != 'token':
            sample[k] = sample.get(k).unsqueeze(0)
    return sample


# Returns closest mode to GT
def find_closest_traj(pred, gt):
    gt = np.expand_dims(gt, 1)
    ade = np.sum((gt - pred) ** 2, axis=-1) ** 0.5
    ade = np.mean(ade, axis=-1)
    labels = torch.from_numpy(np.argmin(ade, axis=-1))
    return labels


def forward_mm(data, model, device, criterion):
    # Raster input
    inputs = data["image"].to(device)

    # Valid history length
    agent_seq_len = torch.sum(data["mask_past"], dim=1).to(device)
    # History points
    history_window = torch.flip(data["agent_past"], [1]).to(device)
    # Mask to indicate valid history points
    history_mask = torch.flip(data['mask_past'], [1]).to(device)

    # Future points
    targets = data["agent_future"].to(device)
    target_mask = data['mask_future'].to(device)

    # Inner GD Loop
    for l in range(5):
        # Train loss on history
        outputs, scores = model(inputs, device, data["agent_state"].to(device), agent_seq_len, hist=True)
        labels_h = find_closest_traj(outputs.cpu().detach().numpy(), history_window.cpu().detach().numpy()).to(device)
        # Regression + Classification loss
        loss_reg = criterion[0](outputs[torch.arange(outputs.size(0)), labels_h, :, :], history_window)
        loss_cls = criterion[1](scores, labels_h)
        loss = loss_reg + loss_cls
        # not all the output steps are valid, so apply mask to ignore invalid ones
        loss = loss * torch.unsqueeze(history_mask, 2)
        loss = loss.mean()
        print('Inner loop Loss: {:.4f}'.format(loss))

        # Gradients wrt model params
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        with torch.no_grad():
            for i, p in enumerate(model.parameters()):
                new_p = p - 1e-4 * grads[i]
                p.data.copy_(new_p)

    # Evaluate loss on targets
    outputs, scores = model(inputs, device, data["agent_state"].to(device), agent_seq_len, hist=False)
    labels = find_closest_traj(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy()).to(device)
    loss_reg = criterion[0](outputs[torch.arange(outputs.size(0)), labels, :, :], targets)
    loss_cls = criterion[1](scores, labels)
    loss = loss_reg + loss_cls
    loss = loss * torch.unsqueeze(target_mask, 2)
    loss = loss.mean()

    return loss


def train(model, train_ds, device, criterion, model_out_dir_):
    # ==== TRAIN LOOP
    epochs = 10000
    batch_size = 8
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # torch.autograd.set_detect_anomaly(True)

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=epochs,
                                                steps_per_epoch=1)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
    tr_it = iter(train_dl)

    epoch_mean_loss = []
    # Outer loop
    for epoch in range(epochs):
        try:
            data = next(tr_it)
        except StopIteration:
            train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
            tr_it = iter(train_dl)
            data = next(tr_it)

        model.train()
        torch.set_grad_enabled(True)

        batch_loss = []
        # Keep a copy of the initial model params 'theta'
        init_params = clone_model_param(model)

        # Inner loop
        for b in range(len(data['token'])):
            print("Sample {}:".format(b))
            sample = get_batch_sample(data, b)

            # model.zero_grad()

            with torch.enable_grad():
                # Gradient descent on history and evaluated on future to get updated params phi
                batch_loss.append(forward_mm(sample, model, device, criterion))

            with torch.no_grad():
                # Reset model params to initial 'theta'
                reset_param_data(model, init_params)

        # Sum all the losses 
        loss = sum(batch_loss)

        # dump_model_graph(loss, model)
        optimizer.zero_grad()
        # Backward pass
        loss.backward()

        # nn.utils.clip_grad_value_(model.parameters(), 0.5)
        # Update theta
        optimizer.step()
        sched.step()

        epoch_mean_loss.append(loss.cpu().detach().item())
        print("Epoch: {}/{} Outer loss: {} loss(avg): {}".format(epoch + 1, epochs, loss.cpu().detach().item(),
                                                                 np.mean(epoch_mean_loss)))

        if (epoch + 1) % 1000 == 0:
            save_model_dict(model, model_out_dir_, epoch + 1)

    return epoch_mean_loss


if __name__ == '__main__':
    torch.cuda.empty_cache()
    in_ch = 3
    out_pts = 12
    poly_deg = 5
    num_modes = 10
    model_out_dir_root = '/scratch/rodney/models/nuScenes'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    train_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-trainval-train.h5', transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MOCAST4_METALR(in_ch, out_pts, poly_deg, num_modes).to(device)

    model_out_dir = os.path.join(model_out_dir_root,
                                 model.__class__.__name__ + time.strftime("_%m_%d_%Y_%H_%M_%S", time.localtime()))
    # Create model out directory
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    # if torch.cuda.device_count() > 1:
    #     # print("Using", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids=[0, 1])

    criterion_reg = nn.MSELoss(reduction="none")
    criterion_cls = nn.CrossEntropyLoss()

    # Training
    losses_train = train(model, train_ds, device, [criterion_reg, criterion_cls], model_out_dir)

    train_ds.close_hf()

    plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
    plt.legend()
    plt.savefig(model_out_dir + '/loss.png')
    plt.show()
