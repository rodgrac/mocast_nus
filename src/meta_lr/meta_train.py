import os
import sys
import torch
import time
import cv2
import argparse
import random
import higher
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append('../')

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
    ade = torch.sum((gt.unsqueeze(1) - pred) ** 2, dim=-1) ** 0.5
    ade = torch.mean(ade, dim=-1)
    return torch.argmin(ade, dim=-1)


def forward_mm(data, f_model, device, criterion, dopt, it):
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
    for _ in range(4):
        # Train loss on history
        outputs, scores = f_model(inputs, device, data["agent_state"].to(device), agent_seq_len, hist=True)
        labels = find_closest_traj(outputs, history_window)
        # Regression + Classification loss
        spt_loss_reg = criterion[0](outputs[torch.arange(outputs.size(0)), labels, :, :], history_window)
        spt_loss_cls = criterion[1](scores, labels)
        spt_loss = spt_loss_reg + spt_loss_cls
        # not all the output steps are valid, so apply mask to ignore invalid ones
        spt_loss = spt_loss * torch.unsqueeze(history_mask, 2)
        spt_loss = spt_loss.mean()
        if it % 100 == 0:
            print('Inner loop Loss: {:.4f}'.format(spt_loss.detach()))

        dopt.step(spt_loss)

    # Evaluate loss on targets
    outputs, scores = f_model(inputs, device, data["agent_state"].to(device), agent_seq_len, hist=False)
    labels = find_closest_traj(outputs, targets)
    qry_loss_reg = criterion[0](outputs[torch.arange(outputs.size(0)), labels, :, :], targets)
    qry_loss_cls = criterion[1](scores, labels)
    qry_loss = qry_loss_reg + qry_loss_cls
    qry_loss = qry_loss * torch.unsqueeze(target_mask, 2)
    qry_loss = qry_loss.mean()

    qry_loss.backward()

    return qry_loss.detach().cpu().numpy()


def train(model, train_dl, device, criterion, model_out_dir_):
    # ==== TRAIN LOOP
    epochs = 5
    log_fr = 100
    meta_optim = torch.optim.Adam(model.parameters(), 1e-4)

    # torch.autograd.set_detect_anomaly(True)

    sched = torch.optim.lr_scheduler.OneCycleLR(meta_optim, 1e-3, epochs=epochs,
                                                steps_per_epoch=len(train_dl))
    # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    epoch_mean_loss = []
    # Outer loop
    for epoch in range(epochs):
        losses_train = []
        progress_bar = tqdm(train_dl)
        for it, data in enumerate(progress_bar):
            model.train()
            torch.set_grad_enabled(True)

            query_loss = []

            inner_opt = torch.optim.SGD(model.dec_parameters, lr=1e-3)

            # Inner loop
            for b in range(len(data['token'])):
                if it % log_fr == 0:
                    print("Sample {}:".format(b))
                sample = get_batch_sample(data, b)

                # with torch.backends.cudnn.flags(enabled=False):
                with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                    # Gradient descent on history and evaluated on future to get updated params phi
                    query_loss.append(forward_mm(sample, fmodel, device, criterion, diffopt, it))

            # nn.utils.clip_grad_value_(model.parameters(), 0.5)

            meta_optim.step()
            sched.step()
            meta_optim.zero_grad()

            avg_batch_loss = sum(query_loss) / len(data['token'])

            losses_train.append(avg_batch_loss)
            if it % log_fr == 0:
                print("Epoch: {}/{} It: {}, Batch outer loss: {} loss(avg): {}".format(epoch + 1, epochs, it, avg_batch_loss,
                                                                               np.mean(losses_train)))
        epoch_mean_loss.extend(losses_train)
        # if (epoch + 1) == 0:
        save_model_dict(model, model_out_dir_, epoch + 1)

    return epoch_mean_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    parser.add_argument("--gpu", type=int, default=0, required=False)
    args = parser.parse_args()
    torch.cuda.empty_cache()
    in_ch = 3
    out_pts = 12
    poly_deg = 5
    num_modes = 10
    batch_size = args.batch_size

    model_out_dir_root = '/scratch/rodney/models/nuScenes'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    train_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-trainval-train.h5', transform)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=batch_size)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

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
    losses_train = train(model, train_dl, device, [criterion_reg, criterion_cls], model_out_dir)

    train_ds.close_hf()

    plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
    plt.legend()
    plt.savefig(model_out_dir + '/loss.png')
    plt.show()
