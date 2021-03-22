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
from utils import save_model_dict, find_closest_traj

from meta_lr.meta_model import MOCAST4_METALR
from nusc_dataloader import NuScenes_HDF
from meta_lr.meta_eval import evaluate


# Prints whole tensor for debug
# torch.set_printoptions(profile="full")

def get_batch_sample(data, ind):
    sample = {}
    for k, v in data.items():
        sample[k] = v[ind]
        if k != 'token':
            sample[k] = sample.get(k).unsqueeze(0)
    return sample


def forward_mm(data, f_model, device, criterion, dopt, in_steps, it):
    # Raster input
    inputs = data["image"].to(device)

    # Valid history length
    agent_seq_len = torch.sum(data["mask_past"], dim=1).to(device)
    # History points
    history_window = torch.flip(data["agent_past"], [1]).to(device)

    # Mask to indicate valid history points
    history_mask = (torch.flip(data['mask_past'], [1]) * torch.softmax(f_model.weight_mask.repeat(inputs.size(0), 1),
                                                                       dim=1)).to(device)

    # Future points
    targets = data["agent_future"].to(device)
    target_mask = data['mask_future'].to(device)
    # targets = torch.cat((history_window, targets), dim=1)
    # target_mask = torch.cat((history_mask, target_mask), dim=1)

    # Inner GD Loop
    for _ in range(in_steps):
        # Train loss on history
        spt_outputs, spt_scores = f_model(inputs, device, data["agent_state"].to(device), agent_seq_len, out_type=0)

        spt_loss = criterion[0](spt_outputs, history_window.unsqueeze(1).repeat(1, 10, 1, 1))
        spt_loss = torch.mean(spt_loss, dim=1)
        # not all the output steps are valid, so apply mask to ignore invalid ones
        spt_loss = spt_loss * torch.unsqueeze(history_mask, 2)
        spt_loss = spt_loss.mean()
        if it % 100 == 0:
            print('Inner loop Loss: {:.4f}'.format(spt_loss.detach()))

        dopt.step(spt_loss)

    # Evaluate loss on targets
    qry_outputs, qry_scores = f_model(inputs, device, data["agent_state"].to(device), agent_seq_len, out_type=1)

    labels = find_closest_traj(qry_outputs, targets)
    loss_reg = criterion[0](qry_outputs[torch.arange(qry_outputs.size(0)), labels, :, :], targets)
    loss_cls = criterion[1](qry_scores, labels)
    qry_loss = loss_reg + loss_cls
    qry_loss = qry_loss * torch.unsqueeze(target_mask, 2)
    qry_loss = qry_loss.mean()

    qry_loss.backward()

    return qry_loss.detach().cpu().numpy()


def train(model, train_dl, device, criterion, outer_optim, inner_steps):
    # ==== TRAIN LOOP
    log_fr = 100
    # torch.autograd.set_detect_anomaly(True)

    model.train()
    torch.set_grad_enabled(True)

    # Outer loop
    epoch_train_loss = []
    progress_bar = tqdm(train_dl)
    for it, data in enumerate(progress_bar):
        query_loss = []

        inner_opt = torch.optim.SGD(model.inner_parameters, lr=1e-3)
        outer_optim[0].zero_grad()
        # Inner loop
        for b in range(len(data['token'])):
            if it % log_fr == 0:
                print("Sample {}:".format(b))
            sample = get_batch_sample(data, b)

            # with torch.backends.cudnn.flags(enabled=False):
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                # Gradient descent on history and evaluated on future to get updated params phi
                query_loss.append(forward_mm(sample, fmodel, device, criterion, diffopt, inner_steps, it))

        # nn.utils.clip_grad_value_(model.parameters(), 0.5)

        outer_optim[0].step()
        # outer_optim[1].step()

        avg_batch_loss = sum(query_loss) / len(data['token'])

        epoch_train_loss.append(avg_batch_loss)
        if it % log_fr == 0:
            print("Epoch: {}/{} Sample: {}, Sample outer loss: {:.4f}, Epoch loss(avg): {:.4f}"
                  .format(epoch + 1, epochs, it, avg_batch_loss, np.mean(epoch_train_loss)))

    return epoch_train_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--gpu", type=int, default=2, required=False)
    args = parser.parse_args()

    torch.cuda.empty_cache()
    in_ch = 3
    out_pts = 12
    poly_deg = 5
    num_modes = 10
    epochs = 5
    inner_steps_ = 5
    batch_size = args.batch_size

    model_out_dir_root = '/scratch/rodney/models/nuScenes'
    pretrained_path = '/MOCAST_4_03_17_2021_12_10_37/Epoch_15_03_17_2021_15_13_50.pth'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    train_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-trainval-train.h5', transform)
    val_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-mini-val.h5', transform)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=batch_size)
    val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=batch_size)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    model = MOCAST4_METALR(in_ch, out_pts, poly_deg, num_modes).to(device)

    print('Loading pretrained model:', model_out_dir_root + pretrained_path)
    model.load_state_dict(torch.load(model_out_dir_root + pretrained_path, map_location=device))

    # Freezing the encoder layers
    for param in list(model.parameters())[:-6]:
        param.requires_grad = False

    model.register_parameter(name='weight_mask', param=nn.Parameter(torch.ones((1, 7), requires_grad=True)))

    model.inner_parameters.append({'params': model.weight_mask})
    model.outer_parameters.append({'params': model.weight_mask})

    model_out_dir = os.path.join(model_out_dir_root,
                                 model.__class__.__name__ + time.strftime("_%m_%d_%Y_%H_%M_%S", time.localtime()))

    # if torch.cuda.device_count() > 1:
    #     # print("Using", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids=[0, 1])

    criterion_reg = nn.MSELoss(reduction="none")
    criterion_cls = nn.CrossEntropyLoss()

    meta_optim = torch.optim.Adam(model.outer_parameters, 1e-4)
    # sched = torch.optim.lr_scheduler.OneCycleLR(meta_optim, 1e-3, epochs=epochs,
    #                                             steps_per_epoch=len(train_dl))

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # Training
        train_loss = train(model, train_dl, device, [criterion_reg, criterion_cls], [meta_optim], inner_steps_)
        train_losses.append(np.mean(train_loss))
        save_model_dict(model, model_out_dir, epoch + 1)
        # Validation
        _, _, _, val_loss = evaluate(model, val_dl, device, [criterion_reg, criterion_cls], inner_steps_)
        print("Epoch {}/{} VAL LOSS: {:.4f}".format(epoch + 1, epochs, np.mean(val_loss)))
        val_losses.append(np.mean(val_loss))

    train_ds.close_hf()
    val_ds.close_hf()

    plt.plot(np.arange(len(train_losses)), train_losses, 'b', label="train loss")
    plt.plot(np.arange(len(val_losses)), val_losses, 'r', label="val loss")
    plt.legend()
    plt.savefig(model_out_dir + '/loss.png')
    plt.show()
