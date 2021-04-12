import sys
import os
import torch
import time
import json
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from model import MOCAST_4
from nusc_dataloader import NuScenes_HDF
from render_prediction import render_map, render_trajectories
from utils import eval_metrics, clone_model_param, reset_param_data, find_closest_traj

sys.path.append('../datasets/nuScenes/nuscenes-devkit/python-sdk')

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.prediction.helper import convert_local_coords_to_global

NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes/'


def forward_mm(data, model, device, criterion, test_opt=False):
    inputs = data["image"].to(device)

    agent_seq_len = torch.sum(data["mask_past"], dim=1).to(device)
    history_window = torch.flip(data["agent_past"], [1]).to(device)
    history_mask = torch.flip(data['mask_past'], [1]).to(device)

    targets = data["agent_future"].to(device)
    target_mask = data['mask_future'].to(device)
    if model.dec:
        targets = torch.cat((history_window, targets), dim=1)
        target_mask = torch.cat((history_mask, target_mask), dim=1)

    if test_opt:
        with torch.no_grad():
            org_model = clone_model_param(model.dec_fc2)

        model.train()
        torch.set_grad_enabled(True)
        with torch.enable_grad():
            loss_ = test_time_opt(data, model, device)
        model.eval()
        torch.set_grad_enabled(False)

    with torch.no_grad():
        outputs, scores = model(inputs, device, data["agent_state"].to(device), agent_seq_len, out_type=2, eval=True)
        labels = find_closest_traj(outputs, targets, target_mask)
        loss_reg = criterion[0](outputs[torch.arange(outputs.size(0)), labels, :, :], targets)
        loss_cls = criterion[1](scores, labels)
        loss = loss_reg + loss_cls
        # not all the output steps are valid, apply mask to ignore invalid ones
        loss = loss * torch.unsqueeze(target_mask, 2)
        loss = loss.mean()

    if test_opt:
        reset_param_data(model.dec_fc2, org_model)

    return outputs, model.sm(scores), loss.detach().cpu().numpy()


def test_time_opt(data, fmodel, device):
    mse_loss = nn.MSELoss(reduction='none')
    optim = torch.optim.SGD(model.dec_fc2.parameters(), lr=1e-2)
    agent_seq_len = torch.sum(data["mask_past"], dim=1).to(device)
    weight_mask = torch.exp(torch.div(torch.arange(-6, 1, dtype=torch.float32), 4)).unsqueeze(0).repeat(
        len(data['token']), 1)

    target = torch.flip(data["agent_past"], [1]).to(device)
    history_mask = (torch.flip(data['mask_past'], [1]) * weight_mask).to(device)
    for epoch in range(5):
        output, _ = fmodel(data["image"].to(device), device, data["agent_state"].to(device), agent_seq_len, out_type=0)
        loss = mse_loss(output, target.unsqueeze(1).repeat(1, 10, 1, 1))
        loss = loss * history_mask.unsqueeze(1).unsqueeze(3)
        loss = loss.mean()
        print("Loss: {:.4f}".format(loss.detach().item()))
        optim.zero_grad()
        loss.backward()
        optim.step()

    return loss.detach()


def dump_predictions(pred_out, scores, token, helper):
    instance, sample = token.split('_')
    annotation = helper.get_sample_annotation(instance, sample)
    for i in range(pred_out.shape[0]):
        pred_out[i, :, :] = convert_local_coords_to_global(pred_out[i, :, :], annotation['translation'],
                                                           annotation['rotation'])
    if dec_type:
        pred_class = Prediction(instance, sample, pred_out[:, 7:, :], scores)
    else:
        pred_class = Prediction(instance, sample, pred_out, scores)
    return pred_class.serialize()


def evaluate(model, val_dl, device, criterion, test_opt=False):
    val_out_ = []
    val_scores_ = []
    val_tokens_ = []
    val_losses_ = []

    model.eval()
    torch.set_grad_enabled(False)

    progress_bar = tqdm(val_dl)
    for data in progress_bar:
        outputs, scores, val_loss = forward_mm(data, model, device, criterion, test_opt=test_opt)

        val_out_.extend(outputs.cpu().numpy())
        val_scores_.extend(scores.cpu().numpy())
        val_tokens_.extend(data["token"])
        val_losses_.append(val_loss)

    return val_out_, val_scores_, val_tokens_, val_losses_


if __name__ == '__main__':
    torch.cuda.empty_cache()
    model_out_dir_root = '/scratch/rodney/models/nusc_sota'
    model_out_dir = model_out_dir_root + '/MOCAST_4_03_17_2021_12_10_37'
    model_path = model_out_dir + "/Epoch_15_03_17_2021_15_13_50.pth"
    ds_type = 'v1.0-trainval'
    # ds_type = 'v1.0-mini'

    in_ch = 3
    out_pts = 12
    poly_deg = 5
    num_modes = 10
    batch_size = 16
    dec_type = 'ortho'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    val_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-' + ds_type + '-val.h5', transform)

    nuscenes = NuScenes(ds_type, dataroot=NUSCENES_DATASET)
    pred_helper = PredictHelper(nuscenes)

    val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MOCAST_4(in_ch, out_pts, poly_deg, num_modes, dec=dec_type).to(device)

    print("Loading model ", model_path)
    model.load_state_dict(torch.load(model_path))

    criterion_reg = nn.MSELoss(reduction="none")
    criterion_cls = nn.CrossEntropyLoss()

    val_out, val_scores, val_tokens, val_losses = evaluate(model, val_dl, device, [criterion_reg, criterion_cls])

    print("Avg val loss: {:.4f}".format(np.mean(val_losses)))

    val_ds.close_hf()

    model_preds = []
    for output, score, token in zip(val_out, val_scores, val_tokens):
        model_preds.append(dump_predictions(output, score, token, pred_helper))

    json.dump(model_preds, open(os.path.join(model_out_dir, 'mocast4_preds.json'), "w"))

    '''############################ Quantitative ###########################################'''
    config = load_prediction_config(pred_helper, '../config/eval_metric_config_1.json')
    print("[Eval] {} metrics".format(model.__class__.__name__ ))
    eval_metrics(model_out_dir + '/mocast4_preds.json', pred_helper, config, model_out_dir + '/mocast4_metrics.json')
    '''############################ Qualitative ###########################################'''

    exit()

    for i in np.random.randint(0, len(val_out), 20):
        img = render_map(pred_helper, val_tokens[i])
        gt_cord = render_trajectories(pred_helper, val_tokens[i])
        gt_n = gt_cord.shape[0]
        fig, ax = plt.subplots(1, 1)
        ax.grid(b=None)
        ax.imshow(img)
        ax.plot(gt_cord[:gt_n - 12, 0],
                gt_cord[:gt_n - 12, 1],
                'w--^',
                linewidth=3,
                markersize=2,
                zorder=650,
                path_effects=[pe.Stroke(linewidth=4, foreground='g'), pe.Normal()])
        ax.plot(gt_cord[-12:, 0],
                gt_cord[-12:, 1],
                'w--o',
                linewidth=3,
                markersize=2,
                zorder=650,
                path_effects=[pe.Stroke(linewidth=4, foreground='g'), pe.Normal()])

        top_3 = np.argsort(val_scores[i])[-1:-4:-1]
        for ind in top_3:
            pred_cord = render_trajectories(pred_helper, val_tokens[i], val_out[i][ind])

            ax.plot(pred_cord[:7, 0],
                    pred_cord[:7, 1],
                    'w--^',
                    linewidth=3,
                    markersize=2,
                    zorder=650,
                    path_effects=[pe.Stroke(linewidth=4, foreground='b'), pe.Normal()])
            ax.plot(pred_cord[7:, 0],
                    pred_cord[7:, 1],
                    'w--o',
                    linewidth=3,
                    markersize=2,
                    zorder=650,
                    path_effects=[pe.Stroke(linewidth=4, foreground='b'), pe.Normal()])
            plt.text(pred_cord[-1][0] + 10, pred_cord[-1][1], "{:0.2f}".format(val_scores[i][ind]))
    #
    plt.show()
