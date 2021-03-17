import sys
import os
import torch
import time
import json
import higher
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from meta_lr.meta_model import MOCAST4_METALR
from nusc_dataloader import NuScenes_HDF
from render_prediction import render_map, render_trajectories
from utils import eval_metrics, find_closest_traj

sys.path.append('../datasets/nuScenes/nuscenes-devkit/python-sdk')

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.prediction.helper import convert_local_coords_to_global

NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes/'


def forward_mm(data, f_model, device, criterion, dopt, in_steps):
    inputs = data["image"].to(device)

    agent_seq_len = torch.sum(data["mask_past"], dim=1).to(device)
    history_window = torch.flip(data["agent_past"], [1]).to(device)
    history_mask = torch.flip(data['mask_past'], [1]).to(device)

    # Future points
    targets = data["agent_future"].to(device)
    target_mask = data['mask_future'].to(device)

    # Inner Loop
    for _ in range(in_steps):
        # Train loss on history
        spt_outputs, spt_scores = f_model(inputs, device, data["agent_state"].to(device), agent_seq_len, out_type=0)

        spt_loss = criterion[0](spt_outputs, history_window.unsqueeze(1).repeat(1, 10, 1, 1))
        # not all the output steps are valid, so apply mask to ignore invalid ones
        spt_loss = spt_loss * torch.unsqueeze(history_mask, 2)
        spt_loss = spt_loss.mean()

        dopt.step(spt_loss)

    # Query loss evaluated only on future
    qry_outputs, qry_scores = f_model(inputs, device, data["agent_state"].to(device), agent_seq_len, out_type=3)
    labels = find_closest_traj(qry_outputs[:, :, 7:, :], targets[:, 7:, :])

    loss_reg_fut = criterion[0](qry_outputs[torch.arange(qry_outputs.size(0)), labels, 7:, :], targets[:, 7:, :])
    loss_reg_hist = criterion[0](qry_outputs[:, :, :7, :], targets[:, :7, :].unsqueeze(1).repeat(1, 10, 1, 1))
    loss_cls = criterion[1](qry_scores, labels)
    qry_loss = torch.cat((loss_reg_fut, torch.mean(loss_reg_hist, dim=1)), dim=1) + loss_cls
    qry_loss = qry_loss * torch.unsqueeze(target_mask, 2)

    # print("Outer loss: {:.4f}".format(qry_loss.mean().detach().item()))

    return qry_outputs, model.sm(qry_scores), qry_loss.mean().detach()


def dump_predictions(pred_out, scores, token, helper):
    instance, sample = token.split('_')
    annotation = helper.get_sample_annotation(instance, sample)
    for i in range(pred_out.shape[0]):
        pred_out[i, :, :] = convert_local_coords_to_global(pred_out[i, :, :], annotation['translation'],
                                                           annotation['rotation'])
    # Dump only future preds for eval
    pred_class = Prediction(instance, sample, pred_out[:, 7:, ], scores)
    return pred_class.serialize()


def evaluate(model, val_dl, device, criterion, inner_steps):
    val_out_ = []
    val_scores_ = []
    val_tokens_ = []
    val_losses_ = []
    progress_bar = tqdm(val_dl)
    model.train()

    for data in progress_bar:
        inner_opt = torch.optim.SGD(model.dec_parameters, lr=5e-3)
        with higher.innerloop_ctx(model, inner_opt, track_higher_grads=False) as (fmodel, diffopt):
            outputs, scores, val_loss = forward_mm(data, fmodel, device, criterion, diffopt, inner_steps)
        val_out_.extend(outputs.cpu().numpy())
        val_scores_.extend(scores.cpu().numpy())
        val_tokens_.extend(data["token"])
        val_losses_.append(val_loss.cpu().numpy())

    return val_out_, val_scores_, val_tokens_, val_losses_


if __name__ == '__main__':
    torch.cuda.empty_cache()
    model_out_dir_root = '/scratch/rodney/models/nuScenes'
    model_path = model_out_dir_root + "/MOCAST4_METALR_03_09_2021_13_07_41/Epoch_5_03_10_2021_11_11_41.pth"
    # ds_type = 'v1.0-trainval'
    ds_type = 'v1.0-mini'

    in_ch = 3
    out_pts = 12
    poly_deg = 5
    num_modes = 10
    inner_steps_ = 5

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    val_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-' + ds_type + '-val.h5', transform)

    nuscenes = NuScenes(ds_type, dataroot=NUSCENES_DATASET)
    pred_helper = PredictHelper(nuscenes)

    val_dl = DataLoader(val_ds, shuffle=True, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MOCAST4_METALR(in_ch, out_pts, poly_deg, num_modes).to(device)

    # if torch.cuda.device_count() > 1:
    #     # print("Using", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids=[0, 1])

    print("Loading model ", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))

    criterion_reg = nn.MSELoss(reduction="none")
    criterion_cls = nn.CrossEntropyLoss()

    # torch.set_grad_enabled(False)

    # Eval function
    val_out, val_scores, val_tokens, val_losses = evaluate(model, val_dl, device, [criterion_reg, criterion_cls],
                                                       inner_steps_)

    val_ds.close_hf()

    print("Avg val loss: {:.4f}".format(np.mean(val_losses)))

    model_preds = []
    for output, score, token in zip(val_out, val_scores, val_tokens):
        model_preds.append(dump_predictions(output, score, token, pred_helper))

    json.dump(model_preds, open(os.path.join('../../out', 'mocast4_preds.json'), "w"))

    '''############################ Quantitative ###########################################'''
    config = load_prediction_config(pred_helper, '../../config/eval_metric_config.json')
    print("[Eval] MOCAST4 metrics")
    eval_metrics('../../out/mocast4_preds.json', pred_helper, config, '../../out/mocast4_metrics.json')

    '''############################ Qualitative ###########################################'''
    for i in np.random.randint(0, len(val_out), 20):
        img = render_map(pred_helper, val_tokens[i])
        gt_cord = render_trajectories(pred_helper, val_tokens[i])
        fig, ax = plt.subplots(1, 1)
        ax.grid(b=None)
        ax.imshow(img)
        ax.plot(gt_cord[:7, 0],
                gt_cord[:7, 1],
                'w--^',
                linewidth=3,
                markersize=2,
                zorder=650,
                path_effects=[pe.Stroke(linewidth=4, foreground='g'), pe.Normal()])
        ax.plot(gt_cord[7:, 0],
                gt_cord[7:, 1],
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
