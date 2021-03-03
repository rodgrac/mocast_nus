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

from meta_lr.meta_model import MOCAST4_METALR
from nusc_dataloader import NuScenes_HDF
from render_prediction import render_map, render_trajectories
from utils import eval_metrics

sys.path.append('../datasets/nuScenes/nuscenes-devkit/python-sdk')

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.prediction.helper import convert_local_coords_to_global

NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes/'


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
    model.train()
    inputs = data["image"].to(device)

    agent_seq_len = torch.sum(data["mask_past"], dim=1).to(device)
    history_window = torch.flip(data["agent_past"], [1]).to(device)
    history_mask = torch.flip(data['mask_past'], [1]).to(device)

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

    return outputs, scores


def dump_predictions(pred_out, scores, token, helper):
    instance, sample = token.split('_')
    annotation = helper.get_sample_annotation(instance, sample)
    for i in range(pred_out.shape[0]):
        pred_out[i, :, :] = convert_local_coords_to_global(pred_out[i, :, :], annotation['translation'],
                                                           annotation['rotation'])
    pred_class = Prediction(instance, sample, pred_out, scores)
    return pred_class.serialize()


torch.cuda.empty_cache()
model_path = "../../models/DataParallel_03_01_2021_17_56_30.pth"
ds_type = 'v1.0-trainval'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])

val_ds = NuScenes_HDF('/scratch/rodney/datasets/nuScenes/processed/nuscenes-' + ds_type + '-val.h5', transform)

nuscenes = NuScenes(ds_type, dataroot=NUSCENES_DATASET)
pred_helper = PredictHelper(nuscenes)

val_dl = DataLoader(val_ds, shuffle=True, batch_size=8, num_workers=8, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MOCAST4_METALR(3, 12, 5, 10, train=False).to(device)

if torch.cuda.device_count() > 1:
    # print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=[0, 1])

print("Loading model ", model_path)
model.load_state_dict(torch.load(model_path))

criterion_reg = nn.MSELoss(reduction="none")
criterion_cls = nn.CrossEntropyLoss()

model.eval()
torch.set_grad_enabled(False)

val_out = []
val_scores = []
val_tokens = []
progress_bar = tqdm(val_dl)

org_params = clone_model_param(model)
for data in progress_bar:
    with torch.enable_grad():
        outputs, scores = forward_mm(data, model, device, [criterion_reg, criterion_cls])

    model.eval()
    update_param_data(model, org_params)

    val_out.extend(outputs.cpu().numpy())
    val_scores.extend(scores.cpu().numpy())
    val_tokens.extend(data["token"])

val_ds.close_hf()

model_preds = []
for output, score, token in zip(val_out, val_scores, val_tokens):
    model_preds.append(dump_predictions(output, score, token, pred_helper))

json.dump(model_preds, open(os.path.join('../../out', 'mocast4_preds.json'), "w"))

'''############################ Quantitative ###########################################'''
config = load_prediction_config(pred_helper, '../../config/eval_metric_config.json')
print("[Eval] MOCAST4 metrics")
eval_metrics('../../out/mocast4_preds.json', pred_helper, config, '../../out/mocast4_metrics.json')

'''############################ Qualitative ###########################################'''
for i in np.random.randint(0, len(val_out), 25):
    img = render_map(pred_helper, val_tokens[i])
    gt_cord = render_trajectories(pred_helper, val_tokens[i])
    fig, ax = plt.subplots(1, 1)
    ax.grid(b=None)
    ax.imshow(img)
    ax.plot(gt_cord[:, 0],
            gt_cord[:, 1],
            'w--o',
            linewidth=4,
            markersize=3,
            zorder=650,
            path_effects=[pe.Stroke(linewidth=5, foreground='r'), pe.Normal()])

    top_3 = np.argsort(val_scores[i])[-1:-4:-1]
    for ind in top_3:
        pred_cord = render_trajectories(pred_helper, val_tokens[i], val_out[i][ind])

        ax.plot(pred_cord[:, 0],
                pred_cord[:, 1],
                'w--o',
                linewidth=4,
                markersize=3,
                zorder=650,
                path_effects=[pe.Stroke(linewidth=5, foreground='b'), pe.Normal()])
        plt.text(pred_cord[-1][0] + 10, pred_cord[-1][1], "{:0.2f}".format(val_scores[i][ind]))
#
plt.show()
