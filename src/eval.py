import sys
import os
import torch
import time
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from model import MOCAST_4
from process_ds import load_obj
from render_prediction import render_map, render_trajectories
from utils import eval_metrics

sys.path.append('../datasets/nuScenes/nuscenes-devkit/python-sdk')

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.prediction.helper import convert_local_coords_to_global

NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes/'


def forward_mm(data, model, device):
    inputs = data["image"].to(device)
    agent_seq_len = data["agent_state_len"].to(device)

    # Forward pass
    outputs, scores = model(inputs, device, data["agent_state"].to(device), agent_seq_len)

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

val_ds = load_obj('../datasets/nuScenes/processed/nuscenes-mini-val.pkl')
model_path = "../models/MOCAST_4_02_01_2021_22_07_15.pth"

nuscenes = NuScenes('v1.0-mini', dataroot=NUSCENES_DATASET)
pred_helper = PredictHelper(nuscenes)

val_dl = DataLoader(val_ds, shuffle=False, batch_size=16, num_workers=16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MOCAST_4(3, 10, 5, 10, 16, train=False).to(device)

print("Loading model ", model_path)
model.load_state_dict(torch.load(model_path))

model.eval()
torch.set_grad_enabled(False)

val_out = []
val_scores = []
val_tokens = []
progress_bar = tqdm(val_dl)
for data in progress_bar:
    outputs, scores = forward_mm(data, model, device)
    val_out.extend(outputs.cpu().numpy())
    val_scores.extend(scores.cpu().numpy())
    val_tokens.extend(data["token"])

model_preds = []
for output, score, token in zip(val_out, val_scores, val_tokens):
    model_preds.append(dump_predictions(output, score, token, pred_helper))

json.dump(model_preds, open(os.path.join('../out', 'mocast4_preds.json'), "w"))


'''############################ Quantitative ###########################################'''
config = load_prediction_config(pred_helper, '../config/eval_metric_config.json')
print("[Eval] MOCAST4 metrics")
eval_metrics('../out/mocast4_preds.json', pred_helper, config, '../out/mocast4_metrics.json')


'''############################ Qualitative ###########################################'''
for i in range(20, len(val_out), 10):
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

    break
#
plt.show()
