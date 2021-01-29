import sys
import os
import torch
import numpy as np
from itertools import chain

sys.path.append('../datasets/nuScenes/nuscenes-devkit/python-sdk')

from utils import *

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.helper import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer


def get_agent_state_hist(sample_annot, helper):
    state_vec = np.zeros((7, 3))
    hist_annot = [sample_annot] + helper._iterate(sample_annot, 3, 'prev')
    for i, h in enumerate(hist_annot):
        state_vec[i][0] = helper.get_velocity_for_agent(h['instance_token'], h['sample_token'])

        state_vec[i][1] = helper.get_acceleration_for_agent(h['instance_token'], h['sample_token'])

        state_vec[i][2] = helper.get_heading_change_rate_for_agent(h['instance_token'], h['sample_token'])

    state_vec = np.nan_to_num(state_vec)
    state_vec_ = torch.flip(torch.Tensor(state_vec), [0])

    return state_vec_


def process_annot(sample, helper, input_rep):
    global count, total_c
    print('Processed {}/{}'.format(count, total_c))
    dict = {}
    instance_token, sample_token = sample.split("_")
    sample_ann = helper.get_sample_annotation(instance_token, sample_token)
    img = input_rep.make_input_representation(instance_token, sample_token)
    future_xy = helper.get_future_for_agent(instance_token, sample_token, seconds=5, in_agent_frame=True)

    dict['image'] = torch.Tensor(img).permute(2, 0, 1)
    dict['agent_state'] = get_agent_state_hist(sample_ann, helper)
    dict['agent_future'] = torch.Tensor(future_xy)

    count += 1
    return dict


def nuScenes_load(ds_name, ds_type, dataroot):
    nuscenes = NuScenes(ds_name, dataroot=dataroot)
    pred_helper = PredictHelper(nuscenes)

    train_set = get_prediction_challenge_split(ds_type, dataroot=dataroot)

    return train_set, pred_helper


def nuScenes_process(ds, helper):
    static_rast = StaticLayerRasterizer(helper)
    agent_rast = AgentBoxesWithFadedHistory(helper, 3)
    input_rep = InputRepresentation(static_rast, agent_rast, Rasterizer())
    train_proc = map(lambda sample: process_annot(sample, helper, input_rep), ds)
    return list(train_proc)


if __name__ == "__main__":
    NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes/'

    count = 0

    train_set, helper = nuScenes_load('v1.0-mini', "mini_train", NUSCENES_DATASET)
    total_c = len(train_set)
    train_ds = nuScenes_process(train_set, helper)

    save_obj(train_ds, '../datasets/nuScenes/processed')
