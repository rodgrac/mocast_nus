import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../datasets/nuScenes/nuscenes-devkit/python-sdk')

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.helper import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer


def get_agent_state_hist(sample_annot, helper, past_xy):
    state_vec = np.zeros((5, 7), dtype=np.float)
    vel, acc, yawr = [], [], []
    hist_annot = [sample_annot] + helper._iterate(sample_annot, 3, 'prev')
    for h in hist_annot:
        temp = helper.get_velocity_for_agent(h['instance_token'], h['sample_token'])
        if not np.isnan(temp):
            vel.append(temp)

        temp = helper.get_acceleration_for_agent(h['instance_token'], h['sample_token'])
        if not np.isnan(temp):
            acc.append(temp)

        temp = helper.get_heading_change_rate_for_agent(h['instance_token'], h['sample_token'])
        if not np.isnan(temp):
            yawr.append(temp)

    if acc:
        state_vec[0, :len(acc)] = np.flip(past_xy[:len(acc), 0])
        state_vec[1, :len(acc)] = np.flip(past_xy[:len(acc), 1])
        state_vec[2, :len(acc)] = np.flip(vel[:len(acc)])
        state_vec[3, :len(acc)] = np.flip(np.array(acc))
        state_vec[4, :len(acc)] = np.flip(yawr[:len(acc)])

    return state_vec.T, len(acc)


def process_annot(sample, helper, input_rep, grps):
    global count, total_c
    print('Processing {}/{}'.format(count, total_c-1))
    try:
        dict = {}
        instance_token, sample_token = sample.split("_")
        sample_ann = helper.get_sample_annotation(instance_token, sample_token)
        img = input_rep.make_input_representation(instance_token, sample_token)

        future_xy = helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
        past_xy = helper.get_past_for_agent(instance_token, sample_token, seconds=3, in_agent_frame=True)
        past_xy = np.concatenate(([[0, 0]], past_xy), axis=0)
        past_xy = np.pad(past_xy, ((0, 7 - past_xy.shape[0]), (0, 0)), 'constant')

        state_vec, seq_len = get_agent_state_hist(sample_ann, helper, past_xy)
        past_mask = np.ones((1, seq_len), dtype=np.uint8)
        past_mask = np.pad(past_mask, ((0, 0), (0, 7 - seq_len)), 'constant')

        dict['image'] = img
        dict['agent_state'] = state_vec.astype(np.float32)
        dict['agent_future'] = np.array(future_xy, dtype=np.float32)
        dict['agent_past'] = np.array(past_xy, dtype=np.float32)
        dict['mask_past'] = past_mask
        dict['mask_future'] = np.ones((1, future_xy.shape[0]), dtype=np.uint8)
        dict['token'] = str(sample)

        dict_to_hdf(grps, dict, count)
        count += 1
        return True

    except:
        print("Something went wrong for sample {}. Skipping...".format(count))
        return False


def nuScenes_load(ds_name, dataroot):
    nuscenes = NuScenes(ds_name, dataroot=dataroot)
    helper = PredictHelper(nuscenes)
    return helper


def nuScenes_process(ds, helper, grps):
    static_rast = StaticLayerRasterizer(helper)
    agent_rast = AgentBoxesWithFadedHistory(helper, 3)
    input_rep = InputRepresentation(static_rast, agent_rast, Rasterizer())
    results = map(lambda sample: process_annot(sample, helper, input_rep, grps), ds)
    return list(results)


def dict_to_hdf(grps_list, data_dict, count):
    dt = h5py.special_dtype(vlen=str)
    print("Saving sample {} to HDF".format(count))

    for gp in grps_list:
        if gp.name[1:] == 'token':
            dset = gp.create_dataset(str(count), (100,), dtype=dt)
            dset[0] = data_dict[gp.name[1:]]
        else:
            gp.create_dataset(str(count), data=data_dict[gp.name[1:]], compression='gzip')


if __name__ == "__main__":
    NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes'
    GRPS = ['image', 'agent_state', 'agent_future', 'agent_past', 'mask_past', 'mask_future', 'token']

    # ds_type = 'v1.0-mini'
    ds_type = 'v1.0-trainval'
    helper = nuScenes_load(ds_type, NUSCENES_DATASET)

    # ----------------------------------------Train Set --------------------------------------------------------------#
    print("Packing training set")
    count = 0
    train_set = get_prediction_challenge_split("train", dataroot=NUSCENES_DATASET)
    print("Packed training set of length {}".format(len(train_set)))
    total_c = len(train_set)

    hf = h5py.File('/scratch/rodney/datasets/nuScenes/processed/nuscenes-{}-train.h5'.format(ds_type), 'w')
    grps = [hf.create_group(keys) for keys in GRPS]

    results = nuScenes_process(train_set, helper, grps)

    print("Number of fails: {}".format(np.count_nonzero(np.array(not results))))

    hf.close()

    # # ----------------------------------------Val Set ----------------------------------------------------------------#
    # print("Packing val set")
    # count = 0
    # val_set = get_prediction_challenge_split("val", dataroot=NUSCENES_DATASET)
    # print("Packed validation set of length {}".format(len(val_set)))
    # total_c = len(val_set)
    #
    # hf = h5py.File('/scratch/rodney/datasets/nuScenes/processed/nuscenes-{}-val.h5'.format(ds_type), 'w')
    # grps = [hf.create_group(keys) for keys in GRPS]
    #
    # results = nuScenes_process(val_set, helper, grps)
    #
    # print("Number of fails: {}".format(np.count_nonzero(np.array(not results))))
    #
    # hf.close()
