import sys
import h5py
import numpy as np
from pyquaternion import Quaternion
from scipy.stats import binned_statistic
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
from nuscenes.prediction.input_representation.utils import convert_to_pixel_coords, get_rotation_matrix
from nuscenes.prediction.helper import quaternion_yaw

vehicle_filter = ['vehicle.car', 'vehicle.construction', 'vehicle.motorcycle', 'vehicle.trailer',
                  'vehicle.truck', 'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.emergency.ambulance',
                  'vehicle.emergency.police']
visibility_filter = ['1', '2', '3', '4']


def agent_list_filtering(inst_token, sample_token):
    annots_new = []
    agent_annots = helper.get_annotations_for_sample(sample_token)
    ego_annot = helper.get_sample_annotation(inst_token, sample_token)
    for annot in agent_annots:
        if annot["instance_token"] != inst_token and annot["category_name"] in vehicle_filter \
                and annot['visibility_token'] in visibility_filter:
            agent_pixel_cord = convert_to_pixel_coords(annot['translation'][:2], ego_annot['translation'][:2],
                                                       (400, 400), 0.1)

            ego_agent_yaw = quaternion_yaw(Quaternion(ego_annot['rotation']))
            rotation_mat = get_rotation_matrix((800, 800, 3), ego_agent_yaw)
            agent_pixel_cord = rotation_mat @ np.append((agent_pixel_cord[1], agent_pixel_cord[0]), 1)

            if int(agent_pixel_cord[0]) in range(800) and int(agent_pixel_cord[1]) in range(800):
                pos = np.zeros(2)
                pos[0] = binned_statistic(agent_pixel_cord[1], agent_pixel_cord[1], bins=14, range=[0, 800])[2][0] - 1
                pos[1] = binned_statistic(agent_pixel_cord[0], agent_pixel_cord[0], bins=14, range=[0, 800])[2][0] - 1
                annot['grid_pos'] = pos
                annots_new.append(annot)

    return annots_new


def get_agent_state_hist(sample_annot, helper):
    hist_pts = HIST_SECS * 2 + 1
    state_vec = np.zeros((5, hist_pts), dtype=np.float)
    vel, acc, yawr = [], [], []
    hist_annot = [sample_annot] + helper._iterate(sample_annot, HIST_SECS, 'prev')

    past_xy = helper.get_past_for_agent(sample_annot['instance_token'], sample_annot['sample_token'], seconds=HIST_SECS,
                                        in_agent_frame=True)
    if not past_xy.size:
        past_xy = [[0, 0]]
    else:
        past_xy = np.concatenate(([[0, 0]], past_xy), axis=0)
    hist_steps = len(past_xy)
    past_xy = np.pad(past_xy, ((0, hist_pts - hist_steps), (0, 0)), 'constant')

    # No agent state if no history
    if not np.sum(past_xy):
        return state_vec.T, 0, past_xy

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

    state_vec[0, :hist_steps] = np.flip(past_xy[:hist_steps, 0])
    state_vec[1, :hist_steps] = np.flip(past_xy[:hist_steps, 1])
    state_vec[2, :len(vel)] = np.flip(np.array(vel))
    state_vec[3, :len(acc)] = np.flip(np.array(acc))
    state_vec[4, :len(yawr)] = np.flip(np.array(yawr))

    return state_vec.T, hist_steps, past_xy


def process_annot(sample, helper, input_rep, grps):
    global count, total_c, max_agents
    hist_pts = HIST_SECS * 2 + 1
    print('Processing {}/{}'.format(count, total_c - 1))
    #try:
    dict = {}
    instance_token, sample_token = sample.split("_")
    ego_ann = helper.get_sample_annotation(instance_token, sample_token)
    img = input_rep.make_input_representation(instance_token, sample_token, input_type=0, resize=jam_rep)

    filtered_agents_ann = agent_list_filtering(instance_token, sample_token)

    # # Ego vehicle
    future_xy = helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
    # Ego state
    ego_state_vec, ego_seq_len, past_xy = get_agent_state_hist(ego_ann, helper)
    ego_past_mask = np.ones((1, ego_seq_len), dtype=np.uint8)
    ego_past_mask = np.pad(ego_past_mask, ((0, 0), (0, hist_pts - ego_seq_len)), 'constant')

    # # Other vehicle agents
    agents_state_vec = np.zeros((max_agents, hist_pts, 5), dtype=np.float32)
    agents_seq_len = np.zeros(max_agents, dtype=np.uint8)
    agents_grid_pos = np.zeros((max_agents, 2), dtype=np.uint8)
    num_agents = 0
    for ann in filtered_agents_ann:
        state_vec, seq_len, _ = get_agent_state_hist(ann, helper)
        if seq_len and (ann['grid_pos'] > -1).all():
            agents_state_vec[num_agents] = state_vec
            agents_seq_len[num_agents] = seq_len
            agents_grid_pos[num_agents] = ann['grid_pos']
            num_agents += 1

    print("Filtered agents:", num_agents)
    # plt.imshow(img)
    # ax = plt.gca()
    # ax.set_xticks(np.arange(0, 225, 225 / 14))
    # ax.set_yticks(np.arange(0, 225, 225 / 14))
    # plt.show()

    dict['image'] = img
    dict['ego_state'] = ego_state_vec.astype(np.float32)
    dict['num_agents'] = np.array(num_agents, dtype=np.uint8)
    dict['agents_state'] = agents_state_vec
    dict['agents_seq_len'] = agents_seq_len
    dict['agents_rel_pos'] = agents_grid_pos
    dict['ego_future'] = np.array(future_xy, dtype=np.float32)
    dict['ego_past'] = np.array(past_xy, dtype=np.float32)
    dict['ego_mask_past'] = ego_past_mask
    dict['ego_mask_future'] = np.ones((1, future_xy.shape[0]), dtype=np.uint8)
    dict['token'] = str(sample)

    dict_to_hdf(grps, dict, count)
    count += 1

    return True

    # except:
    #     print("Something went wrong for sample {}. Skipping...".format(count))
    #     return False


def nuScenes_load(ds_name, dataroot):
    nuscenes = NuScenes(ds_name, dataroot=dataroot)
    helper = PredictHelper(nuscenes)
    return helper


def nuScenes_process(ds, helper, grps):
    static_rast = StaticLayerRasterizer(helper)
    agent_rast = AgentBoxesWithFadedHistory(helper, HIST_SECS)
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

    jam_rep = True
    # ds_type = 'v1.0-mini'
    ds_type = 'v1.0-trainval'
    helper = nuScenes_load(ds_type, NUSCENES_DATASET)
    ds_sets = ['train', 'val']
    # max_agents_l = [27, 31]
    max_agents_l = [48, 36]
    HIST_SECS = 2

    for i, s in enumerate(ds_sets):
        # ----------------------------------------Run process --------------------------------------------------------#
        max_agents = max_agents_l[i]
        count = 0
        if ds_type == 'v1.0-mini':
            split_n = 'mini_' + s
        else:
            split_n = s
        train_set = get_prediction_challenge_split(split_n, dataroot=NUSCENES_DATASET)
        total_c = len(train_set)
        print("Packing training set of samples:", total_c)
        if jam_rep:
            print("JAM Representation")
            GRPS = ['image', 'ego_state', 'agents_state', 'agents_seq_len', 'agents_rel_pos', 'ego_future', 'ego_past',
                    'ego_mask_past', 'ego_mask_future', 'token']
            hf = h5py.File(
                '/scratch/rodney/datasets/nuScenes/processed/nuscenes-jam-{}-{}-{}s.h5'.format(ds_type, s, HIST_SECS),
                'w')
        else:
            print("Fading representation")
            GRPS = ['image', 'agent_state', 'agent_future', 'agent_past', 'mask_past', 'mask_future', 'token']
            hf = h5py.File(
                '/scratch/rodney/datasets/nuScenes/processed/nuscenes-{}-{}-{}s.h5'.format(ds_type, s, HIST_SECS), 'w')

        grps = [hf.create_group(keys) for keys in GRPS]

        results = nuScenes_process(train_set, helper, grps)

        print("Number of fails: {}".format(np.count_nonzero(np.array(not results))))

        hf.close()
