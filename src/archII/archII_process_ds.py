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


def sa_list_filtering(inst_token, sample_token):
    annots_new = []
    sa_annots = helper.get_annotations_for_sample(sample_token)
    ta_annot = helper.get_sample_annotation(inst_token, sample_token)
    for annot in sa_annots:
        if annot["instance_token"] != inst_token and annot["category_name"] in vehicle_filter \
                and annot['visibility_token'] in visibility_filter:
            agent_pixel_cord = convert_to_pixel_coords(annot['translation'][:2], ta_annot['translation'][:2],
                                                       (400, 400), 0.1)

            ta_agent_yaw = quaternion_yaw(Quaternion(ta_annot['rotation']))
            rotation_mat = get_rotation_matrix((800, 800, 3), ta_agent_yaw)
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
        return state_vec.T, 0

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

    return state_vec.T, hist_steps


def process_annot(sample, helper, input_rep, grps):
    global count, total_c, max_agents
    hist_pts = HIST_SECS * 2 + 1
    print('Processing {}/{}'.format(count, total_c - 1))
    #try:
    dict = {}
    instance_token, sample_token = sample.split("_")
    ta_ann = helper.get_sample_annotation(instance_token, sample_token)
    img = input_rep.make_input_representation(instance_token, sample_token, input_type=0, resize=arch_II)

    filtered_sa_ann = sa_list_filtering(instance_token, sample_token)

    # Target agent
    ta_past_xy = helper.get_past_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
    ta_past_xy = np.concatenate(([[0, 0]], ta_past_xy), axis=0)
    ta_past_seqlen = len(ta_past_xy)
    
    ta_past_mask = np.ones((1, ta_past_seqlen), dtype=np.uint8)

    ta_past_xy = np.pad(ta_past_xy, ((0, 13 - ta_past_xy.shape[0]), (0, 0)), 'constant')
    ta_past_mask = np.pad(ta_past_mask, ((0, 0), (0, 13 - ta_past_seqlen)), 'constant')

    ta_future_xy = helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
    
    # Target agent state history
    ta_sthist_tensor, ta_sthist_len = get_agent_state_hist(ta_ann, helper)

    # # Other vehicle agents
    sa_sthist_tensor = np.zeros((max_agents, hist_pts, 5), dtype=np.float32)
    sa_sthist_len = np.zeros(max_agents, dtype=np.uint8)
    sa_grid_pos = np.zeros((max_agents, 2), dtype=np.uint8)
    num_sa = 0
    for ann in filtered_sa_ann:
        state_tensor, seq_len = get_agent_state_hist(ann, helper)
        if seq_len and (ann['grid_pos'] > -1).all():
            sa_sthist_tensor[num_sa] = state_tensor
            sa_sthist_len[num_sa] = seq_len
            sa_grid_pos[num_sa] = ann['grid_pos']
            num_sa += 1

    print("Filtered surrounding agents: ", num_sa)
    # plt.imshow(img)
    # ax = plt.gca()
    # ax.set_xticks(np.arange(0, 225, 225 / 14))
    # ax.set_yticks(np.arange(0, 225, 225 / 14))
    # plt.show()

    dict['image'] = img
    dict['ta_past'] = np.array(ta_past_xy, dtype=np.float32)
    dict['ta_future'] = np.array(ta_future_xy, dtype=np.float32)
    dict['ta_mask_past'] = ta_past_mask
    dict['ta_mask_future'] = np.ones((1, ta_future_xy.shape[0]), dtype=np.uint8)
    dict['ta_state_hist'] = ta_sthist_tensor.astype(np.float32)

    dict['sa_state_hist'] = sa_sthist_tensor
    dict['sa_sthist_len'] = sa_sthist_len
    dict['sa_grid_pos'] = sa_grid_pos
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

    arch_II = True
    # ds_type = 'v1.0-mini'
    ds_type = 'v1.0-trainval'
    helper = nuScenes_load(ds_type, NUSCENES_DATASET)
    ds_sets = ['train', 'val']

    HIST_SECS = 2

    if ds_type == 'v1.0-mini':
        max_agents_l = [27, 31]
    else:
        max_agents_l = [48, 36]

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
        if arch_II:
            print("Arch-II representation")
            GRPS = ['image', 'ta_past', 'ta_future', 'ta_mask_past', 'ta_mask_future', 'ta_state_hist',
                    'sa_state_hist', 'sa_sthist_len', 'sa_grid_pos', 'token']
            hf = h5py.File(NUSCENES_DATASET + '/processed/nuscenes-archII-{}-{}.h5'.format(ds_type, s), 'w')
        else:
            print("Arch-I representation")
            GRPS = ['image', 'ta_past', 'ta_future', 'ta_mask_past', 'ta_mask_future', 'ta_state_hist', 'ta_sthist_len',
                    'token']
            hf = h5py.File(NUSCENES_DATASET + '/processed/nuscenes-archI-{}-{}.h5'.format(ds_type, s), 'w')

        grps = [hf.create_group(keys) for keys in GRPS]

        results = nuScenes_process(train_set, helper, grps)

        # print("Number of fails: {}".format(np.count_nonzero(np.array(not results))))

        hf.close()
