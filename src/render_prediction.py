import sys
import numpy as np
import cv2
from pyquaternion import Quaternion

sys.path.append('../datasets/nuScenes/nuscenes-devkit/python-sdk')

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction.input_representation.utils import convert_to_pixel_coords, get_rotation_matrix
from nuscenes.prediction.helper import quaternion_yaw

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

roi_global = [40, 10, 25, 25]


def init_map(helper):
    static_rast = StaticLayerRasterizer(helper)
    agent_rast = AgentBoxesWithFadedHistory(helper, 3)
    input_rep = InputRepresentation(static_rast, agent_rast, Rasterizer())

    return input_rep


def render_map(helper, token):
    instance_token, sample_token = token.split("_")
    input_rep = init_map(helper)
    img = input_rep.make_input_representation(instance_token, sample_token)
    return img


def crop_offsets():
    row_off = (0, int((roi_global[0] + roi_global[1]) / 0.1))
    col_off = (int(800 / 2 - (roi_global[2] / 0.1)), int(800 / 2 + (roi_global[3] / 0.1)))
    return row_off, col_off


def render_trajectories(helper, token, traj=None):
    img_cord = []
    instance, sample = token.split('_')
    sample_ann = helper.get_sample_annotation(instance, sample)
    location = sample_ann['translation'][:2]
    # Render ground truth
    if traj is None:
        traj = helper.get_future_for_agent(instance, sample, 6, in_agent_frame=False)
    sample_yaw = quaternion_yaw(Quaternion(sample_ann['rotation']))
    rotation_mat = get_rotation_matrix((800, 800, 3), sample_yaw)
    row_f, col_f = crop_offsets()

    for l in range(traj.shape[0]):
        pixel_cords = convert_to_pixel_coords(traj[l], location, (400, 400))
        pixel_cords = (pixel_cords[1], pixel_cords[0])
        pixel_cords = rotation_mat @ np.append(pixel_cords, 1).T
        pixel_cords = pixel_cords - (col_f[0], row_f[0])
        img_cord.append(pixel_cords)
    return np.array(img_cord)

