import os
import sys
import numpy as np

sys.path.append('../datasets/nuScenes/nuscenes-devkit/python-sdk')

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.helper import PredictHelper
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

import matplotlib.pyplot as plt

NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes/'

nuscenes = NuScenes('v1.0-mini', dataroot=NUSCENES_DATASET)

train_set = get_prediction_challenge_split("mini_train", dataroot=NUSCENES_DATASET)

helper = PredictHelper(nuscenes)

instance_token, sample_token = train_set[0].split("_")

annot1 = helper.get_sample_annotation(instance_token, sample_token)

future_xy = helper.get_annotations_for_sample(sample_token)

nusc_map = NuScenesMap(map_name='singapore-onenorth', dataroot=NUSCENES_DATASET)

x, y, yaw = 395, 1095, 0
closest_lane = nusc_map.get_closest_lane(x, y, radius=2)
# print(arcline_path_utils.discretize_lane(nusc_map.get_arcline_path(closest_lane), resolution_meters=1))

static_rast = StaticLayerRasterizer(helper)
agent_rast = AgentBoxesWithFadedHistory(helper, 3)
input_rep = InputRepresentation(static_rast, agent_rast, Rasterizer())

img = input_rep.make_input_representation(instance_token, sample_token)

print([helper.get_velocity_for_agent(instance_token, sample_token),
                                    helper.get_acceleration_for_agent(instance_token, sample_token),
                                    helper.get_heading_change_rate_for_agent(instance_token, sample_token)])

plt.imshow(img)

plt.show()







