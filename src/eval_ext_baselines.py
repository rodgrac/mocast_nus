import json
import os
import torch
import pickle

from process_ds import nuScenes_load
from utils import eval_metrics

from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.helper import convert_local_coords_to_global
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.prediction.models.physics import ConstantVelocityHeading, PhysicsOracle
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP
from nuscenes.prediction.models.covernet import CoverNet

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer


def dump_cv_oracle_pred(helper, config, output_dir):
    """
    Performs inference for all of the baseline models defined in the physics model module.
    :param version: nuScenes dataset version.
    :param data_root: Directory where the NuScenes data is stored.
    :param split_name: nuScenes data split name, e.g. train, val, mini_train, etc.
    :param output_dir: Directory where predictions should be stored.
    :param config_name: Name of config file.
    """
    dataset = get_prediction_challenge_split("mini_val", NUSCENES_DATASET)

    oracle = PhysicsOracle(config.seconds, helper)
    cv_heading = ConstantVelocityHeading(config.seconds, helper)

    cv_preds = []
    oracle_preds = []
    for token in dataset:
        cv_preds.append(cv_heading(token).serialize())
        oracle_preds.append(oracle(token).serialize())

    json.dump(cv_preds, open(os.path.join(output_dir, "cv_preds.json"), "w"))
    json.dump(oracle_preds, open(os.path.join(output_dir, "oracle_preds.json"), "w"))


def dump_mtp_pred(helper, config, output_dir):
    dataset = get_prediction_challenge_split("mini_val", NUSCENES_DATASET)

    backbone = ResNetBackbone('resnet50')
    mtp = MTP(backbone, num_modes=10, seconds=5)

    static_layer_rasterizer = StaticLayerRasterizer(helper)
    agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)
    mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    model_preds = []

    for token in dataset:
        instance_token_img, sample_token_img = token.split('_')
        agent_state_vector = torch.Tensor([[helper.get_velocity_for_agent(instance_token_img, sample_token_img),
                                            helper.get_acceleration_for_agent(instance_token_img, sample_token_img),
                                            helper.get_heading_change_rate_for_agent(instance_token_img,
                                                                                     sample_token_img)]])

        img = mtp_input_representation.make_input_representation(instance_token_img, sample_token_img)
        image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
        out = mtp(image_tensor, agent_state_vector).squeeze()
        if torch.isnan(out).any():
            continue

        pred_loc, scores = out[:-10], out[-10:].detach().numpy()
        pred_loc = pred_loc.view(-1, 2, 10).permute(0, 2, 1).detach().numpy()

        annotation = helper.get_sample_annotation(instance_token_img, sample_token_img)
        for i in range(pred_loc.shape[0]):
            pred_loc[i, :, :] = convert_local_coords_to_global(pred_loc[i, :, :], annotation['translation'],
                                                               annotation['rotation'])

        pred_class = Prediction(instance_token_img, sample_token_img, pred_loc, scores)
        model_preds.append(pred_class.serialize())

    json.dump(model_preds, open(os.path.join(output_dir, 'mtp_preds.json'), "w"))


def dump_covernet_pred(helper, config, output_dir):
    PATH_TO_EPSILON_8_SET = "/home/dsouza.94/projects/mocast/mocast_nus/misc/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"

    dataset = get_prediction_challenge_split("mini_val", NUSCENES_DATASET)

    backbone = ResNetBackbone('resnet50')
    covernet = CoverNet(backbone, num_modes=64)

    trajectories = torch.Tensor(pickle.load(open(PATH_TO_EPSILON_8_SET, 'rb')))

    static_layer_rasterizer = StaticLayerRasterizer(helper)
    agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)
    mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    model_preds = []

    for token in dataset:
        instance_token_img, sample_token_img = token.split('_')
        agent_state_vector = torch.Tensor([[helper.get_velocity_for_agent(instance_token_img, sample_token_img),
                                            helper.get_acceleration_for_agent(instance_token_img, sample_token_img),
                                            helper.get_heading_change_rate_for_agent(instance_token_img,
                                                                                     sample_token_img)]])

        img = mtp_input_representation.make_input_representation(instance_token_img, sample_token_img)
        image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
        logits = covernet(image_tensor, agent_state_vector).argsort(descending=True).squeeze()

        pred_loc, scores = trajectories[logits[:10]], logits[:10]

        if torch.isnan(pred_loc).any():
            continue

        pred_loc, scores = pred_loc.detach().numpy(), scores.detach().numpy()

        annotation = helper.get_sample_annotation(instance_token_img, sample_token_img)
        for i in range(pred_loc.shape[0]):
            pred_loc[i, :, :] = convert_local_coords_to_global(pred_loc[i, :, :], annotation['translation'],
                                                               annotation['rotation'])

        pred_class = Prediction(instance_token_img, sample_token_img, pred_loc, scores)
        model_preds.append(pred_class.serialize())

    json.dump(model_preds, open(os.path.join(output_dir, 'covernet_preds.json'), "w"))


if __name__ == '__main__':
    NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes/'
    out_dir = '../out'

    helper = nuScenes_load('v1.0-mini', NUSCENES_DATASET)
    config = load_prediction_config(helper, '../config/eval_metric_config.json')

    # dump_cv_oracle_pred(helper, config, out_dir)

    # dump_mtp_pred(helper, config, out_dir)

    # dump_covernet_pred(helper, config, out_dir)

    # print("[Eval] CV metrics")
    # eval_metrics('../out/cv_preds.json', helper, config, '../out/cv_metrics.json')
    #
    # print("[Eval] Oracle metrics")
    # eval_metrics('../out/oracle_preds.json', helper, config, '../out/oracle_metrics.json')

    # print("[Eval] MTP metrics")
    # eval_metrics('../out/mtp_preds.json', helper, config, '../out/mtp_metrics.json')

    # print("[Eval] CoverNet metrics")
    # eval_metrics('../out/covernet_preds.json', helper, config, '../out/covernet_metrics.json')
