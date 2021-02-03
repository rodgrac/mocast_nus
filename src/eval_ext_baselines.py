# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

""" Script for running baseline models on a given nuscenes-split. """

import argparse
import json
import os

from process_ds import nuScenes_load
from utils import eval_metrics

from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.compute_metrics import compute_metrics
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.models.physics import ConstantVelocityHeading, PhysicsOracle


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


if __name__ == '__main__':
    NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes/'
    out_dir = '../out'

    helper = nuScenes_load('v1.0-mini', NUSCENES_DATASET)
    config = load_prediction_config(helper, '../config/eval_metric_config.json')

    #dump_cv_oracle_pred(helper, config, out_dir)

    print("[Eval] CV metrics")
    eval_metrics('../out/cv_preds.json', helper, config, '../out/cv_metrics.json')

    print("[Eval] Oracle metrics")
    eval_metrics('../out/oracle_preds.json', helper, config, '../out/oracle_metrics.json')

