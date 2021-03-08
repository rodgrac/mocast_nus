import os
import sys
import yaml
import pickle
import json
import time
import torch

sys.path.append('../datasets/nuScenes/nuscenes-devkit/python-sdk')
from nuscenes.eval.prediction.compute_metrics import compute_metrics


def load_config_data(path):
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def pickle_save_obj(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def pickle_load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def eval_metrics(pred_file, helper, config, out_file):
    predictions = json.load(open(pred_file, "r"))
    results = compute_metrics(predictions, helper, config)
    json.dump(results, open(out_file, "w"), indent=2)
    print(json.dumps(results, indent=4, sort_keys=True))


def save_model_dict(model, out_dir, epoch):
    time_string = time.strftime("_%m_%d_%Y_%H_%M_%S", time.localtime())
    out_file = os.path.join(out_dir, 'Epoch_' + str(epoch) + time_string + '.pth')
    torch.save(model.state_dict(), out_file)
    print("Saved model as " + out_file)
