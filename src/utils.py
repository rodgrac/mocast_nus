import os
import sys
import yaml
import pickle
import json
import time
import torch
from torchviz import make_dot

sys.path.append('../../datasets/nuScenes/nuscenes-devkit/python-sdk')
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


def dump_model_graph(variable, model):
    dot = make_dot(variable, params=dict(model.named_parameters()))
    dot.format = 'png'
    # time_string = time.strftime("_%m_%d_%Y_%H_%M_%S", time.localtime())
    # dot.render(model.__class__.__name__ + '_' + time_string)
    dot.render('torchviz')


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


def clone_model_param(model):
    new_param = {}
    for name, params in model.named_parameters():
        new_param[name] = params.clone()

    return new_param


def reset_param_data(model, new_params):
    for name, params in model.named_parameters():
        params.data.copy_(new_params[name].data)


# Returns closest mode to GT
def find_closest_traj(pred, gt):
    ade = torch.sum((gt.unsqueeze(1) - pred) ** 2, dim=-1) ** 0.5
    ade = torch.mean(ade, dim=-1)
    return torch.argmin(ade, dim=-1)
