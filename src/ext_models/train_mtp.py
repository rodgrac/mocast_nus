import sys
import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from process_ds import nuScenes_load
from utils import save_model_dict

sys.path.append('../datasets/nuScenes/nuscenes-devkit/python-sdk')

from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss


class NuSceneDataset(Dataset):

    def __init__(self, tokens, helper):
        self.tokens = tokens
        self.static_layer_rasterizer = StaticLayerRasterizer(helper)
        self.agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=2)
        self.mtp_input_rep = InputRepresentation(self.static_layer_rasterizer, self.agent_rasterizer, Rasterizer())

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]
        instance_token, sample_token = token.split("_")

        image = self.mtp_input_rep.make_input_representation(instance_token, sample_token)

        image = torch.Tensor(image).permute(2, 0, 1)
        agent_state_vector = torch.Tensor([[helper.get_velocity_for_agent(instance_token, sample_token),
                                            helper.get_acceleration_for_agent(instance_token, sample_token),
                                            helper.get_heading_change_rate_for_agent(instance_token,
                                                                                     sample_token)]]).squeeze()
        agent_state_vector[torch.isnan(agent_state_vector)] = 0

        ground_truth = torch.Tensor(
            helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)).unsqueeze(0)

        return image, agent_state_vector, ground_truth


if __name__ == "__main__":

    NUM_MODES = 10

    NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes/'
    helper = nuScenes_load('v1.0-trainval', NUSCENES_DATASET)

    model_out_dir_root = '/scratch/rodney/models/nuScenes'

    train_set = get_prediction_challenge_split("train", dataroot=NUSCENES_DATASET)

    dataset = NuSceneDataset(train_set, helper)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=16, num_workers=16)

    model_out_dir = os.path.join(model_out_dir_root,'MTP' + time.strftime("_%m_%d_%Y_%H_%M_%S", time.localtime()))
    print("Model out directory:", model_out_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    backbone = ResNetBackbone('resnet50')
    model = MTP(backbone, num_modes=NUM_MODES, seconds=6)
    model = model.to(device)

    loss_function = MTPLoss(NUM_MODES, 1, 5)

    optimizer = optim.Adam(model.parameters(), 1e-4)

    epochs = 15

    model.train()
    torch.set_grad_enabled(True)

    for epoch in range(epochs):
        avg_loss = []
        progress_bar = tqdm(dataloader)
        for img, agent_state_vector, ground_truth in progress_bar:

            img = img.to(device)
            agent_state_vector = agent_state_vector.to(device)
            ground_truth = ground_truth.to(device)

            optimizer.zero_grad()

            prediction = model(img, agent_state_vector)

            loss = loss_function(prediction, ground_truth)
            loss.backward()
            optimizer.step()

            current_loss = loss.cpu().detach().numpy()
            avg_loss.append(current_loss.item())

            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}=> loss: {current_loss.item()}, avgloss: {np.mean(avg_loss)}")

    save_model_dict(model, model_out_dir, epochs)