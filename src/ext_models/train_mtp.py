import sys
import time
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from process_ds import nuScenes_load

sys.path.append('../datasets/nuScenes/nuscenes-devkit/python-sdk')

from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss


class NuSceneDataset(Dataset):

    def __init__(self, tokens, helper):
        self.tokens = tokens
        self.static_layer_representation = StaticLayerRasterizer(helper)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]
        instance_token, sample_token = token.split("_")

        image = self.static_layer_representation.make_representation(instance_token, sample_token)
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes/'
    helper = nuScenes_load('v1.0-mini', NUSCENES_DATASET)

    train_set = get_prediction_challenge_split("mini_train", dataroot=NUSCENES_DATASET)

    dataset = NuSceneDataset(train_set, helper)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=16, num_workers=16)

    backbone = ResNetBackbone('resnet50')
    model = MTP(backbone, NUM_MODES)
    model = model.to(device)

    loss_function = MTPLoss(NUM_MODES, 1, 5)

    optimizer = optim.Adam(model.parameters(), 1e-4)

    progress_bar = tqdm(range(10))

    for _ in progress_bar:
        for img, agent_state_vector, ground_truth in dataloader:

            img = img.to(device)
            agent_state_vector = agent_state_vector.to(device)
            ground_truth = ground_truth.to(device)

            optimizer.zero_grad()

            prediction = model(img, agent_state_vector)

            loss = loss_function(prediction, ground_truth)
            loss.backward()
            optimizer.step()

            current_loss = loss.cpu().detach().numpy()

        progress_bar.set_description(f"loss: {current_loss.item()}")

    time_string = time.strftime("_%m_%d_%Y_%H_%M_%S", time.localtime())
    torch.save(model.state_dict(), '../../models/' + model.__class__.__name__ + time_string + '.pth')
    print("Saved model as ../../models/" + model.__class__.__name__ + time_string + '.pth')