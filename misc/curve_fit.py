import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from orthnet import Legendre
from scipy.fft import dct, idct
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper

NUSCENES_DATASET = '/scratch/rodney/datasets/nuScenes/'


def get_dict(hf, index):
    dict = {}
    for gp in hf.keys():
        if gp == 'token':
            dict[gp] = hf.get(gp).get(str(index)).asstr()[0]
        else:
            dict[gp] = np.array(hf.get(gp).get(str(index)))

    return dict


def fit_full_path(data):
    hist_seq_len = np.sum(data['mask_past']).astype('uint8')
    past_flipped = np.flip(data['agent_past'][:hist_seq_len, :], axis=0)
    agent_path = np.concatenate((past_flipped, data['agent_future']), axis=0)
    l_t = 1 - hist_seq_len
    t = np.arange(l_t, 13)

    coeffs = np.polyfit(t, agent_path, deg=5)

    fit_x = np.poly1d(coeffs[:, 0])
    fit_y = np.poly1d(coeffs[:, 1])

    return np.column_stack((fit_x(t), fit_y(t))), agent_path


def fit_full_path_ortho(data):
    hist_seq_len = np.sum(data['mask_past']).astype('uint8')
    past_flipped = np.flip(data['agent_past'][:hist_seq_len, :], axis=0)
    agent_path = np.concatenate((past_flipped, data['agent_future']), axis=0)
    l_t = 1 - hist_seq_len
    t = np.arange(l_t, 13)
    basis = Legendre(np.expand_dims(t, 1), 5).tensor
    coeffs = np.linalg.lstsq(basis, agent_path)[0]

    return basis @ coeffs, agent_path


def fit_history(data):
    hist_seq_len = np.sum(data['mask_past']).astype('uint8')
    past_flipped = np.flip(data['agent_past'][:hist_seq_len, :], axis=0)
    l_t = 1 - hist_seq_len
    t_in = np.arange(l_t, 1)
    t_out = np.arange(1, 13)

    coeffs = np.polyfit(t_in, past_flipped, deg=2)
    fit_x = np.poly1d(coeffs[:, 0])
    fit_y = np.poly1d(coeffs[:, 1])

    return np.column_stack((fit_x(t_out), fit_y(t_out))), data['agent_future']


def fit_dct(data):
    hist_seq_len = np.sum(data['mask_past']).astype('uint8')
    past_flipped = np.flip(data['agent_past'][:hist_seq_len, :], axis=0)
    agent_path = np.concatenate((past_flipped, data['agent_future']), axis=0)
    # l_t = 1 - hist_seq_len
    # t = np.arange(l_t, 13)
    A = dct(agent_path, axis=0)
    agent_path_rec = idct(A[:int(agent_path.shape[0] / 2), :], axis=0, n=agent_path.shape[0])
    return agent_path_rec, agent_path


def fit_fft(data):
    hist_seq_len = np.sum(data['mask_past']).astype('uint8')
    past_flipped = np.flip(data['agent_past'][:hist_seq_len, :], axis=0)
    agent_path = np.concatenate((past_flipped, data['agent_future']), axis=0)
    # l_t = 1 - hist_seq_len
    # t = np.arange(l_t, 13)
    A = np.fft.rfft(agent_path, axis=0)
    agent_path_rec = np.fft.irfft(A, axis=0, n=agent_path.shape[0])
    return agent_path_rec, agent_path


hf = h5py.File('/scratch/rodney/datasets/nuScenes/processed/nuscenes-v1.0-trainval-train.h5', 'r')

rmse_l = []
for ind in range(len(hf.get('token').values())):
    data = get_dict(hf, ind)

    # regress_path, agent_path = fit_full_path(data)
    regress_path, agent_path = fit_full_path_ortho(data)
    # regress_path, agent_path = fit_history(data)
    rmse = np.sqrt(((agent_path - regress_path) ** 2).mean())
    print("{}/{} RMSE: {}".format(ind, len(hf.get('token').values()), rmse))
    rmse_l.append(rmse)
    #
    # plt.plot(regress_path[:, 0],
    #         regress_path[:, 1],
    #         'w--o',
    #         linewidth=4,
    #         markersize=3,
    #         zorder=650,
    #         path_effects=[pe.Stroke(linewidth=5, foreground='r'), pe.Normal()])
    # plt.plot(agent_path[:, 0],
    #          agent_path[:, 1],
    #          'w--o',
    #          linewidth=4,
    #          markersize=3,
    #          zorder=650,
    #          path_effects=[pe.Stroke(linewidth=5, foreground='b'), pe.Normal()])
    # plt.show()
    # exit()

print('Avg RMSE: {}'.format(np.mean(rmse_l)))
