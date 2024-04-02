#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from DOA.utils.dataset import MyDataset
from DataProcess.functions import denoise_covariance
import torch
import numpy as np
from DOA.utils.model import AMI_LISTA, MUSIC, MVDR, SBL, ISTA
from DOA.functions import Manifold_dictionary, find_peak
from matplotlib import pyplot as plt

configs = {
    'name': 'AMI',
    'num_antennas': 8,
    'num_snps': 256,
    'batch_size': 1,
    'num_sources': 2,
    'stride': 1,
    'mesh': np.linspace(-60, 60, 121),
}

cal_manifold = Manifold_dictionary(num_sensors=configs['num_antennas'], sensor_interval=0.03, wavelength=0.06, num_meshes=len(configs['mesh']), theta=configs['mesh'])
dictionary = cal_manifold.cal_dictionary()
dictionary = torch.from_numpy(dictionary).to(torch.complex64)

folded_path = 'ULA_0.03/S5666'
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'Data', folded_path))

mydata = MyDataset(folder_path=abs_path, num_antennas=configs['num_antennas'], num_snps=configs['num_snps'], stride=configs['stride'])
test_loader = torch.utils.data.DataLoader(mydata, batch_size=configs['batch_size'], shuffle=False)

# model_path = './AMI-LF10.pth'
model_path = './AMI_FT_70.pth'
abs_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'Test', model_path))

model = AMI_LISTA(dictionary=dictionary)
state_dict = torch.load(abs_model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict['model'])
model.eval()
num_samples = len(test_loader)
output_DOA, label_DOA = np.zeros((num_samples, 2)), np.zeros((num_samples, 2))

for idx, (data, label) in enumerate(test_loader):
    data = torch.nan_to_num(data)
    covariance_matrix = torch.matmul(data, data.transpose(1, 2).conj())
    match configs['name']:
        case 'AMI':

            covariance_matrix = denoise_covariance(covariance_matrix, num_sources=configs['num_sources'])
            covariance_vector = covariance_matrix.transpose(0, 2, 1).reshape(-1, configs['num_antennas']**2, 1)
            covariance_vector = torch.from_numpy(covariance_vector).to(torch.complex64)
            label = label.squeeze(0)
            output, output_layers = model(covariance_vector)
            output_detached = output.detach().cpu().numpy()
        case 'MUSIC':
            covariance_matrix = covariance_matrix.detach().cpu().numpy()
            output_detached = np.zeros((configs['batch_size'], len(configs['mesh']), 1))
            for i in range(configs['batch_size']):
                output = MUSIC(covariance_matrix[i], num_antennas=configs['num_antennas'], num_sources=configs['num_sources'], angle_meshes=configs['mesh'])
                output_detached[i] = output
        case 'MVDR':
            covariance_matrix = covariance_matrix.detach().cpu().numpy()
            output_detached = np.zeros((configs['batch_size'], len(configs['mesh']), 1))
            for i in range(configs['batch_size']):
                output = MVDR(covariance_matrix[i], num_antennas=configs['num_antennas'], angle_meshes=configs['mesh'])
                output_detached[i] = output
        case 'SBL':
            data = data.detach().cpu().numpy()
            output_detached = np.zeros((configs['batch_size'], len(configs['mesh']), 1))
            for i in range(configs['batch_size']):
                output = SBL(raw_data=data[i], num_antennas=configs['num_antennas'], angle_meshes=configs['mesh'], max_iteration=1500, error_threshold=1e-3)
                output_detached[i] = output.reshape(-1, 1)
        case 'ISTA':
            covariance_matrix = denoise_covariance(covariance_matrix, num_sources=configs['num_sources'])
            covariance_vector = covariance_matrix.transpose(0, 2, 1).reshape(-1, configs['num_antennas']**2, 1)
            covariance_vector = torch.from_numpy(covariance_vector).to(torch.complex64)
            output_detached = np.zeros((configs['batch_size'], len(configs['mesh']), 1))
            for i in range(configs['batch_size']):
                output = ISTA(covariance_vector[i], dictionary, angle_meshes=configs['mesh'], max_iter=1500, tol=1e-6)
                output_detached[i] = output
    out_peak = find_peak(output_detached, num_sources=2, start_bias=60, is_insert=True)
    output_DOA[idx] = out_peak.reshape(-1)
    label_DOA[idx] = label.detach().cpu().numpy()
# make label_DOAin ascending order in both two columns
order_first_colum = np.argsort(label_DOA[:, 0])
label_DOA = label_DOA[order_first_colum]
output_DOA = output_DOA[order_first_colum]

start = label_DOA[0, 0]
starts = [0]
ends = []
for i in range(label_DOA.shape[0]):
    if label_DOA[i, 0] != label_DOA[i-1, 0] and i > 0:
        starts.append(i)
        ends.append(i-1)
ends.append(label_DOA.shape[0]-1)

num_groups = len(starts)
for i in range(num_groups):
    order1 = np.argsort(label_DOA[starts[i]:ends[i]+1, 1])
    label_DOA[starts[i]:ends[i]+1] = label_DOA[starts[i]:ends[i]+1][order1]
    output_DOA[starts[i]:ends[i]+1] = output_DOA[starts[i]:ends[i]+1][order1]

plt.style.use(['science', 'ieee', 'grid'])
plt.plot(label_DOA[:, 0], label='Label DOA 1', linestyle='-', color='black')
plt.plot(label_DOA[:, 1], label='Label DOA 2', linestyle='-', color='black')
plt.plot(output_DOA[:, 0], label='Output DOA 1', linestyle='-', linewidth=0.6, color='red')
plt.plot(output_DOA[:, 1], label='Output DOA 2', linestyle='-', linewidth=0.6, color='blue')
plt.ylim(-30, 30)
plt.title(f'{configs["name"]}')
plt.legend(loc='upper left', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.show()


