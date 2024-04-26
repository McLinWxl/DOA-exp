#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import os
import sys
import seaborn as sns
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from DOA.utils.dataset import MyDataset
from DataProcess.functions import denoise_covariance
import torch
import numpy as np
from DOA.utils.model import AMI_LISTA, MUSIC, MVDR, SBL, ISTA, DCNN
from DOA.functions import Manifold_dictionary, find_peak
from matplotlib import pyplot as plt
from rich.progress import track

configs = {
    'name': 'AMI',
    'is_fine_tune': True, # True, False
    'num_antennas': 8,
    'num_snps': 256,
    'batch_size': 1,
    'num_sources': 2,
    'stride': 1,
    'mesh': np.linspace(-60, 60, 121),
    'plot_style': 'spectrum', # line, spectrum
    'figure_path': '../Figures/',
    'angle_interval': 0.03,
    'wavelength': 0.06
}

if os.path.exists(configs['figure_path']) is False:
    os.makedirs(configs['figure_path'])

cal_manifold = Manifold_dictionary(num_sensors=configs['num_antennas'], sensor_interval=0.03, wavelength=0.06, num_meshes=len(configs['mesh']), theta=configs['mesh'])
dictionary_np = cal_manifold.cal_dictionary()
dictionary = torch.from_numpy(dictionary_np).to(torch.complex64)

folded_path = 'ULA_0.03/S5666'
abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'Data', folded_path))

mydata = MyDataset(folder_path=abs_path, num_antennas=configs['num_antennas'], num_snps=configs['num_snps'], stride=configs['stride'])
test_loader = torch.utils.data.DataLoader(mydata, batch_size=configs['batch_size'], shuffle=False)

# model_path = './AMI-LF10.pth'
if configs['is_fine_tune']:
    model_path = './AMI_FT_040201.pth'
else:
    model_path = './AMI-LF10.pth'
abs_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'Test', model_path))

model = AMI_LISTA(dictionary=dictionary)
state_dict = torch.load(abs_model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict['model'])
model.eval()
num_samples = len(test_loader)

output_DOA, label_DOA = np.zeros((num_samples, 2)), np.zeros((num_samples, 2))
output_spectrum = np.zeros((num_samples, len(configs['mesh']), 1))

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
                output = MUSIC(covariance_matrix[i], num_antennas=configs['num_antennas'], num_sources=configs['num_sources'], angle_meshes=configs['mesh'], antenna_intarvals=configs['angle_interval'], wavelength_source=configs['wavelength'])
                output_detached[i] = output
        case 'MVDR':
            covariance_matrix = covariance_matrix.detach().cpu().numpy()
            output_detached = np.zeros((configs['batch_size'], len(configs['mesh']), 1))
            for i in range(configs['batch_size']):
                output = MVDR(covariance_matrix[i], num_antennas=configs['num_antennas'], angle_meshes=configs['mesh'], antenna_intarvals=configs['angle_interval'], wavelength_source=configs['wavelength'])
                output_detached[i] = output
        case 'SBL':
            data = data.detach().cpu().numpy()
            output_detached = np.zeros((configs['batch_size'], len(configs['mesh']), 1))
            for i in range(configs['batch_size']):
                output = SBL(raw_data=data[i], num_antennas=configs['num_antennas'], angle_meshes=configs['mesh'], max_iteration=100, error_threshold=1e-6, antenna_intervals=configs['angle_interval'], wavelength=configs['wavelength'])
                output_detached[i] = output.reshape(-1, 1)
        case 'ISTA':
            covariance_matrix = denoise_covariance(covariance_matrix, num_sources=configs['num_sources'])
            covariance_vector = covariance_matrix.transpose(0, 2, 1).reshape(-1, configs['num_antennas']**2, 1)
            output_detached = np.zeros((configs['batch_size'], len(configs['mesh']), 1))
            for i in range(configs['batch_size']):
                output = ISTA(covariance_vector[i], dictionary_np, angle_meshes=configs['mesh'], max_iter=100, tol=1e-6)
                output_detached[i] = output
        case 'DCNN':
            model_path = '/Volumes/WangXinLin/GitLibrary/DOA-exp/Test/DCNN.pth'
            covariance_matrix = covariance_matrix.detach().cpu().numpy()
            covariance_matrix_denoised = denoise_covariance(covariance_matrix, num_sources=configs['num_sources'])
            covariance_vector_denoised = covariance_matrix_denoised.transpose(0, 2, 1).reshape(-1, configs['num_antennas']**2, 1)
            dictionary_torch = dictionary
            covariance_vector_torch = torch.from_numpy(covariance_vector_denoised).to(torch.complex64)
            psedo_spectrum = torch.matmul(dictionary_torch.conj().transpose(1, 0), covariance_vector_torch)
            DCNN_model = DCNN()
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            DCNN_model.load_state_dict(state_dict['model'])
            DCNN_model.eval()
            num_samples_, num_meshes, _ = psedo_spectrum.shape
            input_dcnn = torch.zeros((num_samples_, num_meshes, 2))
            input_dcnn[:, :, 0] = psedo_spectrum.real.reshape(num_samples_, num_meshes)
            input_dcnn[:, :, 1] = psedo_spectrum.imag.reshape(num_samples_, num_meshes)
            out = DCNN_model(input_dcnn.reshape(1, num_meshes, 2))
            output_detached = out.detach().cpu().numpy()
    out_peak = find_peak(output_detached, num_sources=2, start_bias=60, is_insert=True)
    output_DOA[idx] = out_peak.reshape(-1)
    label_DOA[idx] = label.detach().cpu().numpy()
    output_spectrum[idx] = output_detached
# make label_DOAin ascending order in both two columns
order_first_colum = np.argsort(label_DOA[:, 0])
label_DOA = label_DOA[order_first_colum]
output_DOA = output_DOA[order_first_colum]
output_spectrum = output_spectrum[order_first_colum]

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
    output_spectrum[starts[i]:ends[i]+1] = output_spectrum[starts[i]:ends[i]+1][order1]

# Calculate the NMSE
error = 0
correct = 0
for idx in range(num_samples):
    for k in range(2):
        id_side: int = np.abs(k-1)
        error_item = np.square(output_DOA[idx, k] - label_DOA[idx, id_side])
        error += error_item
        if np.abs(output_DOA[idx, k] - label_DOA[idx, id_side]) < 4.5:
            correct += 1
RMSE = np.sqrt(error / (2*num_samples))
ACC = correct / (2*num_samples)

print(f'NMSE: {RMSE}  ;ACC: {ACC}')

# set dpi
plt.rcParams['figure.dpi'] = 500

plt.style.use(['science', 'ieee', 'grid'])
plt.plot(label_DOA[:, 0], label='Label DOA of Two Sources', linestyle='-', color='black')
plt.plot(label_DOA[:, 1], linestyle='-', color='black')
plt.plot(output_DOA[:, 0], label='Estimated DOA of Source 1', linestyle='-', linewidth=0.6, color='red')
plt.plot(output_DOA[:, 1], label='Estimated DOA of Source 2', linestyle='-', linewidth=0.6, color='blue')
plt.ylim(-40, 40)
if configs["name"] == 'AMI':
    if configs['is_fine_tune']:
        plt.title('AMI-LISTA (Fine-tuned)')
    else:
        plt.title('AMI-LISTA')
else:
    plt.title(f'{configs["name"]}')
plt.xlabel('Samples')
plt.ylabel('Angle Meshes')
plt.legend(loc='upper left', prop={'size': 5})
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.savefig(f'{configs["figure_path"]}/{configs["name"]}_DOA.pdf')
plt.show()

plt.style.use(['science', 'ieee', 'grid'])
# sns.set_context('paper')
sns_dataFrame = pd.DataFrame(output_spectrum.reshape(-1, len(configs['mesh'])).T, columns=[i for i in range(num_samples)], index=[i-60 for i in range(len(configs['mesh']))])
ax = sns.heatmap(sns_dataFrame, xticklabels=100, yticklabels=20, cmap='YlGnBu')
ax.tick_params(axis='both', which='both', direction='out', length=1, width=0.5)
sns.despine(top=False, right=False)
sns.despine(top=False, right=False)
plt.xlabel('Samples')
plt.ylabel('Angle Meshes')
if configs["name"] == 'AMI':
    plt.title('AMI-LISTA Spectrum')
else:
    plt.title(f'{configs["name"]} Spectrum')
plt.grid(which='both', axis='both', linestyle='-', linewidth=0.1)
plt.savefig(f'{configs["figure_path"]}/{configs["name"]}_Spectrum.pdf')
plt.show()




