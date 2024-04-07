#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.


import numpy as np
from DataProcess.functions import GenSnapshot, denoise_covariance
from DOA.utils import model
from DOA.functions import Manifold_dictionary
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd
from rich.progress import track
import time

# Current time identifier
current_time = time.strftime('%m%d%H%M%S', time.localtime())

configs = {
    'name': 'AMI',
    'is_fine_tune': False,
    'num_antennas': 8,
    'num_snps': 200,
    'data_path': 'ULA_0.03/S2800/-5_5.npy',

    'sample_frequency': 51200,
    'target_frequency': 2800,
    'length_window': 8192,
    'target_fre_width': 0,
    'stride': 10,
    'mesh': np.linspace(-60, 60, 121),

    'antenna_interval': 0.03,

    'tar_sources': 2,
}

data_identifier = configs['data_path'].split('/')[-2]
if data_identifier in ['S5666', 'S2500', 'S2800']:
    label = configs['data_path'].split('/')[-1].split('.')[0].split('_')
elif data_identifier in ['I2535', 'Complete', 'Inner', 'Outer', 'Ball']:
    label1 = configs['data_path'].split('/')[-1].split('.')[0].split('_')[1]
    label2 = configs['data_path'].split('/')[-1].split('.')[0].split('_')[3]
    label = [label1, label2]
label = [float(i) for i in label]
speed_sound = 340
wave_length = speed_sound / configs['target_frequency']

data = np.load(f'../Data/{configs["data_path"]}')
data_snapshots_generator = GenSnapshot(data=data, sample_frequency=configs['sample_frequency'],
                                       target_frequency=configs['target_frequency'],
                                       length_window=configs['length_window'],
                                       target_fre_width=configs['target_fre_width'], is_half_overlapping=True)
data_snapshots = data_snapshots_generator.get_snapshots(num_antennas=configs['num_antennas'],
                                                        num_snapshots=configs['num_snps'], stride=configs['stride'])
covariance_matrix = np.matmul(data_snapshots, data_snapshots.conj().transpose(0, 2, 1)) / data_snapshots.shape[2]
meshes = configs['mesh']

covariance_matrix_denoised = denoise_covariance(covariance_matrix, num_sources=configs['tar_sources'])

cal_manifold = Manifold_dictionary(num_sensors=configs['num_antennas'], sensor_interval=configs['antenna_interval'],
                                   wavelength=wave_length, num_meshes=len(meshes), theta=meshes)
dictionary = cal_manifold.cal_dictionary()
length_sample = covariance_matrix.shape[0]
covariance_matrix_sample = covariance_matrix[1]
data_snapshots_sample = data_snapshots[1]

sample_length = data_snapshots.shape[0]

output_matrix = np.zeros((length_sample, len(meshes)))

covariance_vector_sample = covariance_matrix_sample.transpose(1, 0).reshape(configs['num_antennas'] ** 2, 1)

covariance_vector_denoised = covariance_matrix_denoised.transpose(0, 2, 1).reshape(length_sample,
                                                                                   configs['num_antennas'] ** 2, 1)

plt.style.use(['science', 'ieee', 'grid'])
# start time
start = time.time()
for i in track(range(sample_length)):
    match configs['name']:
        case 'MUSIC':
            spect = model.MUSIC(covariance_matrix[i], num_antennas=configs['num_antennas'], num_sources=configs['tar_sources'],
                                angle_meshes=meshes, antenna_intarvals=configs['antenna_interval'],
                                wavelength_source=wave_length)
            plt.title('MUSIC Spectrum')
            # plt.plot(meshes, spect)
            output_matrix[i] = spect.reshape(-1)
        case 'MVDR':
            spect = model.MVDR(covariance_matrix[i], num_antennas=configs['num_antennas'], angle_meshes=meshes,
                               antenna_intarvals=configs['antenna_interval'], wavelength_source=wave_length)
            plt.title('MVDR Spectrum')
            output_matrix[i] = spect.reshape(-1)
        case 'SBL':
            spect = model.SBL(raw_data=data_snapshots[i], num_antennas=configs['num_antennas'], angle_meshes=meshes,
                              antenna_intervals=configs['antenna_interval'], wavelength=wave_length, max_iteration=100,
                              error_threshold=1e-6)
            plt.title('SBL Spectrum')
            output_matrix[i] = spect.reshape(-1)
        case 'ISTA':
            spect = model.ISTA(covariance_array=covariance_vector_denoised[i], dictionary=dictionary,
                               angle_meshes=meshes, max_iter=100, tol=1e-6)
            plt.title('ISTA Spectrum')
            output_matrix[i] = spect.reshape(-1)
        case 'DCNN':
            model_path = '../Test/DCNN.pth'
            dictionary_torch = torch.from_numpy(dictionary).to(torch.complex64)
            covariance_vector_torch = torch.from_numpy(covariance_vector_denoised).to(torch.complex64)
            psedo_spectrum = torch.matmul(dictionary_torch.conj().transpose(1, 0), covariance_vector_torch)
            DCNN = model.DCNN()
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            DCNN.load_state_dict(state_dict['model'])
            DCNN.eval()
            num_samples, num_meshes, _ = psedo_spectrum.shape
            input_dcnn = torch.zeros((num_samples, num_meshes, 2))
            input_dcnn[:, :, 0] = psedo_spectrum.real.reshape(num_samples, num_meshes)
            input_dcnn[:, :, 1] = psedo_spectrum.imag.reshape(num_samples, num_meshes)
            out = DCNN(input_dcnn[i].reshape(1, num_meshes, 2))
            spect = out.detach().numpy().reshape(-1)
            plt.title('DCNN Spectrum')
            output_matrix[i] = spect.reshape(-1)

        case 'AMI':
            if configs['is_fine_tune']:
                model_path = '../Test/AMI_FT_040201.pth'
            else:
                model_path = '../Test/AMI-LF10.pth'
            dictionary_torch = torch.from_numpy(dictionary).to(torch.complex64)
            covariance_vector_torch = torch.from_numpy(covariance_vector_denoised).to(torch.complex64)
            AMI = model.AMI_LISTA(dictionary=dictionary_torch)
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            AMI.load_state_dict(state_dict['model'])
            AMI.eval()
            out, out_layers = AMI(covariance_vector_torch[i].reshape(-1, 1))
            spect = out.detach().numpy().reshape(-1)
            plt.title('AMI Spectrum')
            output_matrix[i] = spect.reshape(-1)
        case 'LISTA':
            model_path = '../Test/LISTA-10.pth'
            dictionary_torch = torch.from_numpy(dictionary).to(torch.complex64)
            covariance_vector_torch = torch.from_numpy(covariance_vector_denoised).to(torch.complex64).reshape(1, -1)
            LISTA = model.LISTA(dictionary=dictionary_torch)
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            LISTA.load_state_dict(state_dict['model'])
            LISTA.eval()
            out, out_layers = LISTA(covariance_vector_torch[i])
            spect = out.detach().numpy().reshape(-1)
            plt.title('LISTA Spectrum')
            output_matrix[i] = spect.reshape(-1)
        case 'LISTA_LF':
            model_path = '../Test/LISTA-LF10.pth'
            dictionary = torch.from_numpy(dictionary).to(torch.complex64)

            covariance_vector = torch.from_numpy(covariance_vector_denoised).to(torch.complex64).reshape(1, -1)
            LISTA = model.LISTA(dictionary=dictionary)
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            LISTA.load_state_dict(state_dict['model'])
            LISTA.eval()
            out, out_layers = LISTA(covariance_vector[i])
            spect = out.detach().numpy().reshape(-1)
            plt.title('LISTA-LF Spectrum')
            output_matrix[i] = spect.reshape(-1)
        case 'LISTA_AM':
            model_path = '../Test/AMI-10.pth'
            dictionary = torch.from_numpy(dictionary).to(torch.complex64)
            covariance_vector = torch.from_numpy(covariance_vector_denoised).to(torch.complex64).reshape(1, -1)
            AMI = model.AMI_LISTA(dictionary=dictionary)
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            AMI.load_state_dict(state_dict['model'])
            AMI.eval()
            out, out_layers = AMI(covariance_vector[i])
            spect = out.detach().numpy().reshape(-1)
            plt.title('LISTA-AM Spectrum')
            output_matrix[i] = spect.reshape(-1)
# end time
end = time.time()
time_cost = end - start
time_per_sample = time_cost / sample_length
print(f"Time cost: {time_cost}, Time per sample: {time_per_sample}")
# plt.axvline(x=label[0], color='r', linestyle='--')
# plt.axvline(x=label[1], color='r', linestyle='--')
# plt.matshow(output_matrix)
# plt.show()
plt.style.use(['science', 'ieee', 'grid'])
sns_dataFrame = pd.DataFrame(output_matrix.T, columns=[i for i in range(sample_length)],
                             index=[i - 60 for i in range(len(meshes))])
ax = sns.heatmap(sns_dataFrame, xticklabels=50, yticklabels=20, cmap='YlGnBu')
ax.tick_params(axis='both', which='both', direction='out', length=1, width=0.5)
sns.despine(top=False, right=False)
plt.xlabel('Samples')
plt.ylabel('Angle Meshes')
# plt.title(f'{model_name} Spectrum')
plt.grid(which='both', axis='both', linestyle='-', linewidth=0.1)
plt.savefig(f'../DOA/Figures/{configs['name']}_sample_spectrum_{current_time}.pdf')
plt.show()

#
plt.style.use(['science', 'ieee', 'grid'])
match configs['name']:
    case 'MUSIC':
        spect = model.MUSIC(covariance_matrix_sample, num_antennas=configs['num_antennas'], num_sources=configs['tar_sources'],
                            angle_meshes=meshes, antenna_intarvals=configs['antenna_interval'],
                            wavelength_source=wave_length)
        plt.title('MUSIC Spectrum')
    case 'MVDR':
        spect = model.MVDR(covariance_matrix_sample, num_antennas=configs['num_antennas'], angle_meshes=meshes,
                           antenna_intarvals=configs['antenna_interval'], wavelength_source=wave_length)
        plt.title('MVDR Spectrum')
    case 'SBL':
        spect = model.SBL(raw_data=data_snapshots_sample, num_antennas=configs['num_antennas'], angle_meshes=meshes,
                          antenna_intervals=configs['antenna_interval'], wavelength=wave_length, max_iteration=100,
                          error_threshold=1e-6)
        plt.title('SBL Spectrum')
    case 'ISTA':
        spect = model.ISTA(covariance_array=covariance_vector_sample, dictionary=dictionary, angle_meshes=meshes,
                           max_iter=100, tol=1e-6)
        plt.title('ISTA Spectrum')
    case 'AMI':
        if configs['is_fine_tune']:
            model_path = '../Test/AMI_FT_040201.pth'
        else:
            model_path = '../Test/AMI-LF10.pth'
        dictionary = torch.from_numpy(dictionary).to(torch.complex64)
        covariance_vector_sample = torch.from_numpy(covariance_vector_sample).to(torch.complex64).reshape(1, -1)
        AMI = model.AMI_LISTA(dictionary=dictionary)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        AMI.load_state_dict(state_dict['model'])
        AMI.eval()
        out, out_layers = AMI(covariance_vector_sample)
        spect = out.detach().numpy().reshape(-1)
        plt.title('AMI Spectrum')
    case 'LISTA':
        model_path = '../Test/LISTA-10.pth'
        dictionary = torch.from_numpy(dictionary).to(torch.complex64)
        covariance_vector_sample = torch.from_numpy(covariance_vector_sample).to(torch.complex64).reshape(1, -1)
        LISTA = model.LISTA(dictionary=dictionary)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        LISTA.load_state_dict(state_dict['model'])
        LISTA.eval()
        out, out_layers = LISTA(covariance_vector_sample)
        spect = out.detach().numpy().reshape(-1)
        plt.title('LISTA Spectrum')
    case 'LISTA_LF':
        model_path = '../Test/LISTA-LF10.pth'
        dictionary = torch.from_numpy(dictionary).to(torch.complex64)
        covariance_vector_sample = torch.from_numpy(covariance_vector_sample).to(torch.complex64).reshape(1, -1)
        LISTA = model.LISTA(dictionary=dictionary)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        LISTA.load_state_dict(state_dict['model'])
        LISTA.eval()
        out, out_layers = LISTA(covariance_vector_sample)
        spect = out.detach().numpy().reshape(-1)
        plt.title('LISTA-LF Spectrum')
    case 'LISTA_AM':
        model_path = '../Test/AMI-10.pth'
        dictionary = torch.from_numpy(dictionary).to(torch.complex64)
        covariance_vector_sample = torch.from_numpy(covariance_vector_sample).to(torch.complex64).reshape(1, -1)
        AMI = model.AMI_LISTA(dictionary=dictionary)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        AMI.load_state_dict(state_dict['model'])
        AMI.eval()
        out, out_layers = AMI(covariance_vector_sample)
        spect = out.detach().numpy().reshape(-1)
        plt.title('LISTA-AM Spectrum')
plt.plot(meshes, spect)
plt.axvline(x=label[0], color='r', linestyle='--')
plt.axvline(x=label[1], color='r', linestyle='--')
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.xlabel('Angle')
plt.ylabel('Spectrum')
plt.show()
print(data_snapshots.shape)
print(data.shape)
