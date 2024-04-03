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

model_name = 'AMI'
label = [-5, 0]
num_antennas = 8
num_snps = 256
data = np.load('ULA_0.03/S5666/-15_15.npy')
data_snapshots_generator = GenSnapshot(data, 51200, 5666, 8192, target_fre_width=15, is_half_overlapping=True)
data_snapshots = data_snapshots_generator.get_snapshots(num_antennas=num_antennas, num_snapshots=num_snps, stride=20)
covariance_matrix = np.matmul(data_snapshots, data_snapshots.conj().transpose(0, 2, 1)) / data_snapshots.shape[2]
meshes = np.linspace(-60, 60, 121)

covariance_matrix_denoised = denoise_covariance(covariance_matrix, num_sources=2)

cal_manifold = Manifold_dictionary(num_sensors=num_antennas, sensor_interval=0.03, wavelength=0.06, num_meshes=len(meshes), theta=meshes)
dictionary = cal_manifold.cal_dictionary()
length_sample = covariance_matrix.shape[0]
covariance_matrix_sample = covariance_matrix[6]
data_snapshots_sample = data_snapshots[6]

sample_length = data_snapshots.shape[0]

output_matrix = np.zeros((length_sample, len(meshes)))

covariance_vector_sample = covariance_matrix_sample.transpose(1, 0).reshape(num_antennas**2, 1)

covariance_vector_denoised = covariance_matrix_denoised.transpose(0, 2, 1).reshape(length_sample, num_antennas**2, 1)

plt.style.use(['science', 'ieee', 'grid'])
for i in range(sample_length):
    match model_name:
        case 'MUSIC':
            spect = model.MUSIC(covariance_matrix[i], num_antennas=num_antennas, num_sources=2, angle_meshes=meshes)
            plt.title('MUSIC Spectrum')
            plt.plot(meshes, spect)
            output_matrix[i] = spect.reshape(-1)
        case 'MVDR':
            spect = model.MVDR(covariance_matrix[i], num_antennas=num_antennas, angle_meshes=meshes)
            plt.title('MVDR Spectrum')
            output_matrix[i] = spect.reshape(-1)
        case 'SBL':
            spect = model.SBL(raw_data=data_snapshots[i], num_antennas=num_antennas, angle_meshes=meshes, max_iteration=1500, error_threshold=1e-3)
            plt.title('SBL Spectrum')
            output_matrix[i] = spect.reshape(-1)
        case 'ISTA':
            spect = model.ISTA(covariance_array=covariance_vector_denoised[i], dictionary=dictionary, angle_meshes=meshes, max_iter=1500, tol=1e-6)
            plt.title('ISTA Spectrum')
            output_matrix[i] = spect.reshape(-1)
        case 'AMI':
            model_path = '../Test/AMI_FT_040201.pth'
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

plt.axvline(x=label[0], color='r', linestyle='--')
plt.axvline(x=label[1], color='r', linestyle='--')
plt.matshow(output_matrix)
plt.show()
#
# plt.style.use(['science', 'ieee', 'grid'])
# match model_name:
#     case 'MUSIC':
#         spect = model.MUSIC(covariance_matrix_sample, num_antennas=num_antennas, num_sources=2, angle_meshes=meshes)
#         plt.title('MUSIC Spectrum')
#     case 'MVDR':
#         spect = model.MVDR(covariance_matrix_sample, num_antennas=num_antennas, angle_meshes=meshes)
#         plt.title('MVDR Spectrum')
#     case 'SBL':
#         spect = model.SBL(raw_data=data_snapshots_sample, num_antennas=num_antennas, angle_meshes=meshes, max_iteration=1500, error_threshold=1e-3)
#         plt.title('SBL Spectrum')
#     case 'ISTA':
#         spect = model.ISTA(covariance_array=covariance_vector_sample, dictionary=dictionary, angle_meshes=meshes, max_iter=1500, tol=1e-6)
#         plt.title('ISTA Spectrum')
#     case 'AMI':
#         model_path = '../Test/AMI_FT_040201.pth'
#         dictionary = torch.from_numpy(dictionary).to(torch.complex64)
#         covariance_vector_sample = torch.from_numpy(covariance_vector_sample).to(torch.complex64).reshape(1, -1)
#         AMI = model.AMI_LISTA(dictionary=dictionary)
#         state_dict = torch.load(model_path, map_location=torch.device("cpu"))
#         AMI.load_state_dict(state_dict['model'])
#         AMI.eval()
#         out, out_layers = AMI(covariance_vector_sample)
#         spect = out.detach().numpy().reshape(-1)
#         plt.title('AMI Spectrum')
#     case 'LISTA':
#         model_path = '../Test/LISTA-10.pth'
#         dictionary = torch.from_numpy(dictionary).to(torch.complex64)
#         covariance_vector_sample = torch.from_numpy(covariance_vector_sample).to(torch.complex64).reshape(1, -1)
#         LISTA = model.LISTA(dictionary=dictionary)
#         state_dict = torch.load(model_path, map_location=torch.device("cpu"))
#         LISTA.load_state_dict(state_dict['model'])
#         LISTA.eval()
#         out, out_layers = LISTA(covariance_vector_sample)
#         spect = out.detach().numpy().reshape(-1)
#         plt.title('LISTA Spectrum')
#     case 'LISTA_LF':
#         model_path = '../Test/LISTA-LF10.pth'
#         dictionary = torch.from_numpy(dictionary).to(torch.complex64)
#         covariance_vector_sample = torch.from_numpy(covariance_vector_sample).to(torch.complex64).reshape(1, -1)
#         LISTA = model.LISTA(dictionary=dictionary)
#         state_dict = torch.load(model_path, map_location=torch.device("cpu"))
#         LISTA.load_state_dict(state_dict['model'])
#         LISTA.eval()
#         out, out_layers = LISTA(covariance_vector_sample)
#         spect = out.detach().numpy().reshape(-1)
#         plt.title('LISTA-LF Spectrum')
#     case 'LISTA_AM':
#         model_path = '../Test/AMI-10.pth'
#         dictionary = torch.from_numpy(dictionary).to(torch.complex64)
#         covariance_vector_sample = torch.from_numpy(covariance_vector_sample).to(torch.complex64).reshape(1, -1)
#         AMI = model.AMI_LISTA(dictionary=dictionary)
#         state_dict = torch.load(model_path, map_location=torch.device("cpu"))
#         AMI.load_state_dict(state_dict['model'])
#         AMI.eval()
#         out, out_layers = AMI(covariance_vector_sample)
#         spect = out.detach().numpy().reshape(-1)
#         plt.title('LISTA-AM Spectrum')
# plt.plot(meshes, spect)
# plt.axvline(x=label[0], color='r', linestyle='--')
# plt.axvline(x=label[1], color='r', linestyle='--')
# plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
# plt.xlabel('Angle')
# plt.ylabel('Spectrum')
# plt.show()
# print(data_snapshots.shape)
# print(data.shape)
