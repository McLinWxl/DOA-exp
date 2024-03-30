#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.


import numpy as np
from DataProcess.functions import GenSnapshot
from DOA.utils import model
import matplotlib.pyplot as plt

model_name = 'MUSIC'
label = [-5, 0]

data = np.load('ULA_0.03/S5666/-5_0.npy')
data_snapshots_generator = GenSnapshot(data, 51200, 5666, 8192, True)
data_snapshots = data_snapshots_generator.get_snapshots(num_antennas=8, num_snapshots=256)
covariance_matrix = np.matmul(data_snapshots, data_snapshots.conj().transpose(0, 2, 1)) / data_snapshots.shape[2]
meshes = np.linspace(-60, 60, 121)
covariance_matrix = covariance_matrix[0]
plt.style.use(['science', 'ieee', 'grid'])
match model_name:
    case 'MUSIC':
        spect = model.MUSIC(covariance_matrix, num_antennas=8, num_sources=2, angle_meshes=meshes)
        plt.title('MUSIC Spectrum')
    case 'MVDR':
        spect = model.MVDR(covariance_matrix, num_antennas=8, angle_meshes=meshes)
        plt.title('MVDR Spectrum')
plt.plot(meshes, spect)
plt.axvline(x=label[0], color='r', linestyle='--')
plt.axvline(x=label[1], color='r', linestyle='--')
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.1)
plt.xlabel('Angle')
plt.ylabel('Spectrum')
plt.show()
print(data_snapshots.shape)
print(data.shape)