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
from DOA.functions import Manifold_dictionary, find_peak, DoA2Spect
from matplotlib import pyplot as plt

# Set seeds
torch.manual_seed(3407)
np.random.seed(3407)

configs = {
    'name': 'AMI',
    'num_antennas': 8,
    'num_snps': 256,
    'num_epochs': 1350,
    'batch_size': 32,
    'num_sources': 2,
    'stride': 100,
    'num_layers': 10,
    'mesh': np.linspace(-60, 60, 121),
    'model_path': '../../Test',
}

cal_manifold = Manifold_dictionary(num_sensors=configs['num_antennas'], sensor_interval=0.03, wavelength=0.06, num_meshes=len(configs['mesh']), theta=configs['mesh'])
dictionary = cal_manifold.cal_dictionary()
dictionary = torch.from_numpy(dictionary).to(torch.complex64)

folded_path = '../../Data/ULA_0.03/S5666'

mydata = MyDataset(folder_path=folded_path, num_antennas=configs['num_antennas'], num_snps=configs['num_snps'], stride=configs['stride'])
train_set, test_set = torch.utils.data.random_split(mydata, [int(0.5*len(mydata)), len(mydata)-int(0.5*len(mydata))])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=configs['batch_size'], shuffle=True, drop_last=True)

model_path = '../../Test/AMI-LF10.pth'

model = AMI_LISTA(dictionary=dictionary)
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict['model'])
num_samples = len(train_loader)
output_DOA, label_DOA = np.zeros((num_samples, 2)), np.zeros((num_samples, 2))

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
loss = torch.nn.MSELoss()

best_loss = 100
for epoch in range(configs['num_epochs']+1):
    for idx, (data, label) in enumerate(train_loader):
        mse_loss = 0
        train_loss = 0
        # remove inf and nan in the data
        data = torch.nan_to_num(data)
        covariance_matrix = torch.matmul(data, data.transpose(1, 2).conj())
        covariance_matrix = denoise_covariance(covariance_matrix, num_sources=configs['num_sources'])
        covariance_vector = covariance_matrix.transpose(0, 2, 1).reshape(-1, configs['num_antennas']**2, 1)
        covariance_vector = torch.from_numpy(covariance_vector).to(torch.complex64)
        label_spectrum = DoA2Spect(label, num_sources=configs['num_sources'], num_meshes=len(configs['mesh']))
        label_spectrum = label_spectrum / np.sqrt(configs['num_sources'])
        label_spectrum = torch.from_numpy(label_spectrum).to(torch.float32)
        output, layers_output = model(covariance_vector)

        optimizer.zero_grad()
        # for i in range(configs['num_layers']):
        #     mse_loss = mse_loss + (loss(layers_output[:, i].to(torch.float32), label_spectrum.to(torch.float32)))
        mse_loss = loss(output.to(torch.float32), label_spectrum.to(torch.float32))
        mse_loss.backward()
        optimizer.step()
        train_loss += mse_loss.item()
    print(f"Epoch: {epoch}, Loss: {train_loss/len(train_loader)}")

    if train_loss < best_loss:
        torch.save({'model': model.state_dict()}, f"{configs['model_path']}/AMI_FT_best.pth")
        best_loss = train_loss






