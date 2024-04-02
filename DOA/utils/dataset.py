#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import numpy as np
from DataProcess.functions import GenSnapshot
from DOA.utils import model
from DOA.functions import Manifold_dictionary
import matplotlib.pyplot as plt
import torch
import os


def read_data(file_path) -> dict:
    files = os.listdir(file_path)
    data = {}
    for file in files:
        label = list(map(int, file.split('.')[0].split('_')))
        file_path = os.path.join(folder_name, file)
        data = np.load(file_path)
        data[label] = data
    return data


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, num_antennas, num_snps):
        self.data = data
        self.num_antennas = num_antennas
        self.num_snps = num_snps

    def split_data(self):
        data_splited = np.zeros((0, 0))
        label_splited = np.zeros((0, 0))
        for key in self.data.keys():
            data = self.data[key]
            data_snapshots_generator = GenSnapshot(data, 51200, 5666, 8192, target_fre_width=15, is_half_overlapping=True)
            data_snapshots = data_snapshots_generator.get_snapshots(num_antennas=self.num_antennas, num_snapshots=self.num_snps)
            data_splited = np.vstack((data_splited, data_snapshots))



if __name__ == "__main__":
    folder_name = '../../Data/ULA_0.03/S5666'
    a = read_data(folder_name)