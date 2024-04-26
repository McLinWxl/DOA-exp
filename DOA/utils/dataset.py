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


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folder_path, num_antennas, num_snps, **kwargs):
        self.num_antennas = num_antennas
        self.num_snps = num_snps
        self.fre_sample = kwargs.get('fre_sample', 51200)
        self.target_fre = kwargs.get('target_fre', 5666)
        self.length_window = kwargs.get('length_window', 8192)
        self.target_fre_width = kwargs.get('target_fre_width', 0)
        self.is_half_overlapping = kwargs.get('is_half_overlapping', True)
        self.stride = kwargs.get('stride', 100)
        self.num_sources = kwargs.get('num_sources', 2)
        self.folder_path = folder_path
        self.is_saved = kwargs.get('is_saved', True)

        self.data, self.label = self.split_data(saved=self.is_saved, path_save=self.folder_path)

    def make_snapshots_data(self, file_path) -> tuple:
        files = os.listdir(file_path)
        data_all = np.zeros((0, 0))
        label_all = np.zeros((0, 0))
        identity = file_path.split('/')[-1]
        for idx, file in enumerate(files):
            # DEBUG
            if identity in ['Complete', 'Inner', 'Outer', 'Ball', 'I2535']:
                label = [file.split('.')[0].split('_')[1], file.split('.')[0].split('_')[3]]
                label = list(map(int, label))
            elif identity == 'S5666':
                label = list(map(int, file.split('.')[0].split('_')))
            else:
                raise ValueError('The file name is not correct!')
            file_path = os.path.join(folder_name, file)
            data = np.load(file_path)
            snp_gen = GenSnapshot(data, self.fre_sample, self.target_fre, self.length_window,
                                  target_fre_width=self.target_fre_width, is_half_overlapping=self.is_half_overlapping)
            data_snp = snp_gen.get_snapshots(num_antennas=self.num_antennas, num_snapshots=self.num_snps,
                                             stride=self.stride)
            if idx == 0:
                data_all = data_snp
                label = np.array(label).reshape(-1, self.num_sources).repeat(data_snp.shape[0], axis=0)
                label_all = label
            else:
                data_all = np.vstack((data_all, data_snp))
                label = np.array(label).reshape(-1, self.num_sources).repeat(data_snp.shape[0], axis=0)
                label_all = np.vstack((label_all, label))
            # release memory
            del data, data_snp
        return data_all, label_all


    def split_data(self, saved, **kwargs):
        path_save = kwargs.get('path_save', '../../Data/ULA_0.03/S5666')
        if not saved:
            data_, label_ = self.make_snapshots_data(folder_name)
            # save data to npy file
            data_save = {
                'data': data_,
                'label': label_
            }
            np.save(os.path.join(path_save, 'data_snp_4372.npy'), data_save)
        else:
            data_save = np.load(os.path.join(path_save, 'data_snp.npy'), allow_pickle=True)
            data_ = data_save.item().get('data')
            label_ = data_save.item().get('label')
        return data_, label_

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    folder_name = '../../Data/ULA_0.03/S5666'
    mydata = MyDataset(folder_name,
                       num_antennas=8,
                       num_snps=256,
                       target_fre=5666,
                       is_saved=False,
                       stride=128)
