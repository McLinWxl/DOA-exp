#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import matplotlib.pyplot as plt
from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track


def load_data(file_path, remove_start=0, remove_end=0) -> np.ndarray:
    data_all = np.zeros((0, 0))
    with TdmsFile.open(file_path) as tdms_file:
        for _, group in track(enumerate(tdms_file.groups())):
            group_name = group.name
            # print(f'Group Name: {group_name}')
            for idx, channel in enumerate(group.channels()):
                channel_name = channel.name
                # print(f'Channel Name: {channel_name}')
                data = channel[:]
                if idx == 0:
                    data_all = data
                    plt.style.use(['science', 'ieee', 'grid'])
                    plt.plot(data, linewidth=0.5)
                    plt.xlabel('Sample')
                    plt.ylabel('Amplitude')
                    plt.savefig(f"../Test/spect.pdf")
                    plt.show()
                else:
                    data_all = np.vstack((data_all, data))
    print(f"Data shape: {data_all.shape}")
    return data_all


if __name__ == "__main__":
    f_s = 51200
    # time =
    f_path = "../../../328Data/test_20240329_140658-RealData/2024-03-29 14-06-59.tdms"
    data = load_data(f_path)
    # save data to npy file
    np.save("./ULA_0.03/S5666/-20_10.npy", data)

