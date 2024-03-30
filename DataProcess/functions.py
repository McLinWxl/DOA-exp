import numpy as np
import matplotlib.pyplot as plt


class GenSnapshot:
    def __init__(self, data: np.ndarray, sample_frequency, target_frequency, length_window, is_half_overlapping):
        """
        Generate frequency domain snapshots from time domain data
        :param data: Time domain data
        :param sample_frequency: Sample frequency
        :param target_frequency: Target frequency
        :param length_window: Length of the window
        :param is_half_overlapping: Whether the overlap is half
        :return: Frequency domain snapshots
        """
        self.data = data
        self.sample_frequency = sample_frequency
        self.target_frequency = target_frequency
        self.length_window = length_window
        self.is_half_overlapping = is_half_overlapping

    def get_snapshots(self, num_antennas, num_snapshots) -> np.ndarray:
        data_snp = self.gen_snapshot()
        num_samples = data_snp.shape[1] // num_snapshots
        data_snp = data_snp[:, :num_samples * num_snapshots]
        data_snp_sliced = np.zeros((num_samples, num_antennas, num_snapshots), dtype=np.complex64)
        for i in range(num_samples):
            data_snp_sliced[i] = data_snp[(16-num_antennas)//2: (16+num_antennas)//2, i * num_snapshots:(i + 1) * num_snapshots]
        return data_snp_sliced

    def gen_snapshot(self) -> np.ndarray:
        num_antennas, length_signal = self.data.shape
        match self.is_half_overlapping:
            case True:
                num_snapshots = length_signal // (self.length_window // 2) - 1
                stride = self.length_window // 2
                data_snapshots = np.zeros((num_antennas, num_snapshots), dtype=np.complex64)
                for i in range(num_snapshots):
                    for j in range(num_antennas):
                        #TODO: Find the maximum value of the fft
                        fft = np.fft.fft(self.data[j, i * stride:i * stride + self.length_window])
                        # plt.style.use(['science', 'ieee', 'grid'])
                        # plt.plot(fft[0:4000])
                        # plt.xlabel('Frequency')
                        # plt.ylabel('Amplitude')
                        # plt.savefig(f"../Test/fft.pdf")
                        # plt.show()
                        data_snapshots[j, i] = fft[self.target_frequency * self.length_window // self.sample_frequency + 1]
            case False:
                num_snapshots = length_signal // self.length_window
                stride = self.length_window
                data_snapshots = np.zeros((num_antennas, num_snapshots), dtype=np.complex64)
                for i in range(num_snapshots):
                    for j in range(num_antennas):
                        fft = np.fft.fft(self.data[j, i * stride:i * stride + self.length_window])
                        data_snapshots[j, i] = fft[self.target_frequency * self.length_window // self.sample_frequency + 1]
            case _:
                raise ValueError("is_half_overlapping should be either True or False")
        return self.norm_(data_snapshots)

    def norm_(self, input: np.ndarray) -> np.ndarray:
        """
        Normalize covariance matrix
        :param input:
        :return: Normalized covariance matrix
        """
        output = input / np.linalg.norm(input, axis=0, keepdims=True, ord=np.inf)
        return output


