import numpy as np
import matplotlib.pyplot as plt
import heapq


def norm_(input: np.ndarray) -> np.ndarray:
    """
    Normalize covariance matrix (num_antennas, num_snapshots)
    :param input:
    :return: Normalized covariance matrix
    """
    assert input.ndim == 2
    output = input / np.linalg.norm(input, axis=0, keepdims=True, ord=np.inf)
    return output


class GenSnapshot:
    def __init__(self, data: np.ndarray, sample_frequency, target_frequency, length_window, **kwargs):
        """
        Generate frequency domain snapshots from time domain data
        :param data: Time domain data
        :param sample_frequency: Sample frequency
        :param target_frequency: Target frequency
        :param target_fre_width:
        :param length_window: Length of the window
        :param is_half_overlapping: Whether the overlap is half
        :return: Frequency domain snapshots
        """
        self.data = data[:, sample_frequency*10:-sample_frequency*10]
        self.sample_frequency = sample_frequency
        self.target_frequency = target_frequency
        self.length_window = length_window
        self.target_fre_width = kwargs.get('target_fre_width', 10)
        self.is_half_overlapping = kwargs.get('is_half_overlapping', True)

    def get_snapshots(self, num_antennas: int, num_snapshots: int, stride: int) -> np.ndarray:
        data_snp = self.gen_snapshot()
        num_samples = (data_snp.shape[1] - num_snapshots) // stride + 1
        data_snp = data_snp[:, :num_samples * num_snapshots]
        data_snp_sliced = np.zeros((num_samples, num_antennas, num_snapshots), dtype=np.complex64)
        for i in range(num_samples):
            data_snp_window = data_snp[(16-num_antennas)//2: (16+num_antennas)//2, i * stride:i * stride + num_snapshots]
            data_snp_sliced[i] = norm_(data_snp_window)
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
                        # if i == 0:
                        #     plt.style.use(['science', 'ieee', 'grid'])
                        #     length_fft = fft.shape[0]
                        #     x_label = np.arange(0, length_fft) * 51200 / 8192
                        #     plt.plot(x_label[20:length_fft//4], fft[20:length_fft//4])
                        #     plt.xlabel('Frequency')
                        #     plt.ylabel('Amplitude')
                        #     plt.savefig(f"/Volumes/WangXinLin/GitLibrary/DOA-exp/Test/Figures/fft{j}.pdf")
                        #     plt.show()
                        #
                        #     plt.plot(self.data[j])
                        #     plt.xlabel('Sample')
                        #     plt.ylabel('Amplitude')
                        #     plt.savefig(f"/Volumes/WangXinLin/GitLibrary/DOA-exp/Test/Figures/data{j}.pdf")
                        #     plt.show()
                        target_frequency_dft = self.target_frequency * self.length_window // self.sample_frequency + 1
                        if self.target_fre_width == 0:
                            data_snapshots[j, i] = fft[target_frequency_dft]
                        else:
                            highest_idx = np.argmax((fft[target_frequency_dft - self.target_fre_width:target_frequency_dft + self.target_fre_width]))
                            data_snapshots[j, i] = fft[target_frequency_dft - self.target_fre_width + highest_idx]
            case False:
                num_snapshots = length_signal // self.length_window
                stride = self.length_window
                data_snapshots = np.zeros((num_antennas, num_snapshots), dtype=np.complex64)
                for i in range(num_snapshots):
                    for j in range(num_antennas):
                        fft = np.fft.fft(self.data[j, i * stride:i * stride + self.length_window])
                        target_frequency_dft = self.target_frequency * self.length_window // self.sample_frequency + 1
                        highest_idx = np.argmax((fft[target_frequency_dft - self.target_fre_width:target_frequency_dft + self.target_fre_width]))
                        data_snapshots[j, i] = fft[target_frequency_dft - self.target_fre_width + highest_idx]
            case _:
                raise ValueError("is_half_overlapping should be either True or False")
        a = norm_(data_snapshots)
        return norm_(data_snapshots)


def denoise_covariance(covariance_matrix, num_sources):
    """
    Minus the noise variance (estimated by the smallest eigenvalue) from the covariance matrix.
    :param covariance_matrix:
    :param num_sources:
    :return: Denoised covariance matrix
    """
    nums, M, M = covariance_matrix.shape
    covariance_matrix_clean = np.zeros((nums, M, M)) + 1j * np.zeros((nums, M, M))
    for i in range(nums):
        eigvalue = np.linalg.eigvals(covariance_matrix[i])
        smallest_eigvalue = heapq.nsmallest(int(M - num_sources), eigvalue)
        noise_variance = np.mean(smallest_eigvalue)
        noise_matrix = noise_variance * np.eye(M)
        covariance_matrix_clean[i] = covariance_matrix[i] - noise_matrix
    return covariance_matrix_clean


