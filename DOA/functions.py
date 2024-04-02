import numpy as np
import scipy.signal
import heapq


class Manifold_dictionary:
    def __init__(self, num_sensors, sensor_interval, wavelength, num_meshes, theta):
        self.num_sensors = num_sensors
        self.sensor_interval = sensor_interval
        self.wavelength = wavelength
        self.num_meshes = num_meshes
        self.theta = theta
        self.manifold = self.cal_manifold(num_sensors, sensor_interval, wavelength)

    def cal_dictionary(self):
        dictionary = np.zeros((self.num_sensors ** 2, self.num_meshes), dtype=np.complex64)
        w_m = np.zeros((self.num_sensors, 121)) + 1j * np.zeros((self.num_sensors, 121))
        for i in range(self.num_sensors):
            # s = numpy.exp(-1j * numpy.pi * 2 * self.sensor_interval * i * numpy.sin(numpy.deg2rad(self.theta)) / self.wavelength)
            # B = numpy.diag(s)
            # phi = numpy.matmul(self.manifold, B)
            for j in range(self.num_meshes):
                steer_vec = self.manifold[:, j].reshape(-1, 1)
                steer_map = np.matmul(steer_vec, steer_vec.T.conjugate())
                w_m[:, j] = steer_map[:, i]
            dictionary[i * self.num_sensors:(i + 1) * self.num_sensors, :] = w_m
        return dictionary

    def cal_manifold(self, num_sensors, sensor_interval, wavelength):
        return np.exp(1j * np.pi * 2 * sensor_interval * np.arange(num_sensors)[:, np.newaxis] * np.sin(
            np.deg2rad(self.theta)) / wavelength)


def find_peak(spectrum, num_sources=2, start_bias=60, is_insert=True):
    numTest, num_mesh, _ = spectrum.shape
    angles = np.zeros((num_sources, numTest))
    for num in range(numTest):
        # li = spectrum[num, :].reshape(-1)
        if is_insert:
            angle = Spect2DoA(spectrum[num, :].reshape(1, num_mesh, 1), num_sources=num_sources,
                              start_bias=start_bias)
        else:
            angle = Spect2DoA_no_insert(spectrum[num, :].reshape(1, num_mesh, 1), num_sources=num_sources,
                                        start_bias=start_bias)
        angles[:, num] = angle.reshape(-1)
    return np.sort(angles, axis=0)[::-1]


def Spect2DoA(Spectrum, num_sources=2, start_bias=60):
    """
    :param Spectrum: (num_samples, num_meshes, 1)
    :param num_sources:
    :param height_ignore:
    :param start_bias:
    :return: (num_samples, num_sources)
    """
    num_samples, num_meshes, _ = Spectrum.shape
    angles = np.zeros((num_samples, num_sources))
    for num in range(num_samples):
        li_0 = Spectrum[num, :].reshape(-1)
        # li_0[li_0 < 0] = 0
        li = li_0
        angle = np.zeros(num_sources) - 5
        peaks_idx = np.zeros(num_sources)
        grids_mesh = np.arange(num_meshes) - start_bias
        peaks, _ = scipy.signal.find_peaks(li)
        max_spectrum = heapq.nlargest(num_sources, li[peaks])
        for i in range(len(max_spectrum)):
            peaks_idx[i] = np.where(li == max_spectrum[i])[0][0]
            angle[i] = (
                li[int(peaks_idx[i] + 1)] / (li[int(peaks_idx[i] + 1)]
                                             + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i] + 1)]
                + li[int(peaks_idx[i])] / (li[int(peaks_idx[i] + 1)]
                                           + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i])]
                if li[int(peaks_idx[i] - 1)] < li[int(peaks_idx[i] + 1)]
                else li[int(peaks_idx[i] - 1)] / (li[int(peaks_idx[i] - 1)]
                                                  + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i] - 1)]
                     + li[int(peaks_idx[i])] / (li[int(peaks_idx[i] - 1)]
                                                + li[int(peaks_idx[i])]) * grids_mesh[int(peaks_idx[i])]
            )
        angles[num] = angle.reshape(-1)
    return np.sort(angles, axis=1)[::-1]


def Spect2DoA_no_insert(Spectrum, num_sources=2, start_bias=60):
    """
    :param Spectrum: (num_samples, num_meshes, 1)
    :param num_sources:
    :param height_ignore:
    :param start_bias:
    :return: (num_samples, num_sources)
    """
    num_samples, num_meshes, _ = Spectrum.shape
    angles = np.zeros((num_samples, num_sources))
    grids_mesh = np.arange(num_meshes) - start_bias
    for num in range(num_samples):
        li_0 = Spectrum[num, :].reshape(-1)
        # li_0[li_0 < 0] = 0
        li = li_0
        angle = np.zeros(num_sources) - 5
        peaks, _ = scipy.signal.find_peaks(li)
        max_spectrum = heapq.nlargest(num_sources, li[peaks])
        for i in range(len(max_spectrum)):
            angle[i] = grids_mesh[np.where(li == max_spectrum[i])[0][0]]
        angles[num] = angle.reshape(-1)
    return np.sort(angles, axis=1)[::-1]


def DoA2Spect(DoA, num_meshes=121, num_sources=2, start_bias=60):
    """
    :param DoA: (num_samples, num_sources)
    :param num_meshes:
    :param num_sources:
    :param start_bias:
    :return: (num_samples, num_meshes, 1)
    """
    num_samples, _ = DoA.shape
    spectrum = np.zeros((num_samples, num_meshes, 1))
    for num in range(num_samples):
        for i in range(num_sources):
            spectrum[num, int(DoA[num, i] + start_bias)] = 1
    return spectrum
