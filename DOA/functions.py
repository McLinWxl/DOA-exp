import numpy as np


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
