#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import numpy as np


def MUSIC(CovarianceMatrix: np.ndarray, num_antennas: int, num_sources: int, angle_meshes: np.ndarray) -> np.ndarray:
    """
    Multi-Signal Classification (MUSIC) algorithm
    :param CovarianceMatrix:
    :param num_sources:
    :param angle_meshes:
    :return:
    """
    w, V = np.linalg.eig(CovarianceMatrix)
    w_index_order = np.argsort(w)
    V_noise = V[:, w_index_order[0:-num_sources]]
    noise_subspace = np.matmul(V_noise, np.matrix.getH(V_noise))
    doa_search = angle_meshes
    p_music = np.zeros((len(doa_search), 1))
    for doa_index in range(len(doa_search)):
        a = np.exp(
            1j * np.pi * np.arange(num_antennas)[:, np.newaxis] * np.sin(np.deg2rad(doa_search[doa_index])))
        p_music[doa_index] = np.abs(1 / np.matmul(np.matmul(np.matrix.getH(a), noise_subspace), a).reshape(-1)[0])
    p_music = p_music / np.max(p_music)
    p_music = 10 * np.log10(p_music)
    return p_music - np.min(p_music)


def MVDR(CovarianceMatrix: np.ndarray, num_antennas: int, angle_meshes: np.ndarray):
    """
    Minimum Variance Distortionless Response (MVDR) algorithm
    :param CovarianceMatrix:
    :param num_antennas:
    :param angle_meshes:
    :return:
    """
    sigma = []
    for i in range(len(angle_meshes)):
        a = np.exp(1j * np.pi * np.arange(num_antennas)[:, np.newaxis] * np.sin(np.deg2rad(angle_meshes[i])))
        sigma.append(1 / ((a.conj().T @ np.linalg.pinv(CovarianceMatrix) @ a) + 1e-20))
    sigma = np.array(sigma).reshape([-1, 1])
    sigma = np.abs(sigma)
    return (sigma - np.min(sigma)) / (np.max(sigma) - np.min(sigma))

class ModelMethods:

    def __init__(self, dictionary, **kwargs):
        """
        :param dictionary: **required**
        :param num_layers: 10
        :param device: 'cpu'
        """
        super(ModelMethods, self).__init__()
        self.num_sensors = int(np.sqrt(dictionary.shape[0]))
        self.num_meshes = dictionary.shape[1]
        self.num_sources = kwargs.get('num_sources', 2)
        self.num_snapshots = kwargs.get('num_snapshots', 256)
        self.dictionary = dictionary
        resolution = kwargs.get('resolution', 1)
        self.angles = np.linspace(-0.5 * (self.num_meshes - 1) * resolution, 0.5 * (self.num_meshes - 1) * resolution,
                                  self.num_meshes)

    def ISTA(self, covariance_array, max_iter=500, tol=1e-6):
        """
        :param covariance_array:
        :param max_iter:
        :param tol:
        :return:
        """
        predict = np.zeros((self.num_meshes, 1))
        stop_flag = False
        num_iter = 0
        W = self.dictionary
        while not stop_flag and num_iter < max_iter:
            predict0 = predict
            mu = np.max(np.linalg.eigvals(W.T.conj() @ W))
            alpha = 1 / mu
            theta = alpha * 0.1
            G = np.eye(self.num_meshes) - alpha * W.T.conj() @ W
            H = alpha * W.T.conj()
            r = np.matmul(G, predict) + np.matmul(H, covariance_array)
            predict = np.abs(np.maximum(np.abs(r) - theta, 0) * np.sign(r))
            if np.linalg.norm(predict - predict0) / np.linalg.norm(predict) < tol:
                stop_flag = True
            num_iter += 1
        return (predict - np.min(predict)) / (np.max(predict) - np.min(predict))

    def MUSIC(self, CovarianceMatrix):
        """
        :param CovarianceMatrix
        :return:
        """
        w, V = np.linalg.eig(CovarianceMatrix)
        w_index_order = np.argsort(w)
        V_noise = V[:, w_index_order[0:-self.num_sources]]
        noise_subspace = np.matmul(V_noise, np.matrix.getH(V_noise))
        doa_search = self.angles
        p_music = np.zeros((len(doa_search), 1))
        for doa_index in range(len(doa_search)):
            a = np.exp(
                1j * np.pi * np.arange(self.num_sensors)[:, np.newaxis] * np.sin(np.deg2rad(doa_search[doa_index])))
            p_music[doa_index] = np.abs(1 / np.matmul(np.matmul(np.matrix.getH(a), noise_subspace), a).reshape(-1)[0])
        p_music = p_music / np.max(p_music)
        p_music = 10 * np.log10(p_music)
        return p_music - np.min(p_music)

    def SBL(self, raw_data, max_iteration=500, error_threshold=1e-3):
        """
        :param raw_data:
        :param max_iteration:
        :param error_threshold:
        :return:
        """
        A = np.exp(1j * np.pi * np.arange(self.num_sensors)[:, np.newaxis] * np.sin(np.deg2rad(self.angles)))
        mu = A.T.conjugate() @ np.linalg.pinv(A @ A.T.conjugate()) @ raw_data
        sigma2 = 0.1 * np.linalg.norm(raw_data, 'fro') ** 2 / (self.num_sensors * self.num_snapshots)
        gamma = np.diag((mu @ mu.T.conjugate()).real) / self.num_snapshots
        ItrIdx = 1
        stop_iter = False
        gamma0 = gamma
        while not stop_iter and ItrIdx < max_iteration:
            gamma0 = gamma
            Q = sigma2 * np.eye(self.num_sensors) + np.dot(np.dot(A, np.diag(gamma)), A.T.conjugate())
            Qinv = np.linalg.pinv(Q)
            Sigma = np.diag(gamma) - np.dot(np.dot(np.dot(np.diag(gamma), A.T.conjugate()), Qinv),
                                            np.dot(A, np.diag(gamma)))
            mu = np.dot(np.dot(np.diag(gamma), A.T.conjugate()), np.dot(Qinv, raw_data))
            sigma2 = ((np.linalg.norm(raw_data - np.dot(A, mu), 'fro') ** 2 + self.num_snapshots * np.trace(
                np.dot(np.dot(A, Sigma), A.T.conjugate()))) /
                      (self.num_sensors * self.num_snapshots)).real
            mu_norm = np.diag(mu @ mu.T.conjugate()) / self.num_snapshots
            gamma = np.abs(mu_norm + np.diag(Sigma))

            if np.linalg.norm(gamma - gamma0) / np.linalg.norm(gamma) < error_threshold:
                stop_iter = True
            ItrIdx += 1
        return gamma

    def MVDR(self, covarianceMatrix):
        """
        :param covarianceMatrix:
        :return:
        """
        sigma = []
        for i in range(self.num_meshes):
            a = np.exp(1j * np.pi * np.arange(self.num_sensors)[:, np.newaxis] * np.sin(np.deg2rad(self.angles[i])))
            sigma.append(1 / ((a.conj().T @ np.linalg.pinv(covarianceMatrix) @ a) + 1e-20))
        sigma = np.array(sigma).reshape([-1, 1])
        sigma = np.abs(sigma)
        return (sigma - np.min(sigma)) / (np.max(sigma) - np.min(sigma))