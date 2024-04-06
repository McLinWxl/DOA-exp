#  Copyright (c) 2024. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import numpy as np
import torch

def MUSIC(CovarianceMatrix: np.ndarray, num_antennas: int, num_sources: int, angle_meshes: np.ndarray, antenna_intarvals: float, wavelength_source: float) -> np.ndarray:
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
        a = np.exp(1j * np.pi * 2 * antenna_intarvals * np.arange(num_antennas)[:, np.newaxis] * np.sin(
            np.deg2rad(doa_search[doa_index])) / wavelength_source)
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


def SBL(raw_data, num_antennas: int, angle_meshes: np.ndarray, max_iteration=100, error_threshold=1e-3):
    """
    :param angle_meshes:
    :param num_antennas:
    :param raw_data:
    :param max_iteration:
    :param error_threshold:
    :return:
    """
    _, num_snapshots = raw_data.shape
    A = np.exp(1j * np.pi * np.arange(num_antennas)[:, np.newaxis] * np.sin(np.deg2rad(angle_meshes)))
    mu = A.T.conjugate() @ np.linalg.pinv(A @ A.T.conjugate()) @ raw_data
    sigma2 = 0.1 * np.linalg.norm(raw_data, 'fro') ** 2 / (num_antennas * num_snapshots)
    gamma = np.diag((mu @ mu.T.conjugate()).real) / num_snapshots
    ItrIdx = 1
    stop_iter = False
    gamma0 = gamma
    while not stop_iter and ItrIdx < max_iteration:
        gamma0 = gamma
        Q = sigma2 * np.eye(num_antennas) + np.dot(np.dot(A, np.diag(gamma)), A.T.conjugate())
        Qinv = np.linalg.pinv(Q)
        Sigma = np.diag(gamma) - np.dot(np.dot(np.dot(np.diag(gamma), A.T.conjugate()), Qinv),
                                        np.dot(A, np.diag(gamma)))
        mu = np.dot(np.dot(np.diag(gamma), A.T.conjugate()), np.dot(Qinv, raw_data))
        sigma2 = ((np.linalg.norm(raw_data - np.dot(A, mu), 'fro') ** 2 + num_snapshots * np.trace(
            np.dot(np.dot(A, Sigma), A.T.conjugate()))) /
                  (num_antennas * num_snapshots)).real
        mu_norm = np.diag(mu @ mu.T.conjugate()) / num_snapshots
        gamma = np.abs(mu_norm + np.diag(Sigma))

        if np.linalg.norm(gamma - gamma0) / np.linalg.norm(gamma) < error_threshold:
            stop_iter = True
        ItrIdx += 1
    return gamma


def ISTA(covariance_array, dictionary, angle_meshes: np.ndarray, max_iter=100, tol=1e-6):
    """
    :param dictionary:
    :param angle_meshes:
    :param covariance_array:
    :param max_iter:
    :param tol:
    :return:
    """
    angle_meshes = len(angle_meshes)
    predict = np.zeros((angle_meshes, 1))
    stop_flag = False
    num_iter = 0
    W = dictionary
    while not stop_flag and num_iter < max_iter:
        predict0 = predict
        mu = np.max(np.linalg.eigvals(W.T.conj() @ W))
        alpha = 1 / mu
        theta = alpha * 0.1
        G = np.eye(angle_meshes) - alpha * W.T.conj() @ W
        H = alpha * W.T.conj()
        r = np.matmul(G, predict) + np.matmul(H, covariance_array)
        predict = np.abs(np.maximum(np.abs(r) - theta, 0) * np.sign(r))
        if np.linalg.norm(predict - predict0) / np.linalg.norm(predict) < tol:
            stop_flag = True
        num_iter += 1
    return (predict - np.min(predict)) / (np.max(predict) - np.min(predict))


class AMI_LISTA(torch.nn.Module):
    def __init__(self, dictionary, **kwargs):
        """
        :param dictionary **required**
        :param num_layers: 10
        :param device: 'cpu
        :param mode: None ('tied', 'single', or 'both')
        """
        super(AMI_LISTA, self).__init__()
        self.num_sensors = int(np.sqrt(dictionary.shape[0]))
        M2 = self.num_sensors ** 2
        self.num_meshes = dictionary.shape[1]
        self.M2 = M2
        self.num_layers = kwargs.get('num_layers', 10)
        self.device = kwargs.get('device', 'cpu')
        self.mode = kwargs.get('mode', None)

        print(f'mode: {self.mode}')
        if not self.mode:
            self.W1 = torch.nn.Parameter(torch.eye(M2).repeat(self.num_layers, 1, 1)
                                         + 1j * torch.zeros([self.num_layers, M2, M2]), requires_grad=True)
            self.W2 = torch.nn.Parameter(torch.eye(M2).repeat(self.num_layers, 1, 1)
                                         + 1j * torch.zeros([self.num_layers, M2, M2]), requires_grad=True)
        elif self.mode == 'tied':
            self.W1 = torch.nn.Parameter(torch.eye(M2) + 1j * torch.zeros([M2, M2]), requires_grad=True)
            self.W2 = torch.nn.Parameter(torch.eye(M2) + 1j * torch.zeros([M2, M2]), requires_grad=True)
        elif self.mode == 'single':
            self.W = torch.nn.Parameter(torch.eye(M2).repeat(self.num_layers, 1, 1)
                                        + 1j * torch.zeros([self.num_layers, M2, M2]), requires_grad=True)
        elif self.mode == 'both':
            self.W = torch.nn.Parameter(torch.eye(M2) + 1j * torch.zeros([M2, M2]), requires_grad=True)
        self.theta = torch.nn.Parameter(0.001 * torch.ones(self.num_layers), requires_grad=True)
        self.gamma = torch.nn.Parameter(0.001 * torch.ones(self.num_layers), requires_grad=True)
        self.leakly_relu = torch.nn.LeakyReLU()
        self.dictionary = dictionary
        self.relu = torch.nn.ReLU()

    def forward(self, covariance_vector: torch.Tensor):
        dictionary = self.dictionary.to(torch.complex64)
        covariance_vector = covariance_vector.reshape(-1, self.M2, 1).to(self.device).to(torch.complex64)
        covariance_vector = covariance_vector / torch.linalg.matrix_norm(covariance_vector, ord=np.inf, keepdim=True)
        batch_size = covariance_vector.shape[0]
        x0 = torch.matmul(dictionary.conj().T, covariance_vector).real.float()
        # x0 /= torch.norm(x0, dim=1, keepdim=True)
        x_real = x0
        x_layers_virtual = torch.zeros(batch_size, self.num_layers, self.num_meshes, 1).to(self.device)
        for layer in range(self.num_layers):
            identity_matrix = (torch.eye(self.num_meshes) + 1j * torch.zeros([self.num_meshes, self.num_meshes])).to(
                self.device)
            if not self.mode:
                W1 = self.W1[layer]
                W2 = self.W2[layer]
            elif self.mode == 'tied':
                W1 = self.W1
                W2 = self.W2
            elif self.mode == 'single':
                W1 = self.W[layer]
                W2 = self.W[layer]
            elif self.mode == 'both':
                W1 = self.W
                W2 = self.W
            else:
                raise Exception('mode error')
            W1D = torch.matmul(W1, dictionary)
            W2D = torch.matmul(W2, dictionary)
            Wt = identity_matrix - self.gamma[layer] * torch.matmul(W2D.conj().T, W2D)
            We = self.gamma[layer] * W1D.conj().T
            s = torch.matmul(Wt, x_real + 1j * torch.zeros_like(x_real)) + torch.matmul(We, covariance_vector)
            s_abs = torch.abs(s)
            if layer < self.num_layers - 1:
                x_real = self.leakly_relu(s_abs - self.theta[layer])
            else:
                x_real = self.relu(s_abs - self.theta[layer])
            x_real = x_real / (torch.norm(x_real, dim=1, keepdim=True) + 1e-20)
            # x_real = x_real / torch.mean(torch.norm(covariance_vector, dim=1, keepdim=True))
            x_layers_virtual[:, layer] = x_real
        return x_real, x_layers_virtual


class LISTA(torch.nn.Module):

    def __init__(self, dictionary, **kwargs):
        """
        :param dictionary: **required**
        :param num_layers: 10
        :param device: 'cpu'
        """
        super(LISTA, self).__init__()
        self.num_sensors = int(np.sqrt(dictionary.shape[0]))
        self.num_meshes = dictionary.shape[1]
        self.num_layers = kwargs.get('num_layers', 10)
        self.device = kwargs.get('device', 'cpu')
        num_sensors_powered = self.num_sensors * self.num_sensors

        self.We = torch.nn.Parameter(torch.randn([self.num_layers, self.num_meshes, num_sensors_powered]) +
                                     1j * torch.randn([self.num_layers, self.num_meshes, num_sensors_powered]),
                                     requires_grad=True)
        self.Wg = torch.nn.Parameter(torch.randn([self.num_layers, self.num_meshes, self.num_meshes]) +
                                     1j * torch.randn([self.num_layers, self.num_meshes, self.num_meshes]),
                                     requires_grad=True)
        self.theta = torch.nn.Parameter(0.01 * torch.ones(self.num_layers), requires_grad=True)

        self.num_sensors_2p = num_sensors_powered
        self.relu = torch.nn.ReLU()
        self.dictionary = dictionary

    def forward(self, covariance_vector, device="cpu"):
        dictionary = self.dictionary.to(torch.complex64)
        covariance_vector = covariance_vector.reshape(-1, self.num_sensors_2p, 1).to(torch.complex64).to(self.device)
        covariance_vector = covariance_vector / torch.linalg.matrix_norm(covariance_vector, ord=np.inf, keepdim=True)
        batchSize = covariance_vector.shape[0]
        x_eta = torch.matmul(dictionary.conj().T, covariance_vector).real.float()
        # x_eta /= torch.norm(x_eta, dim=1, keepdim=True)
        covariance_vector = covariance_vector.to(device)
        x_layers_virtual = torch.zeros(batchSize, self.num_layers, self.num_meshes, 1)

        for t in range(self.num_layers):
            We = self.We[t]
            Wg = self.Wg[t]
            z = torch.matmul(We, covariance_vector) + torch.matmul(Wg, (x_eta + 1j * torch.zeros_like(x_eta)))
            x_abs = torch.abs(z)
            # apply soft-thresholding on xabs, return xabs
            x_eta = self.relu(x_abs - self.theta[t])
            x_norm = x_eta.norm(dim=1, keepdim=True)
            x_eta = x_eta / (torch.sqrt(torch.tensor(2.)) * (x_norm + 1e-20))
            x_layers_virtual[:, t] = x_eta
        return x_eta, x_layers_virtual
