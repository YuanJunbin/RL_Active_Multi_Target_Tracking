import torch

from torch import tensor
from torch.optim import SGD, Adam
from models.policy_net import PolicyNet
from models.policy_net_att import PolicyNetAtt
from torch.distributions import MultivariateNormal
from utilities.utils import landmark_motion, triangle_SDF, get_transformation, phi, circle_SDF


class ModelBasedAgent:

    def __init__(self, num_landmarks, init_info, A, B, W, radius, psi, kappa, V, lr):
        self._init_info = init_info
        self._info = None

        self._num_landmarks = num_landmarks
        self._A = A
        self._B = B
        self._W = W
        self._psi = psi
        self._radius = radius
        self._kappa = kappa
        self._V = V
        self._inv_V = V ** (-1)

        # input_dim = num_landmarks * 4 + 3
        input_dim = num_landmarks * 4
        self._policy = PolicyNet(input_dim=input_dim)

        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr)

    def reset_agent_info(self):
        self._info = self._init_info * torch.ones((self._num_landmarks, 2))

    def reset_estimate_mu(self, mu_real):
        self._mu_update = mu_real + torch.normal(mean=torch.zeros(self._num_landmarks, 2), std=torch.sqrt(self._V))  # with the shape of (num_landmarks, 2)

    def eval_policy(self):
        self._policy.eval()

    def train_policy(self):
        self._policy.train()

    def plan(self, v, x):
        self._mu_predict = torch.clip(landmark_motion(self._mu_update, v, self._A, self._B),
                                      min=-tensor([self._num_landmarks, self._num_landmarks]),
                                      max=tensor([self._num_landmarks, self._num_landmarks]))
        self._info = (self._info**(-1) + self._W)**(-1)

        q_predict = torch.vstack(((self._mu_predict[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.sin(x[2]),
                          (x[0] - self._mu_predict[:, 0]) * torch.sin(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.cos(x[2]))).T

        # net_input = torch.hstack((x, self._info.flatten(), next_mu.flatten()))

        net_input = torch.hstack((self._info.flatten(), q_predict.flatten()))
        # net_input = q.flatten()
        action = self._policy.forward(net_input)
        return action

    def update_info_mu(self, mu_real, x):
        q_real = torch.vstack(((mu_real[:, 0] - x[0]) * torch.cos(x[2]) + (mu_real[:, 1] - x[1]) * torch.sin(x[2]),
                          (x[0] - mu_real[:, 0]) * torch.sin(x[2]) + (mu_real[:, 1] - x[1]) * torch.cos(x[2]))).T
        sensor_value = torch.zeros(self._num_landmarks, 2)
        SDF_real = triangle_SDF(q_real, self._psi, self._radius)
        for i in range(self._num_landmarks):
            if SDF_real[i] <= 0:
                sensor_value[i] = mu_real[i] + torch.normal(mean=torch.zeros(1, 2), std=torch.sqrt(self._V))
                # print(sensor_value[i])
            else:
                sensor_value[i] = self._mu_predict[i].flatten()
                # print(sensor_value[i])

        info_mat = torch.diag(self._info.flatten())
        R_mat = torch.eye(self._num_landmarks * 2) * self._V[0]  # sensor uncertainty covariance matrix
        S_mat = torch.inverse(info_mat) + R_mat
        kalman_gain = torch.inverse(info_mat) @ torch.inverse(S_mat)

        self._mu_update = torch.reshape(self._mu_predict.flatten() +
                                        (kalman_gain @ (sensor_value.flatten() -
                                                        (self._mu_predict).flatten())).flatten(), (self._num_landmarks, 2))

        q_update = torch.vstack(
            ((self._mu_update[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.sin(x[2]),
             (x[0] - self._mu_update[:, 0]) * torch.sin(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.cos(x[2]))).T
        SDF_update = triangle_SDF(q_update, self._psi, self._radius)
        M = (1 - phi(SDF_update, self._kappa))[:, None] * self._inv_V.repeat(self._num_landmarks, 1)
        # Assuming A = I:
        self._info = self._info + M

    # def update_policy(self, debug=False):
    #     self._policy_optimizer.zero_grad()
    #
    #     if debug:
    #         param_list = []
    #         grad_power = 0
    #         for i, p in enumerate(self._policy.parameters()):
    #             param_list.append(p.data.detach().clone())
    #             if p.grad is not None:
    #                 grad_power += (p.grad**2).sum()
    #             else:
    #                 grad_power += 0
    #
    #         print("Gradient power before backward: {}".format(grad_power))
    #
    #     reward = - torch.sum(torch.log(self._info))
    #     reward.backward()
    #     self._policy_optimizer.step()
    #
    #     if debug:
    #         grad_power = 0
    #         total_param_ssd = 0
    #         for i, p in enumerate(self._policy.parameters()):
    #             if p.grad is not None:
    #                 grad_power += (p.grad ** 2).sum()
    #             else:
    #                 grad_power += 0
    #             total_param_ssd += ((param_list[i] - p.data) ** 2).sum()
    #
    #         print("Gradient power after backward: {}".format(grad_power))
    #         print("SSD of weights after applying the gradient: {}".format(total_param_ssd))
    #
    #     return -reward.item()

    def set_policy_grad_to_zero(self):
        self._policy_optimizer.zero_grad()

    def update_policy_grad(self, train=True):
        reward = - torch.sum(torch.log(self._info))
        if train == True:
            reward.backward()
        return -reward.item()

    # def update_policy_grad(self, mu, x):
    #     reward = ((x[:2] - mu)**2).sum()
    #     reward.backward()
    #     return reward.item()

    def policy_step(self, debug=False):
        if debug:
            param_list = []
            for i, p in enumerate(self._policy.parameters()):
                param_list.append(p.data.detach().clone())

        self._policy_optimizer.step()

        if debug:
            total_param_rssd = 0
            grad_power = 0
            for i, p in enumerate(self._policy.parameters()):
                if p.grad is not None:
                    grad_power += (p.grad ** 2).sum()
                else:
                    grad_power += 0
                total_param_rssd += ((param_list[i] - p.data) ** 2).sum().sqrt()

            print("Gradient power after backward: {}".format(grad_power))
            print("RSSD of weights after applying the gradient: {}".format(total_param_rssd))

    def get_policy_state_dict(self):
        return self._policy.state_dict()

    def load_policy_state_dict(self, load_model):
        self._policy.load_state_dict(torch.load(load_model))


class ModelBasedAgentAtt:

    def __init__(self, max_num_landmarks, init_info, A, B, W, radius, psi, kappa, V, lr):
        self._init_info = init_info
        self._info = None

        self._max_num_landmarks = max_num_landmarks
        self._A = A
        self._B = B
        self._W = W
        self._psi = psi
        self._radius = radius
        self._kappa = kappa
        self._V = V
        self._inv_V = V ** (-1)

        # input_dim = num_landmarks * 4 + 3
        input_dim = max_num_landmarks * 5 + 3
        self._policy = PolicyNetAtt(input_dim=input_dim)

        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr)

    def reset_agent_info(self):
        self._info = self._init_info * torch.ones((self._num_landmarks, 2))

    def reset_estimate_mu(self, mu_real):
        self._num_landmarks = mu_real.size()[0]
        self._mu_update = mu_real + torch.normal(mean=torch.zeros(self._num_landmarks, 2), std=torch.sqrt(self._V))  # with the shape of (num_landmarks, 2)
        self._padding = torch.zeros(2 * (self._max_num_landmarks - self._num_landmarks))
        self._mask = torch.tensor([True] * self._num_landmarks + [False] * (self._max_num_landmarks - self._num_landmarks))

    def eval_policy(self):
        self._policy.eval()

    def train_policy(self):
        self._policy.train()

    def plan(self, v, x):
        # self._mu_predict = torch.clip(landmark_motion(self._mu_update, v, self._A, self._B),
        #                               min=-tensor([self._num_landmarks, self._num_landmarks]),
        #                               max=tensor([self._num_landmarks, self._num_landmarks]))
        self._mu_predict = landmark_motion(self._mu_update, v, self._A, self._B)
        # print(f"info mat is {self._info}")
        # print(f"W is {self._W}")
        self._info = (self._info**(-1) + self._W)**(-1)

        q_predict = torch.vstack(((self._mu_predict[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.sin(x[2]),
                          (x[0] - self._mu_predict[:, 0]) * torch.sin(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.cos(x[2]))).T

        # net_input = torch.hstack((x, self._info.flatten(), next_mu.flatten()))

        agent_pos_local = torch.zeros(3)
        net_input = torch.hstack((agent_pos_local, self._info.flatten(),
                                  self._padding, q_predict.flatten(), self._padding, self._mask))
        # net_input = q.flatten()
        action = self._policy.forward(net_input)
        return action

    def update_info_mu(self, mu_real, x):
        q_real = torch.vstack(((mu_real[:, 0] - x[0]) * torch.cos(x[2]) + (mu_real[:, 1] - x[1]) * torch.sin(x[2]),
                          (x[0] - mu_real[:, 0]) * torch.sin(x[2]) + (mu_real[:, 1] - x[1]) * torch.cos(x[2]))).T
        sensor_value = torch.zeros(self._num_landmarks, 2)
        SDF_real = triangle_SDF(q_real, self._psi, self._radius)
        for i in range(self._num_landmarks):
            if SDF_real[i] <= 0:
                sensor_value[i] = mu_real[i] + torch.normal(mean=torch.zeros(1, 2), std=torch.sqrt(self._V))
                # print(sensor_value[i])
            else:
                sensor_value[i] = self._mu_predict[i].flatten()
                # print(sensor_value[i])

        info_mat = torch.diag(self._info.flatten())
        R_mat = torch.eye(self._num_landmarks * 2) * self._V[0]  # sensor uncertainty covariance matrix
        S_mat = torch.inverse(info_mat) + R_mat
        kalman_gain = torch.inverse(info_mat) @ torch.inverse(S_mat)

        self._mu_update = torch.reshape(self._mu_predict.flatten() +
                                        (kalman_gain @ (sensor_value.flatten() -
                                                        (self._mu_predict).flatten())).flatten(), (self._num_landmarks, 2))

        q_update = torch.vstack(
            ((self._mu_update[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.sin(x[2]),
             (x[0] - self._mu_update[:, 0]) * torch.sin(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.cos(x[2]))).T
        SDF_update = triangle_SDF(q_update, self._psi, self._radius)
        M = (1 - phi(SDF_update, self._kappa))[:, None] * self._inv_V.repeat(self._num_landmarks, 1)
        # Assuming A = I:
        self._info = self._info + M

    def set_policy_grad_to_zero(self):
        self._policy_optimizer.zero_grad()

    def update_policy_grad(self, train=True):
        reward = - torch.sum(torch.log(self._info))
        if train == True:
            reward.backward()
        return -reward.item()

    def policy_step(self, debug=False):
        if debug:
            param_list = []
            for i, p in enumerate(self._policy.parameters()):
                param_list.append(p.data.detach().clone())

        self._policy_optimizer.step()

        if debug:
            total_param_rssd = 0
            grad_power = 0
            for i, p in enumerate(self._policy.parameters()):
                if p.grad is not None:
                    grad_power += (p.grad ** 2).sum()
                else:
                    grad_power += 0
                total_param_rssd += ((param_list[i] - p.data) ** 2).sum().sqrt()

            print("Gradient power after backward: {}".format(grad_power))
            print("RSSD of weights after applying the gradient: {}".format(total_param_rssd))

    def get_policy_state_dict(self):
        return self._policy.state_dict()

    def load_policy_state_dict(self, load_model):
        self._policy.load_state_dict(torch.load(load_model))

class ModelBasedAgentAtt2D:

    def __init__(self, max_num_landmarks, init_info, A, B, W, radius, kappa, V, lr):
        self._init_info = init_info
        self._info = None

        self._max_num_landmarks = max_num_landmarks
        self._A = A
        self._B = B
        self._W = W
        self._radius = radius
        self._kappa = kappa
        self._V = V
        self._inv_V = V ** (-1)

        # input_dim = num_landmarks * 4 + 3
        input_dim = max_num_landmarks * 5 + 3
        self._policy = PolicyNetAtt(input_dim=input_dim)

        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr)

    def reset_agent_info(self):
        self._info = self._init_info * torch.ones((self._num_landmarks, 2))

    def reset_estimate_mu(self, mu_real):
        self._num_landmarks = mu_real.size()[0]
        self._mu_update = mu_real + torch.normal(mean=torch.zeros(self._num_landmarks, 2), std=torch.sqrt(self._V))  # with the shape of (num_landmarks, 2)
        self._padding = torch.zeros(2 * (self._max_num_landmarks - self._num_landmarks))
        self._mask = torch.tensor([True] * self._num_landmarks + [False] * (self._max_num_landmarks - self._num_landmarks))

    def eval_policy(self):
        self._policy.eval()

    def train_policy(self):
        self._policy.train()

    def plan(self, v, x):
        self._mu_predict = landmark_motion(self._mu_update, v, self._A, self._B)
        # print(f"info mat is {self._info}")
        # print(f"W is {self._W}")
        self._info = (self._info**(-1) + self._W)**(-1)

        q_predict = torch.vstack(((self._mu_predict[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.sin(x[2]),
                          (x[0] - self._mu_predict[:, 0]) * torch.sin(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.cos(x[2]))).T

        agent_pos_local = torch.zeros(3)
        net_input = torch.hstack((agent_pos_local, self._info.flatten(),
                                  self._padding, q_predict.flatten(), self._padding, self._mask))

        action = self._policy.forward(net_input)
        return action

    def update_info_mu(self, mu_real, x):
        q_real = torch.vstack(((mu_real[:, 0] - x[0]) * torch.cos(x[2]) + (mu_real[:, 1] - x[1]) * torch.sin(x[2]),
                          (x[0] - mu_real[:, 0]) * torch.sin(x[2]) + (mu_real[:, 1] - x[1]) * torch.cos(x[2]))).T
        sensor_value = torch.zeros(self._num_landmarks, 2)
        SDF_real = circle_SDF(q_real, self._radius)
        for i in range(self._num_landmarks):
            if SDF_real[i] <= 0:
                sensor_value[i] = mu_real[i] + torch.normal(mean=torch.zeros(1, 2), std=torch.sqrt(self._V))
                # print(f"target {i} detected")
                # print(sensor_value[i])
            else:
                sensor_value[i] = self._mu_predict[i].flatten()
                # print(sensor_value[i])

        info_mat = torch.diag(self._info.flatten())
        R_mat = torch.eye(self._num_landmarks * 2) * self._V[0]  # sensor uncertainty covariance matrix
        S_mat = torch.inverse(info_mat) + R_mat
        kalman_gain = torch.inverse(info_mat) @ torch.inverse(S_mat)

        self._mu_update = torch.reshape(self._mu_predict.flatten() +
                                        (kalman_gain @ (sensor_value.flatten() -
                                                        (self._mu_predict).flatten())).flatten(), (self._num_landmarks, 2))

        q_update = torch.vstack(
            ((self._mu_update[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.sin(x[2]),
             (x[0] - self._mu_update[:, 0]) * torch.sin(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.cos(x[2]))).T
        SDF_update = circle_SDF(q_update, self._radius)
        M = (1 - phi(SDF_update, self._kappa))[:, None] * self._inv_V.repeat(self._num_landmarks, 1)
        # Assuming A = I:
        self._info = self._info + M

    def set_policy_grad_to_zero(self):
        self._policy_optimizer.zero_grad()

    def update_policy_grad(self, train=True):
        reward = - torch.sum(torch.log(self._info))
        if train == True:
            reward.backward()
        return -reward.item()

    def policy_step(self, debug=False):
        if debug:
            param_list = []
            for i, p in enumerate(self._policy.parameters()):
                param_list.append(p.data.detach().clone())

        self._policy_optimizer.step()

        if debug:
            total_param_rssd = 0
            grad_power = 0
            for i, p in enumerate(self._policy.parameters()):
                if p.grad is not None:
                    grad_power += (p.grad ** 2).sum()
                else:
                    grad_power += 0
                total_param_rssd += ((param_list[i] - p.data) ** 2).sum().sqrt()

            print("Gradient power after backward: {}".format(grad_power))
            print("RSSD of weights after applying the gradient: {}".format(total_param_rssd))

    def get_policy_state_dict(self):
        return self._policy.state_dict()

    def load_policy_state_dict(self, load_model):
        self._policy.load_state_dict(torch.load(load_model))

class ModelBasedAgentAtt4D:
    def __init__(self, max_num_landmarks, radius, obs_cov, lr, kappa, dt=0.5):
        self._max_num_landmarks = max_num_landmarks
        self._radius = radius

        self._info = None

        input_dim = max_num_landmarks * 5 + 3
        self._motion_noise_cov = None # (N, 4, 4) Q matrix
        self._obs_cov = obs_cov
        self._R_inv = torch.inverse(obs_cov)
        self._obs_dist = torch.distributions.MultivariateNormal(torch.zeros(2), covariance_matrix=self._obs_cov)

        self._H = tensor([[1., 0., 0., 0.],
                          [0., 1., 0., 0.]])
        
        self._F = tensor([[1., 0., dt, 0.],
                          [0., 1., 0., dt],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]])
        
        self._F_inv = torch.linalg.inv(self._F)
        self._F_inv_T = self._F_inv.T

        self._info_increment = self._H.T @ self._R_inv @ self._H

        self._policy = PolicyNetAtt(input_dim=input_dim)
        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr)
        self._kappa = kappa

    def reset_agent_info(self, target_list, mu_real):
        mu_list = []
        info_list = []
        motion_cov_list = []

        for target in target_list:
            mu_list.append(target.init_mean)
            info_list.append(torch.linalg.inv(target.cov_init))
            motion_cov_list.append(target.Q)

        self._mu_update = torch.stack(mu_list, dim=0).detach()            # shape: (N, 4)
        self._info = torch.stack(info_list, dim=0).detach()               # shape: (N, 4, 4)
        self._motion_cov = torch.stack(motion_cov_list, dim=0).detach()

        # print(f"init mu update is {self._mu_update}")
        # print(f"init info is {self._info}")

    def reset_estimate_mu(self, mu_real):
        self._num_landmarks = mu_real.size()[0]
        self._padding = torch.zeros(2 * (self._max_num_landmarks - self._num_landmarks))
        self._mask = torch.tensor([True] * self._num_landmarks + [False] * (self._max_num_landmarks - self._num_landmarks))

    def eval_policy(self):
        self._policy.eval()

    def train_policy(self):
        self._policy.train()

    def plan(self, x):
        self._mu_predict = (self._F @ self._mu_update.transpose(0, 1)).transpose(0, 1)  # (N, 4)

        info_predict = []
        for i in range(self._num_landmarks):
            Y_mat = self._info[i]
            Q_mat = self._motion_cov[i]

            cov_mat = torch.linalg.inv(Y_mat)
            cov_pred = self._F @ cov_mat @ self._F.T + Q_mat

            Y_pred = torch.linalg.inv(cov_pred)
            info_predict.append(Y_pred)
        
        # print(f"predict: prev info is {self._info}")
        # print(f"predict: Q is {self._motion_cov}")

        self._info = torch.stack(info_predict, dim=0)

        eps = 1e-6
        _info_rec = self._info + eps * torch.eye(4, device=self._info.device)

        # print(f"predict: after info is {self._info}")

        # print(f"plan: info mat is {self._info}")

        q_predict = torch.vstack(((self._mu_predict[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.sin(x[2]),
                          (x[0] - self._mu_predict[:, 0]) * torch.sin(x[2]) + (self._mu_predict[:, 1] - x[1]) * torch.cos(x[2]))).T

        # net_input = torch.hstack((x, self._info.flatten(), next_mu.flatten()))
        info_diag_pos = torch.stack([_info_rec[i][[0, 1], [0, 1]] for i in range(self._num_landmarks)])  # shape: (N, 2)

        agent_pos_local = torch.zeros(3, device=self._mu_predict.device)
        net_input = torch.hstack((agent_pos_local, info_diag_pos.flatten(),
                                  self._padding, q_predict.flatten(), self._padding, self._mask))
        # net_input = q.flatten()
        print(f"net input is {net_input}")
        if torch.isnan(net_input).any():
            raise RuntimeError("NaN detected in net input first")
        action = self._policy.forward(net_input)
        print(f"action output is {action}")
        if torch.isnan(action).any():
            raise RuntimeError("NaN detected in action first")
        return action

    def update_info_mu(self, mu_real, x):
        q_real = torch.vstack(((mu_real[:, 0] - x[0]) * torch.cos(x[2]) + (mu_real[:, 1] - x[1]) * torch.sin(x[2]),
                               (x[0] - mu_real[:, 0]) * torch.sin(x[2]) + (mu_real[:, 1] - x[1]) * torch.cos(x[2]))).T
        
        sensor_value = torch.zeros(self._num_landmarks, 2)
        SDF_real = circle_SDF(q_real, self._radius)
        for i in range(self._num_landmarks):
            if SDF_real[i] <= 0:
                obs_noise = self._obs_dist.rsample()
                sensor_value[i] = mu_real[i, :2] + obs_noise
            else:
                sensor_value[i] = self._mu_predict[i, :2].flatten()

        R_inv = self._R_inv
        H_mat = self._H

        self._mu_update = torch.zeros_like(self._mu_predict) # shape: (N, 4)

        for i in range(self._num_landmarks):
            mu_i = self._mu_predict[i]     # shape: (4,)
            Y_i  = self._info[i]           # shape: (4, 4)
            z_i  = sensor_value[i]         # shape: (2,)

            # Compute updated information matrix:
            Y_post = Y_i + H_mat.T @ R_inv @ H_mat

            # Compute updated information vector:
            eta_post = Y_i @ mu_i + H_mat.T @ R_inv @ z_i

            # Recover updated mean: mu_post = Y_post^{-1} @ eta_post
            mu_post = torch.linalg.solve(Y_post, eta_post)

            # Store
            self._mu_update[i] = mu_post

        # print(f"mu_update is {self._mu_update}")
        # print("x =", x)
        # print("cos(x[2]) =", torch.cos(x[2]))
        # print("sin(x[2]) =", torch.sin(x[2]))
        # print("mu_update[:, 0] =", self._mu_update[:, 0])
        # print("mu_update[:, 1] =", self._mu_update[:, 1])

        q_update = torch.vstack(((self._mu_update[:, 0] - x[0]) * torch.cos(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.sin(x[2]),
                                 (x[0] - self._mu_update[:, 0]) * torch.sin(x[2]) + (self._mu_update[:, 1] - x[1]) * torch.cos(x[2]))).T
        
        # print(f"q_update is {q_update}")

        SDF_update = circle_SDF(q_update, self._radius)
        soft_gate = 1.0 - torch.sigmoid(self._kappa * SDF_update)

        # print(f"udpate: prev info is {self._info}")
        # print(f"udpate: info_increment is {self._info_increment}")
        # print(f"udpate: sdf update is {SDF_update}")
        # print(f"udpate: soft gate is {soft_gate}")

        self._info = self._info + soft_gate[:, None, None] * self._info_increment  # info_increment shape: (4, 4)

        # print(f"udpate: updated info is {self._info}")

    def set_policy_grad_to_zero(self):
        self._policy_optimizer.zero_grad()

    def update_policy_grad(self, train=True):
        # Convert information matrix to covariance
        eps = 1e-6
        cov_matrices = torch.linalg.inv(self._info + eps * torch.eye(4, device=self._info.device))  # shape: (N, 4, 4)
        pos_covs = cov_matrices[:, :2, :2]  # shape: (N, 2, 2)
        # log_dets = torch.logdet(pos_covs)  # shape: (N,)
        sign, log_dets = torch.linalg.slogdet(pos_covs)
        if not torch.all(sign > 0):
            raise RuntimeError("Non-positive-definite covariance detected before reward")

        reward = - torch.sum(log_dets)
        
        if train:
            reward.backward()
            for i in range(pos_covs.shape[0]):
                eigvals = torch.linalg.eigvalsh(pos_covs[i])
                print(f"Eigenvalues of pos_cov[{i}]:", eigvals)
            for name, p in self._policy.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    print(f"[NaN DETECTED] in gradient of {name} IMMEDIATELY after backward()")
                    raise RuntimeError("Confirmed: backward produced NaN")
            print(f"backward without Nan")
            # torch.nn.utils.clip_grad_norm_(self._policy.parameters(), max_norm=1.0)
        return -reward.item()
    
    def policy_step(self, debug=False):
        if debug:
            param_list = []
            for i, p in enumerate(self._policy.parameters()):
                param_list.append(p.data.detach().clone())

        # Check gradients before step
        for name, p in self._policy.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"[ERROR] NaN or Inf in gradient of {name}")
                    raise RuntimeError("Gradient NaN detected before optimizer step")

        # Check for gradient norm explosion (optional but recommended)
        total_grad_norm = 0.0
        for p in self._policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"[DEBUG] Total gradient L2 norm: {total_grad_norm:.4e}")

        self._policy_optimizer.step()

        # Check for NaNs in weights *after* step
        for name, p in self._policy.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"[ERROR] NaN or Inf in weights of {name} after step")
                raise RuntimeError("NaN detected in weights after optimizer step")

        if debug:
            total_param_rssd = 0
            grad_power = 0
            for i, p in enumerate(self._policy.parameters()):
                if p.grad is not None:
                    grad_power += (p.grad ** 2).sum()
                else:
                    grad_power += 0
                total_param_rssd += ((param_list[i] - p.data) ** 2).sum().sqrt()

            print("Gradient power after backward: {}".format(grad_power))
            print("RSSD of weights after applying the gradient: {}".format(total_param_rssd))

    # def policy_step(self, debug=False):
    #     if debug:
    #         param_list = []
    #         for i, p in enumerate(self._policy.parameters()):
    #             param_list.append(p.data.detach().clone())

    #     self._policy_optimizer.step()

    #     if debug:
    #         total_param_rssd = 0
    #         grad_power = 0
    #         for i, p in enumerate(self._policy.parameters()):
    #             if p.grad is not None:
    #                 grad_power += (p.grad ** 2).sum()
    #             else:
    #                 grad_power += 0
    #             total_param_rssd += ((param_list[i] - p.data) ** 2).sum().sqrt()

    #         print("Gradient power after backward: {}".format(grad_power))
    #         print("RSSD of weights after applying the gradient: {}".format(total_param_rssd))

    def get_policy_state_dict(self):
        return self._policy.state_dict()

    def load_policy_state_dict(self, load_model):
        self._policy.load_state_dict(torch.load(load_model))