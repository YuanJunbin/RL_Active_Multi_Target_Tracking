import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from typing import Tuple
from torch import tensor
from utilities.utils import SE2_kinematics, Simple_kinematics, landmark_motion_real


class SimpleEnv:

    def __init__(self, num_landmarks, horizon, width, height, tau, A, B, V, W, landmark_motion_scale, psi, radius):
        self._num_landmarks = num_landmarks
        self._horizon = horizon
        self._env_size = tensor([self._num_landmarks*2, self._num_landmarks*2])
        self._tau = tau
        self._A = A
        self._B = B
        self._V = V  # sensor_std ** 2 with the shape of (2, )
        self._W = W  # motion_std ** 2 with the shape of (2, )
        self._landmark_motion_scale = landmark_motion_scale

        self._mu = None
        self._v = None
        self._landmark_motion_bias = None
        self._x = None
        self._step_num = None

        self._psi = psi
        self._radius = radius

    def reset(self):
        mu = (torch.rand((self._num_landmarks, 2)) - 0.5) * self._env_size

        landmark_motion_bias = (torch.rand(2) - 0.5) * 2
        v = (torch.rand((self._num_landmarks, 2)) + landmark_motion_bias - 0.5) * self._landmark_motion_scale

        x = torch.empty(3)
        x[:2] = (torch.rand(2) - 0.5) * self._env_size
        x[2] = (torch.rand(1) * 2 - 1) * torch.pi

        self._mu_real = mu
        self._v = v
        self._landmark_motion_bias = landmark_motion_bias
        self._x = x
        self._step_num = 0

        self.history_poses = [self._x.detach().numpy().tolist()]
        self.fig = plt.figure(1)
        self.ax = self.fig.gca()

        return self._mu_real, v, x, False

    def step(self, action: tensor) -> Tuple[tensor, tensor, tensor, bool]:
        self._x = SE2_kinematics(self._x, action, self._tau)
        # self._x[:2] = torch.clip(self._x[:2], min=torch.zeros(2), max=self._env_size)

        self._mu_real = torch.clip(landmark_motion_real(self._mu_real, self._v, self._A, self._B, self._W),
                                   min=-self._env_size/2, max=self._env_size/2)

        self._v = (torch.rand((self._num_landmarks, 2)) + self._landmark_motion_bias - 0.5) *\
                  self._landmark_motion_scale

        done = False
        self._step_num += 1
        if self._step_num >= self._horizon:
            done = True

        self.history_poses.append(self._x.detach().numpy().tolist())

        return self._mu_real, self._v, self._x, done

    # def render(self):
    #     render_size = 50
    #     arrow_length = 20
    #     canvas = 255 * np.ones((self._env_size[1] * render_size, self._env_size[0] * render_size), dtype=np.uint8)
    #     canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    #
    #     # cv2.circle(canvas, (0, int(self._env_size[1] / 4 * render_size)), 10, (255, 0, 0), -1)
    #
    #     for landmark_pos in self._mu:
    #         cv2.circle(canvas, (int(landmark_pos[0] * render_size), int(landmark_pos[1] * render_size)), 10,
    #                    (255, 0, 0), -1)
    #
    #     robot_pose = self._x.detach().numpy()
    #     # robot_pose = np.array([0, 5, np.pi * 0.0])
    #
    #     cv2.circle(canvas, (int(robot_pose[0] * render_size), int(robot_pose[1] * render_size)), 10, (0, 0, 255), -1)
    #     canvas = cv2.arrowedLine(canvas, (int(robot_pose[0] * render_size), int(robot_pose[1] * render_size)),
    #                              (int(robot_pose[0] * render_size + arrow_length * np.cos(robot_pose[2])),
    #                               int(robot_pose[1] * render_size + arrow_length * np.sin(robot_pose[2]))),
    #                              (0, 0, 255), 2, tipLength=0.5)
    #
    #     FoV_corners = np.array([(int(robot_pose[0] * render_size), int(robot_pose[1] * render_size)),
    #                             (int((robot_pose[0] + self._radius *
    #                                   np.cos(robot_pose[2] + self._psi) / np.cos(self._psi)) * render_size),
    #                              int((robot_pose[1] + self._radius *
    #                                   np.sin(robot_pose[2] + self._psi) / np.cos(self._psi)) * render_size)),
    #                             (int((robot_pose[0] + self._radius *
    #                                   np.cos(robot_pose[2] - self._psi) / np.cos(self._psi)) * render_size),
    #                              int((robot_pose[1] + self._radius *
    #                                   np.sin(robot_pose[2] - self._psi) / np.cos(self._psi)) * render_size))])
    #
    #     cv2.polylines(canvas, [FoV_corners], isClosed=True, color=(0, 255, 0), thickness=2)
    #
    #     cv2.namedWindow('map', cv2.WINDOW_GUI_NORMAL)
    #     cv2.imshow('map', canvas)
    #     # cv2.resizeWindow('map', *render_size)
    #     cv2.waitKey(100)

    def _plot(self, legend, title='trajectory'):
        self.landmarks = self._mu_real.flatten().detach().numpy().reshape(self._num_landmarks*2, 1)

        # plot agent trajectory
        plt.tick_params(labelsize=15)
        history_poses = np.array(self.history_poses)  # with the shape of (self._num_landmarks, 2)
        self.ax.plot(history_poses[:, 0], history_poses[:, 1], c='black', linewidth=3, label='agent trajectory')

        # plot agent trajectory start & end
        self.ax.scatter(history_poses[0, 0], history_poses[0, 1], marker='>', s=70, c='red', label="start")
        self.ax.scatter(history_poses[-1, 0], history_poses[-1, 1], marker='s', s=70, c='red', label="end")

        self.ax.scatter(history_poses[-1, 0] + np.cos(history_poses[-1, 2])*0.5,
                     history_poses[-1, 1] + np.sin(history_poses[-1, 2])*0.5, marker='o', c='black')

        # plot landmarks
        self.ax.scatter(self.landmarks[list(range(0, self._num_landmarks*2, 2)), :],
                        self.landmarks[list(range(1, self._num_landmarks*2+1, 2)), :], s=50, c='blue', label="landmark")

        # axes
        self.ax.set_xlabel("x", fontdict={'size': 20})
        self.ax.set_ylabel("y", fontdict={'size': 20})

        # title
        # self.ax.set_title(title, fontdict={'size': 16})

        self.ax.set_facecolor('whitesmoke')
        plt.grid(alpha=0.4)
        # legend
        if legend == True:
            self.ax.legend()
            plt.legend(prop={'size': 14})


    def render(self, mode='human'):
        self.ax.cla()

        # plot
        self._plot(True)

        # display
        plt.draw()
        plt.pause(0.2)

    def save_plot(self, name='default.png', title='trajectory', legend=False):
        self.ax.cla()
        self._plot(legend, title=title)
        self.fig.savefig(name, bbox_inches = 'tight')

    def close (self):
        plt.close('all')


class SimpleEnvAtt:
    def __init__(self, max_num_landmarks, horizon, tau, A, B, V, W, landmark_motion_scale, psi, radius):
        self._max_num_landmarks = max_num_landmarks

        self._tau = tau
        self._A = A
        self._B = B
        self._V = V  # sensor_std ** 2 with the shape of (2, )
        self._W = W  # motion_std ** 2 with the shape of (2, )
        self._landmark_motion_scale = landmark_motion_scale

        self._mu = None
        self._v = None
        self._landmark_motion_bias = None
        self._x = None
        self._step_num = None

        self._psi = psi
        self._radius = radius

    def reset(self):
        self._num_landmarks = torch.randint(3, 7, (1, )).item()
        # self._env_size = tensor([self._num_landmarks * 4, self._num_landmarks * 4])
        # self._horizon = self._num_landmarks * 3
        self._env_size = tensor([2000, 2000])
        self._horizon = 120
        mu = (torch.rand((self._num_landmarks, 2)) - 0.5) * self._env_size

        landmark_motion_bias = (torch.rand(2) - 0.5) * 1.6
        v = (torch.rand((self._num_landmarks, 2)) - 0.5) * self._landmark_motion_scale + landmark_motion_bias

        x = torch.empty(3)
        x[:2] = (torch.rand(2) - 0.5) * self._env_size * 1.25
        x[2] = (torch.rand(1) * 2 - 1) * torch.pi

        self._mu_real = mu
        self._v = v
        self._landmark_motion_bias = landmark_motion_bias
        self._x = x
        self._step_num = 0

        self.history_poses = [self._x.detach().numpy().tolist()]
        self.fig = plt.figure(1)
        self.ax = self.fig.gca()

        return self._mu_real, v, x, False

    def step(self, action: tensor) -> Tuple[tensor, tensor, tensor, bool]:
        self._x = SE2_kinematics(self._x, action, self._tau)
        # self._x[:2] = torch.clip(self._x[:2], min=torch.zeros(2), max=self._env_size)

        # self._mu_real = torch.clip(landmark_motion_real(self._mu_real, self._v, self._A, self._B, self._W),
        #                            min=-self._env_size/2, max=self._env_size/2)
        self._mu_real = landmark_motion_real(self._mu_real, self._v, self._A, self._B, self._W)

        self._v = (torch.rand((self._num_landmarks, 2)) - 0.5) * self._landmark_motion_scale + \
                  self._landmark_motion_bias

        done = False
        self._step_num += 1
        if self._step_num >= self._horizon:
            done = True

        self.history_poses.append(self._x.detach().numpy().tolist())

        return self._mu_real, self._v, self._x, done

    def _plot(self, legend, title='trajectory'):
        self.landmarks = self._mu_real.flatten().detach().numpy().reshape(self._num_landmarks*2, 1)

        # plot agent trajectory
        plt.tick_params(labelsize=15)
        history_poses = np.array(self.history_poses)  # with the shape of (self._num_landmarks, 2)
        self.ax.plot(history_poses[:, 0], history_poses[:, 1], c='black', linewidth=3, label='agent trajectory')

        # plot agent trajectory start & end
        self.ax.scatter(history_poses[0, 0], history_poses[0, 1], marker='>', s=70, c='red', label="start")
        self.ax.scatter(history_poses[-1, 0], history_poses[-1, 1], marker='s', s=70, c='red', label="end")

        self.ax.scatter(history_poses[-1, 0] + np.cos(history_poses[-1, 2])*0.5,
                     history_poses[-1, 1] + np.sin(history_poses[-1, 2])*0.5, marker='o', c='black')

        # plot landmarks
        self.ax.scatter(self.landmarks[list(range(0, self._num_landmarks*2, 2)), :],
                        self.landmarks[list(range(1, self._num_landmarks*2+1, 2)), :], s=50, c='blue', label="landmark")

        # axes
        self.ax.set_xlabel("x", fontdict={'size': 20})
        self.ax.set_ylabel("y", fontdict={'size': 20})

        # title
        # self.ax.set_title(title, fontdict={'size': 16})

        self.ax.set_facecolor('whitesmoke')
        plt.grid(alpha=0.4)
        # legend
        if legend == True:
            self.ax.legend()
            plt.legend(prop={'size': 14})


    def render(self, mode='human'):
        self.ax.cla()

        # plot
        self._plot(True)

        # display
        plt.draw()
        plt.pause(0.3)

    def save_plot(self, name='default.png', title='trajectory', legend=False):
        self.ax.cla()
        self._plot(legend, title=title)
        self.fig.savefig(name, bbox_inches = 'tight')

    def close (self):
        plt.close('all')

class SimpleEnvAtt2D:
    def __init__(self, max_num_landmarks, horizon, tau, init_width, init_height, A, B, V, W, landmark_motion_scale, radius):
        self._max_num_landmarks = max_num_landmarks

        self._tau = tau
        self._A = A
        self._B = B
        self._V = V  # sensor_std ** 2 with the shape of (2, )
        self._W = W  # motion_std ** 2 with the shape of (2, )
        self._landmark_motion_scale = landmark_motion_scale
        self._horizon = horizon

        self._mu = None
        self._v = None
        self._landmark_motion_bias = None
        self._x = None
        self._step_num = None
        self._env_size = tensor([init_width, init_height])

        self._radius = radius

    def reset(self):
        self._num_landmarks = torch.randint(1, 6, (1, )).item()
        # self._num_landmarks = 4
        # print(f"num_landmarks is {self._num_landmarks}")
        # self._env_size = tensor([self._num_landmarks * 4, self._num_landmarks * 4])
        # self._horizon = self._num_landmarks * 3
        mu = (torch.rand((self._num_landmarks, 2)) - 0.5) * self._env_size

        v = (torch.rand((self._num_landmarks, 2)) - 0.5) * 2 * self._landmark_motion_scale

        x = torch.empty(3)
        x[:2] = tensor([0, 0])
        x[2] = (torch.rand(1) * 2 - 1) * torch.pi

        self._mu_real = mu
        self._v = v
        self._x = x
        self._step_num = 0

        self.history_poses = [self._x.detach().numpy().tolist()]
        self.fig = plt.figure(1)
        self.ax = self.fig.gca()

        return self._mu_real, v, x, False

    def step(self, action: tensor) -> Tuple[tensor, tensor, tensor, bool]:
        self._x = Simple_kinematics(self._x, action, self._tau)
        self._mu_real = landmark_motion_real(self._mu_real, self._v, self._A, self._B, self._W)

        done = False
        self._step_num += 1
        if self._step_num >= self._horizon:
            done = True

        self.history_poses.append(self._x.detach().numpy().tolist())

        return self._mu_real, self._v, self._x, done

    def _plot(self, legend, title='trajectory'):
        self.landmarks = self._mu_real.flatten().detach().numpy().reshape(self._num_landmarks*2, 1)

        # plot agent trajectory
        plt.tick_params(labelsize=15)
        history_poses = np.array(self.history_poses)  # with the shape of (self._num_landmarks, 2)
        self.ax.plot(history_poses[:, 0], history_poses[:, 1], c='black', linewidth=3, label='agent trajectory')

        # plot agent trajectory start & end
        self.ax.scatter(history_poses[0, 0], history_poses[0, 1], marker='>', s=70, c='red', label="start")
        self.ax.scatter(history_poses[-1, 0], history_poses[-1, 1], marker='s', s=70, c='red', label="end")

        self.ax.scatter(history_poses[-1, 0] + np.cos(history_poses[-1, 2])*0.5,
                     history_poses[-1, 1] + np.sin(history_poses[-1, 2])*0.5, marker='o', c='black')

        # plot landmarks
        self.ax.scatter(self.landmarks[list(range(0, self._num_landmarks*2, 2)), :],
                        self.landmarks[list(range(1, self._num_landmarks*2+1, 2)), :], s=50, c='blue', label="landmark")

        # axes
        self.ax.set_xlabel("x", fontdict={'size': 20})
        self.ax.set_ylabel("y", fontdict={'size': 20})

        # title
        # self.ax.set_title(title, fontdict={'size': 16})

        self.ax.set_facecolor('whitesmoke')
        plt.grid(alpha=0.4)
        # legend
        if legend == True:
            self.ax.legend()
            plt.legend(prop={'size': 14})


    def render(self, mode='human'):
        self.ax.cla()

        # plot
        self._plot(True)

        # display
        plt.draw()
        plt.pause(0.3)

    def save_plot(self, name='default.png', title='trajectory', legend=False):
        self.ax.cla()
        self._plot(legend, title=title)
        self.fig.savefig(name, bbox_inches = 'tight')

    def close (self):
        plt.close('all')

class TargetMotion4D:
    def __init__(self, tau, init_cov_mat, prop_cov_mat, target_id):
        self.target_id = target_id
        self.dt = tau
        self.cov_init = init_cov_mat
        self.Q = prop_cov_mat
        self.init_dist = torch.distributions.MultivariateNormal(torch.zeros(4), self.cov_init)
        self.noise_dist = torch.distributions.MultivariateNormal(torch.zeros(4), self.Q)
        self.F = tensor([[1, 0, self.dt, 0],
                 [0, 1, 0, self.dt],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
        
        self.init_mean = None
        self.state = None

    def reset_target(self, reset_mean):
        state_offset = self.init_dist.sample()
        
        self.init_mean = reset_mean
        self.state = reset_mean + state_offset

    def move_target(self):
        move_noise = self.noise_dist.sample()
        self.state = self.F @ self.state + move_noise

class SimpleEnvAtt4D:
    def __init__(self, max_num_landmarks, horizon, tau,
                 init_pos_range, init_vel_range,
                 q_min, q_max,
                 init_pos_cov_min, init_pos_cov_max,
                 init_vel_cov_min, init_vel_cov_max,
                 radius):
        
        self._max_num_landmarks = max_num_landmarks

        self._max_steps = horizon
        self._tau = tau
        self._init_pos_range = init_pos_range
        self._init_vel_range = init_vel_range
        self._q_min = q_min
        self._q_max = q_max
        self._init_pos_cov_min = init_pos_cov_min
        self._init_pos_cov_max = init_pos_cov_max
        self._init_vel_cov_min = init_vel_cov_min
        self._init_vel_cov_max = init_vel_cov_max

        self._mu_real = []
        self._target_config = []
        self._x = None
        self._step_num = None

        self._radius = radius
        self._target_list = []

    def reset(self):
        self._num_landmarks = torch.randint(1, self._max_num_landmarks + 1, (1, )).item()

        # init target param: init cov and motion noise cov
        self._target_list = []
        for i in range(self._num_landmarks):
            x_cov = np.random.uniform(self._init_pos_cov_min, self._init_pos_cov_max)
            y_cov = np.random.uniform(self._init_pos_cov_min, self._init_pos_cov_max)
            vx_cov = np.random.uniform(self._init_vel_cov_min, self._init_vel_cov_max)
            vy_cov = np.random.uniform(self._init_vel_cov_min, self._init_vel_cov_max)
            init_cov = tensor([[x_cov, 0, 0, 0],
                               [0, y_cov, 0, 0],
                               [0, 0, vx_cov, 0],
                               [0, 0, 0, vy_cov]])
            
            q = np.random.uniform(self._q_min, self._q_max)
            dt = self._tau
            noise_cov = q * tensor([[dt**3 / 3, 0, dt**2 / 2, 0],
                                    [0, dt**3 / 3, 0, dt**2 / 2],
                                    [dt**2 / 2, 0, dt, 0],
                                    [0, dt**2 / 2, 0, dt]])
            
            target = TargetMotion4D(tau=self._tau, init_cov_mat=init_cov, prop_cov_mat=noise_cov, target_id=i + 1)

            reset_x = np.random.uniform(self._init_pos_range[0], self._init_pos_range[1])
            reset_y = np.random.uniform(self._init_pos_range[0], self._init_pos_range[1])
            reset_vx = np.random.uniform(self._init_vel_range[0], self._init_vel_range[1])
            reset_vy = np.random.uniform(self._init_vel_range[0], self._init_vel_range[1])
            target.reset_target(tensor([reset_x, reset_y, reset_vx, reset_vy]))

            self._target_list.append(target)

        x = torch.empty(3)
        x[:2] = torch.tensor([0, 0])
        x[2] = (torch.rand(1) * 2 - 1) * torch.pi

        self._x = x
        self._step_num = 0
        self.history_poses = [self._x.detach().numpy().tolist()]
        self.fig = plt.figure(1)
        self.ax = self.fig.gca()

        self._mu_real = torch.vstack([tgt.state for tgt in self._target_list])

        return self._mu_real, self._target_list, x, False

    def reset_w_config(self, target_list):
        for target in target_list:
            self._target_list.append(target)
        
        x = torch.empty(3)
        x[:2] = torch.tensor([0, 0])
        x[2] = (torch.rand(1) * 2 - 1) * torch.pi

        self._x = x
        self._step_num = 0
        self.history_poses = [self._x.detach().numpy().tolist()]
        self.fig = plt.figure(1)
        self.ax = self.fig.gca()

        self._mu_real = torch.vstack([tgt.state for tgt in self._target_list])

        return self._mu_real, self._target_list, x, False

    def step(self, action: tensor) -> Tuple[tensor, tensor, tensor, bool]:
        # print(f"before kinematics x = {self._x}")
        self._x = Simple_kinematics(self._x, action, self._tau)
        # print(f"after kinematics x = {self._x}")
        for target in self._target_list:
            target.move_target()
        
        self._mu_real = torch.vstack([tgt.state for tgt in self._target_list])

        done = False
        self._step_num += 1

        if self._step_num >= self._max_steps:
            done = True

        self.history_poses.append(self._x.detach().numpy().tolist())

        return self._mu_real, self._x, done

    def _plot(self, legend, title='trajectory'):
        self.landmarks = self._mu_real.flatten().detach().numpy().reshape(self._num_landmarks*2, 1)

        # plot agent trajectory
        plt.tick_params(labelsize=15)
        history_poses = np.array(self.history_poses)  # with the shape of (self._num_landmarks, 2)
        self.ax.plot(history_poses[:, 0], history_poses[:, 1], c='black', linewidth=3, label='agent trajectory')

        # plot agent trajectory start & end
        self.ax.scatter(history_poses[0, 0], history_poses[0, 1], marker='>', s=70, c='red', label="start")
        self.ax.scatter(history_poses[-1, 0], history_poses[-1, 1], marker='s', s=70, c='red', label="end")

        self.ax.scatter(history_poses[-1, 0] + np.cos(history_poses[-1, 2])*0.5,
                        history_poses[-1, 1] + np.sin(history_poses[-1, 2])*0.5, marker='o', c='black')

        # plot landmarks
        self.ax.scatter(self.landmarks[list(range(0, self._num_landmarks*2, 2)), :],
                        self.landmarks[list(range(1, self._num_landmarks*2+1, 2)), :], s=50, c='blue', label="landmark")

        # axes
        self.ax.set_xlabel("x", fontdict={'size': 20})
        self.ax.set_ylabel("y", fontdict={'size': 20})

        # plot believe ellipses
        # TODO: implement later

        # title
        # self.ax.set_title(title, fontdict={'size': 16})

        self.ax.set_facecolor('whitesmoke')
        plt.grid(alpha=0.4)
        # legend
        if legend == True:
            self.ax.legend()
            plt.legend(prop={'size': 14})


    def render(self, target_beliefs, mode='human'):
        self.ax.cla()

        # plot
        self._plot(True)

        # display
        plt.draw(target_beliefs)
        plt.pause(0.3)

    def save_plot(self, name='default.png', title='trajectory', legend=False):
        self.ax.cla()
        self._plot(legend, title=title)
        self.fig.savefig(name, bbox_inches = 'tight')

    def close (self):
        plt.close('all')