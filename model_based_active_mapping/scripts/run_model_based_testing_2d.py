import os, sys, yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
# torch.manual_seed(0)
import argparse

from torch import tensor
from envs.simple_env import SimpleEnvAtt2D
from agents.model_based_agent import ModelBasedAgentAtt2D
import numpy as np
from matplotlib.patches import Ellipse, Circle
from scipy.stats import chi2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='model-based mapping')
parser.add_argument('--network-type', type=int, default=1, help='by default, it should attention block,'
                                                                'otherwise, it would be MLP')
args = parser.parse_args()

def run_model_based_testing(params_filename):
    assert os.path.exists(params_filename)
    with open(os.path.join(params_filename)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    max_num_landmarks = params['max_num_landmarks']
    num_landmarks = params['num_landmarks']
    horizon = params['horizon']
    env_width = params['env_width']
    env_height = params['env_height']

    tau = params['tau']

    A = torch.zeros((2, 2))
    A[0, 0] = params['motion']['A']['_1']
    A[1, 1] = params['motion']['A']['_2']

    B = torch.zeros((2, 2))
    B[0, 0] = params['motion']['B']['_1']
    B[1, 1] = params['motion']['B']['_2']

    W = torch.zeros(2)
    W[0] = params['motion']['W']['_1']
    W[1] = params['motion']['W']['_2']

    landmark_motion_scale = params['motion']['landmark_motion_scale']

    init_info = params['init_info']

    radius = params['FoV']['radius']
    kappa = params['FoV']['kappa']

    V = torch.zeros(2)
    V[0] = params['FoV']['V']['_1']
    V[1] = params['FoV']['V']['_2']

    lr = params['lr']
    max_epoch = params['max_epoch']
    batch_size = params['batch_size']
    num_test_trials = params['num_test_trials']

    visualizer = visualizer_env_with_belief(num_landmarks)

    env = SimpleEnvAtt2D(max_num_landmarks=max_num_landmarks, horizon=horizon, tau=tau, init_width = env_width, init_height = env_height,
                        A=A, B=B, V=V, W=W, landmark_motion_scale=landmark_motion_scale, radius=radius)
    
    agent = ModelBasedAgentAtt2D(max_num_landmarks=max_num_landmarks, init_info=init_info, A=A, B=B, W=W,
                                radius=radius, kappa=kappa, V=V, lr=lr)

    agent.load_policy_state_dict('./checkpoints/test_train_2d.pth')

    agent.eval_policy()

    mu_real, v, x, done = env.reset()
    num_landmarks = mu_real.size()[0]
    agent.reset_estimate_mu(mu_real)
    agent.reset_agent_info()
    visualizer.render(env, agent)
    while not done:
        # action = agent.plan(v, x)
        _ = agent.plan(v, x)
        angle = np.random.uniform(-np.pi, np.pi)
        action = tensor([30, angle])

        mu_real, v, x, done = env.step(action)
        agent.update_info_mu(mu_real, x)
        visualizer.render(env, agent)

    reward = agent.update_policy_grad(False) / num_landmarks
    print("num_landmark:", num_landmarks, "reward:", reward)

class visualizer_env_with_belief:
    def __init__(self, num_targets):
        self.fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        self.ax_main = axs[0]
        self.ax_plot = axs[1]
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True)

        self.num_targets = num_targets
        self.targets_traj = [[[], []] for _ in range(num_targets)]
        self.U_history = []
    
    def render(self, env, agent):
        self.ax_main.clear()
        self.ax_plot.clear()
        
        # plot agent
        agent_pose = env._x
        self.ax_main.scatter(agent_pose[0].item(), agent_pose[1].item(), color='black', s=25, marker='o')
        
        # plot agent trajectory
        agent_traj = np.array(env.history_poses)
        self.ax_main.plot(agent_traj[:, 0], agent_traj[:, 1], c='black', linewidth=3, label='agent trajectory')

        target_state_list = env._mu_real

        # print(f"state lise is {target_state_list}")

        for i in range(target_state_list.size()[0]):
            self.targets_traj[i][0].append(target_state_list[i, 0].item())
            self.targets_traj[i][1].append(target_state_list[i, 1].item())

        # plot targets and belief
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_targets))
        
        confidence_levels = [0.4, 0.9, 0.998]
        chi_square_values = [chi2.ppf(confidence, 2) for confidence in confidence_levels]
        alphas = [0.8, 0.5, 0.2]
        uncertainty_value = 0

        for j in range(target_state_list.size()[0]):
            self.ax_main.plot(target_state_list[j, 0].item(), target_state_list[j, 1].item(), 'o', color=colors[j], markeredgecolor='k', markersize=8, label=f'Target {j + 1}')
            self.ax_main.plot(self.targets_traj[j][0], self.targets_traj[j][1], '-', color=colors[j], markersize=2)

            cov_j = torch.linalg.inv(torch.diag(agent._info[j])).detach().numpy()
            mean_j = agent._mu_update[j]
            uncertainty_value = uncertainty_value + np.log(np.linalg.det(cov_j))

            for chi_square_value, alpha in zip(chi_square_values, alphas):
                eigenvalues, eigenvectors = np.linalg.eig(cov_j)
                angle = np.arctan2(*eigenvectors[:,0][::-1])
                
                width, height = 2 * np.sqrt(eigenvalues * chi_square_value)
                
                ellipse = Ellipse(xy=(mean_j[0].item(), mean_j[1].item()), width=width, height=height, angle=np.degrees(angle), edgecolor=None, facecolor=colors[j], lw=1, alpha=alpha)
                self.ax_main.add_patch(ellipse)
        
        circle = Circle((agent_pose[0].item(), agent_pose[1].item()), agent._radius, color='black', fill=False, linestyle='-', label='Agent FOV', zorder=5)
        self.ax_main.add_patch(circle)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title("Agent & Target Tracking")
        self.ax_main.legend()

        self.U_history.append(uncertainty_value)
        self.ax_plot.plot(self.U_history, color='purple', linewidth=1.5)
        self.ax_plot.set_xlabel("Step")
        self.ax_plot.set_ylabel("U")
        self.ax_plot.grid(True)
        self.ax_plot.set_xlim(0, 1.2 * env._step_num)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

if __name__ == '__main__':
    # torch.manual_seed(0)
    # torch.autograd.set_detect_anomaly(True)
    run_model_based_testing(params_filename=os.path.join(os.path.abspath(os.path.join("", os.pardir)),
                                                          "params/params_test_2d.yaml"))
