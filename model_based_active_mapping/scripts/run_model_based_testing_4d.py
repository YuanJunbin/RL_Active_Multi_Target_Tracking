import os, sys, yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
torch.manual_seed(0)
import argparse
import matplotlib.pyplot as plt
from torch import tensor
from envs.simple_env import SimpleEnvAtt4D, TargetMotion4D
from agents.model_based_agent import ModelBasedAgentAtt4D
from matplotlib.patches import Ellipse, Circle
from scipy.stats import chi2

parser = argparse.ArgumentParser(description='model-based mapping')
parser.add_argument('--network-type', type=int, default=1, help='by default, it should attention block,'
                                                                'otherwise, it would be MLP')
args = parser.parse_args()

def run_model_based_testing_4D(params_filename, target_traj_file = None):
    assert os.path.exists(params_filename)
    with open(os.path.join(params_filename)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    max_num_landmarks = params['max_num_landmarks']
    num_landmarks = params['num_landmarks']
    horizon = params['horizon']
    tau = params['tau']

    obs_cov = tensor(params['drone']['observation_cov'], dtype=torch.float32)
    radius = params['drone']['sensor_range']

    visualizer = visualizer_env_with_belief(num_landmarks)

    env = SimpleEnvAtt4D(max_num_landmarks=None, horizon=horizon, tau=tau,
                         init_pos_range=None, init_vel_range=None, q_min=None, q_max=None, 
                         init_pos_cov_min=None, init_pos_cov_max=None,
                         init_vel_cov_min=None, init_vel_cov_max=None,
                         radius=radius)
    agent = ModelBasedAgentAtt4D(max_num_landmarks=max_num_landmarks, radius=radius, obs_cov=obs_cov, lr=0.0003, kappa=0.4, dt=tau)

    agent.load_policy_state_dict('./checkpoints/best_model_seed0.pth')
    agent.eval_policy()
    
    target_list = []
    for target_config in params['targets']:
        target = TargetMotion4D(tau=tau, target_id=target_config['id'],
                                init_cov_mat=tensor(target_config['init_covariance']),
                                prop_cov_mat=tensor(target_config['prop_covariance']))
        
        if target_traj_file is None:
            target.reset_target(tensor([target_config['init_x'], target_config['init_y'], target_config['init_vx'], target_config['init_vy']]))
        
        target_list.append(target)

    mu_real, _, x, done = env.reset_w_config(target_list)
    
    agent.reset_estimate_mu(mu_real)
    agent.reset_agent_info(target_list, mu_real)
    visualizer.render(env, agent)
    while not done:
        _ = agent.plan(x)
        angle = np.random.uniform(-np.pi, np.pi)
        action = tensor([0, angle])
        mu_real, x, done = env.step(action)
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

        target_list = env._target_list
        for i, tgt in enumerate(target_list):
            self.targets_traj[i][0].append(tgt.state[0].item())
            self.targets_traj[i][1].append(tgt.state[1].item())

        # plot targets and belief
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_targets))
        
        confidence_levels = [0.4, 0.9, 0.998]
        chi_square_values = [chi2.ppf(confidence, 2) for confidence in confidence_levels]
        alphas = [0.8, 0.5, 0.2]
        uncertainty_value = 0

        for i, (target, color) in enumerate(zip(target_list, colors)):
            self.ax_main.plot(target.state[0].item(), target.state[1].item(), 'o', color=color, markeredgecolor='k', markersize=8, label=f'Target {target.target_id}')
            self.ax_main.plot(self.targets_traj[i][0], self.targets_traj[i][1], '-', color=color, markersize=2)

            cov_i = torch.linalg.inv(agent._info[i]).detach().numpy()
            mean_i = agent._mu_update[i]
            uncertainty_value = uncertainty_value + np.log(np.linalg.det(cov_i[:2, :2]))

            for chi_square_value, alpha in zip(chi_square_values, alphas):
                eigenvalues, eigenvectors = np.linalg.eig(cov_i[:2, :2])
                angle = np.arctan2(*eigenvectors[:,0][::-1])
                
                width, height = 2 * np.sqrt(eigenvalues * chi_square_value)
                
                ellipse = Ellipse(xy=(mean_i[0].item(), mean_i[1].item()), width=width, height=height, angle=np.degrees(angle), edgecolor=None, facecolor=color, lw=1, alpha=alpha)
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
    run_model_based_testing_4D(params_filename=os.path.join(os.path.abspath(os.path.join("", os.pardir)), "params/params_test_4d.yaml"))
