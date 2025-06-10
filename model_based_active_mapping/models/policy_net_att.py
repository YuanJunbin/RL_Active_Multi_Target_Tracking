import torch

from torch import nn


class PolicyNetAtt(nn.Module):

    def __init__(self,
                 input_dim: int,
                 policy_dim: int = 2):

        super(PolicyNetAtt, self).__init__()

        self.num_landmark = int((input_dim - 3) / 5)

        self.agent_pos_fc1_pi = nn.Linear(3, 32)
        self.agent_pos_fc2_pi = nn.Linear(32, 32)
        self.landmark_fc1_pi = nn.Linear(4, 64)
        self.landmark_fc2_pi = nn.Linear(64, 32)
        self.info_fc1_pi = nn.Linear(64, 64)
        self.action_fc1_pi = nn.Linear(64, 64)
        self.action_fc2_pi = nn.Linear(64, policy_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    # def forward(self, observation: torch.Tensor) -> torch.Tensor:
    #     if len(observation.size()) == 1:
    #         observation = observation[None, :]
    #
    #     # compute the policy
    #     # embeddings of agent's position
    #     agent_pos_embedding = self.relu(self.agent_pos_fc1_pi(observation[:, :3]))
    #     agent_pos_embedding = self.relu(self.agent_pos_fc2_pi(agent_pos_embedding))
    #
    #     # embeddings of landmarks
    #     estimated_landmark_pos = observation[:, 3:]
    #     landmark_embedding = self.relu(self.landmark_fc1_pi(estimated_landmark_pos))
    #     landmark_embedding = self.relu(self.landmark_fc2_pi(landmark_embedding))
    #
    #     # attention
    #     attention = torch.dot(landmark_embedding.squeeze(), agent_pos_embedding.squeeze()) / 4
    #     att = self.softmax(attention)
    #     landmark_embedding_att = self.relu(att * landmark_embedding.squeeze())
    #
    #     info_embedding = self.relu(self.info_fc1_pi(torch.cat((agent_pos_embedding.squeeze(),
    #                                                            landmark_embedding_att))))
    #     action = self.tanh(self.action_fc1_pi(info_embedding))
    #     action = self.action_fc2_pi(action)
    #
    #     if action.size()[0] == 1:
    #         action = action.flatten()
    #
    #     return action

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # print("[DEBUG] agent_pos_fc1_pi weight: min=", self.agent_pos_fc1_pi.weight.min().item(), "max=", self.agent_pos_fc1_pi.weight.max().item())
        # print("[DEBUG] agent_pos_fc1_pi bias: min=", self.agent_pos_fc1_pi.bias.min().item(), "max=", self.agent_pos_fc1_pi.bias.max().item())
        if len(observation.size()) == 1:
            observation = observation[None, :]

        # print(f"[DEBUG] Obs range: min={observation.min().item():.3f}, max={observation.max().item():.3f}")


        # compute the policy
        # embeddings of agent's position
        agent_pos_embedding = self.relu(self.agent_pos_fc1_pi(observation[:, :3]))
        agent_pos_embedding = self.relu(self.agent_pos_fc2_pi(agent_pos_embedding))
        # print(f"[DEBUG] Agent pos embedding: {agent_pos_embedding}")

        # embeddings of landmarkss
        info_vector = observation[:, 3: 3 + 2 * self.num_landmark]
        estimated_landmark_pos = observation[:, 3 + 2 * self.num_landmark: - self.num_landmark]
        landmark_info = torch.cat((estimated_landmark_pos.reshape(observation.size()[0], self.num_landmark, 2),
                                   info_vector.reshape(observation.size()[0], self.num_landmark, 2)), 2)
        landmark_embedding = self.relu(self.landmark_fc1_pi(landmark_info))
        landmark_embedding = self.relu(self.landmark_fc2_pi(landmark_embedding))
        # print(f"[DEBUG] Landmark emb: shape={landmark_embedding.shape}, min={landmark_embedding.min().item():.3e}, max={landmark_embedding.max().item():.3e}")

        # attention
        landmark_embedding_tr = torch.transpose(landmark_embedding, 1, 2)

        # mask
        mask = observation[:, - self.num_landmark:].unsqueeze(1)
        attention = torch.matmul(agent_pos_embedding.unsqueeze(1), landmark_embedding_tr) / 4
        attention = attention.masked_fill(mask == 0, -1e10)
        # print(f"[DEBUG] Attention raw min/max: {attention.min().item():.3e}/{attention.max().item():.3e}")


        att = self.softmax(attention)
        # print(f"[DEBUG] Attention weights: {att.squeeze().detach().cpu().numpy()}")

        landmark_embedding_att = self.relu((torch.matmul(att, torch.transpose(landmark_embedding_tr, 1, 2)).squeeze(1)))

        info_embedding = self.relu(self.info_fc1_pi(torch.cat((agent_pos_embedding, landmark_embedding_att), 1)))
        action = self.tanh(self.action_fc1_pi(info_embedding))
        action = self.tanh(self.action_fc2_pi(action))

        if action.size()[0] == 1:
            action = action.flatten()
        
        # print(f"[DEBUG] Final action: {action.detach().cpu().numpy()}")

        # scaled_action = torch.hstack(((1 + action[0]) * 2.0, action[1] * torch.pi/3))
        scaled_action = torch.hstack(((1 + action[0]) * 15.1, action[1] * torch.pi))

        return scaled_action
