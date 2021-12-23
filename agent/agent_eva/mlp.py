from agent.agent_eva.common import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import os

EPISODE = int(os.environ.get('EPISODE'))

run = int(os.environ.get("RUN"))

class Agent(nn.Module):
    def __init__(self,in_feature,out_feature):
        """
        Args:
            in_feature:feature of observation and action (and agent one-hot embedding)
        """
        super(Agent, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.fc1 = nn.Linear(in_feature,HIDDEN_SIZE)
        self.mlp = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE,out_feature)

    def forward(self,obs):
        """
        - B batch_size
        - N number of agents( all agents share the weight)

        Args:
            obs: (Any,in_feature)
        Return:
            out: (Any,out_feature)
        """
        x = self.fc1(obs)
        x1 = F.relu(x)
        x2 = F.relu(self.mlp(x1))
        out = self.fc2(x2)
        return out

class MLPQMIX:
    def __init__(self, obs_dim, act_dim, num_agent):
        self.obs_dim = obs_dim + num_agent  # include one hot encode
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device

        # Initialize the agents
        self.agents = Agent(self.obs_dim, act_dim).to(device)  # all agents share the same AgentNet

    def choose_action_global(self,obs):
        with torch.no_grad():
            obs = OneHot(obs)
            N, _ = obs.shape
            obs = torch.FloatTensor([obs]).to(self.device).reshape(self.num_agent, -1)
            out = self.agents(obs)
            q_value = out.cpu().detach().numpy()
            action_greedy = np.argmax(q_value, axis=1)
            return action_greedy

    def choose_action(self, obs,action_available,index):
        """
        Args:
            obs:(N,obs_feature) not using one_hot to encode distinct agents
        Returns:
            logits: (N) the chosen action using epsilon-greedy
        """
        with torch.no_grad():
            obs = OneHot(obs)
            N, _ = obs.shape
            obs = torch.FloatTensor([obs]).to(self.device).reshape(self.num_agent, -1)
            out = self.agents(obs)
            q_value = out.cpu().detach().numpy()
            q_value[index][action_available == 0] = -99999
            action_greedy = np.argmax(q_value, axis=1)

        return action_greedy


    def load_model(self, run_dir, episode):
        filename = os.path.join(run_dir, "qmix_agent_%d.pth" % episode)
        f = torch.load(filename,device)
        if 'agents' in f.keys():
            self.agents.load_state_dict(f['agents'])
        else:
            self.agents.load_state_dict(f)
        self.agents.eval()

