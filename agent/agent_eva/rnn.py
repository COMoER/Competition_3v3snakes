from agent.agent_eva.common import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import os

class Agent(nn.Module):
    def __init__(self, in_feature, out_feature):
        """
        Args:
            in_feature:feature of observation and action (and agent one-hot embedding)
        """
        super(Agent, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.fc1 = nn.Linear(in_feature, HIDDEN_SIZE)
        # GRU
        self.GRU = nn.GRUCell(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, out_feature)

    def init_hidden(self):
        """
        Because the hidden size is needed to expand according to the input tensor,
        so there just generate the size 1 hidden
        """
        return self.fc1.weight.new_zeros((1, HIDDEN_SIZE))

    def forward(self, obs, h):
        """
        - B batch_size
        - N number of agents( all agents share the weight)

        Args:
            obs: (Any,in_feature)
            h:(Any..Any,hidden_size)
        Return:
            h_next:(Any,hidden_feature)
            out: (Any,out_feature)
        """
        h = h.reshape(-1, HIDDEN_SIZE)
        x = self.fc1(obs)
        x = F.relu(x)
        h_next = self.GRU(x, h)
        out = self.fc2(h_next)
        return h_next, out

    def update(self, agent):
        """
        - only for target agents
        - every 200 episode, update the target network param as in DQN
        Args:
            agent: the agent for training
        """
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)

class RNNQMIX:
    def __init__(self, obs_dim, act_dim, num_agent):
        self.obs_dim = obs_dim + num_agent  # include one hot encode
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device

        # Initialize the agents
        self.agents = Agent(self.obs_dim, act_dim).to(device)  # all agents share the same AgentNet

        self.hidden_layer = None

    def reset(self):
        """
        reset the hidden layer of each rnn when every episode begin
        """
        self.hidden_layer = self.agents.init_hidden()
        self.hidden_layer = self.hidden_layer.expand(self.num_agent, -1)
    def update_hidden(self,h):
        self.hidden_layer = h
    def choose_action_global(self,obs):
        with torch.no_grad():
            obs = OneHot(obs)
            N, _ = obs.shape
            obs = torch.FloatTensor([obs]).to(self.device).reshape(self.num_agent, -1)
            self.hidden_layer, out = self.agents(obs, self.hidden_layer)
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
            h, out = self.agents(obs, self.hidden_layer)
            q_value = out.cpu().detach().numpy()
            q_value[index][action_available == 0] = -99999
            action_greedy = np.argmax(q_value, axis=1)

        return action_greedy,h


    def load_model(self, run_dir, episode):
        filename = os.path.join(run_dir, "qmix_agent_%d.pth" % episode)
        f = torch.load(filename,device)
        if 'agents' in f.keys():
            self.agents.load_state_dict(f['agents'])
        else:
            self.agents.load_state_dict(f)
        self.agents.eval()