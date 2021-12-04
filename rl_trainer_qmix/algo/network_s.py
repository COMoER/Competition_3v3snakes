from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 64

# the implement of qmix refer to https://github.com/Gouet/QMIX-Starcraft.git
class state_encoder(nn.Module):
    """
    Since the input global state is a matrix input, so we should firstly encode the matrix to a feature vector
    """
    def __init__(self,width,height,out_feature,mode):
        super(state_encoder, self).__init__()
        self.mode = mode
        if mode == 1:
            # one_hot conv
            self.conv = nn.Conv2d(7,10,3)
            self.bn = nn.BatchNorm2d(10)
            self.pool = nn.MaxPool2d(2,2)
            self.fc = nn.Linear((width//2-1)*(height//2-1)*10,out_feature)
        elif mode == 2:
            # conv
            self.conv = nn.Conv2d(1, 4, kernel_size=3)
            self.bn = nn.BatchNorm2d(4)
            self.pool = nn.MaxPool2d(2, 2)  # 4*9*4
            self.fc = nn.Linear((height // 2 - 1) * (width // 2 - 1) * 4, out_feature)
        elif mode == 3:
            # fc
            self.fc = nn.Linear(width * height, out_feature)

    def forward(self,state):
        """
        Args:
            state: (B,C,H,W)
        """
        B = state.shape[0]
        if self.mode < 3:
            x = F.relu(self.bn(self.conv(state)))
            x = self.pool(x)
            x = x.view(B, -1)
            x = F.relu(self.fc(x))
        else:
            x = state.view(B,-1)
            x = F.relu(self.fc(x))
        return x

    def soft_update(self, agent, tau):
        """
        - only for target agents
        Args:
            agent: the agent for training
        """
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data * (1.0 - tau) + param.data * tau

    def hard_update(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)
class MixingNet(nn.Module):
    def __init__(self,num_agent,state_dim,hidden_dim):
        super(MixingNet, self).__init__()
        self.n_agent = num_agent
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        # affine transformation generate the weights on num_agent qs to hidden layer
        self.hypernetwork_w1 = nn.Linear(state_dim,hidden_dim*num_agent)
        # bias for mixing network layer1
        self.hypernetwork_b1 = nn.Linear(state_dim,hidden_dim)

        self.hypernetwork_w2 = nn.Linear(state_dim,hidden_dim)
        # as the reference mixing network layer2 bia treat as V(s) as REINFORCE with baseline
        self.V = nn.Sequential(nn.Linear(state_dim,hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim,1)
                               )

    def forward(self,qs,state):
        """
        Mixing Net receive the global state and the Q_i for each agent

        Args:
            qs:(B,T,N)
            state:(B,T,state_dim)
        """
        B = qs.shape[0]
        T = qs.shape[1]
        w1 = torch.abs(self.hypernetwork_w1(state)).view(B*T,self.hidden_dim,self.n_agent)
        b1 = self.hypernetwork_b1(state).view(B*T,self.hidden_dim,1)
        w2 = torch.abs(self.hypernetwork_w2(state)).view(B*T,1,self.hidden_dim)
        v = self.V(state).view(B*T,1,1)
        qs = qs.view(B*T,-1,1)
        h = F.elu(torch.bmm(w1,qs) + b1)
        q = torch.bmm(w2,h)
        q = q + v
        return q.view(B,T)

    def soft_update(self, agent, tau):
        """
        - only for target agents
        Args:
            agent: the agent for training
        """
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data * (1.0 - tau) + param.data * tau

    def hard_update(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)



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
    def soft_update(self,agent,tau):
        """
        - only for target agents
        Args:
            agent: the agent for training
        """
        for param,target_param in zip(agent.parameters(),self.parameters()):
            target_param.data * (1.0 - tau) + param.data * tau

    def hard_update(self,agent):
        for param,target_param in zip(agent.parameters(),self.parameters()):
            target_param.data.copy_(param.data)