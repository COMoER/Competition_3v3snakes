import os
import torch
import numpy as np
# from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import device
from algo.network import MixingNet, Agent, state_encoder

EPSILON_MAX = 1.
EPSILON_MIN = .05
DELAY_TIMES = 5e4

MIXING_HIDDEN_SIZE = 32


def OneHot(obs: np.ndarray):
    """
    Args:
        obs:(B,N,obs_dim)
    Returns:
        obs:(B,N,obs_dim+1)
    """
    assert obs.ndim == 3
    B, N, _ = obs.shape
    encode = np.eye(N).reshape((1, N, N))  # one hot encoding
    encode = np.repeat(encode, B, axis=0)
    return np.concatenate([obs, encode], axis=2)


class QMIX:

    def __init__(self, obs_dim, act_dim, state_width_dim, state_height_dim, state_feature_dim, num_agent, args):
        self.obs_dim = obs_dim + num_agent  # include one hot encode
        self.act_dim = act_dim
        self.state_width_dim = state_width_dim
        self.state_height_dim = state_height_dim
        self.state_feature_dim = state_feature_dim
        self.num_agent = num_agent
        self.device = device
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.training = True

        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self._epsilon = EPSILON_MAX

        # Initialize the agents
        self.agents = Agent(self.obs_dim, act_dim).to(device)  # all agents share the same AgentNet
        self.state_encoder = state_encoder(state_width_dim, state_height_dim, state_feature_dim).to(device)
        self.mixing = MixingNet(num_agent, state_feature_dim, MIXING_HIDDEN_SIZE).to(device)

        self.target_agents = Agent(self.obs_dim, act_dim).to(device)  # target network
        self.target_state_encoder = state_encoder(state_width_dim, state_height_dim, state_feature_dim).to(device)
        self.target_mixing = MixingNet(num_agent, state_feature_dim, MIXING_HIDDEN_SIZE).to(device)

        self.hidden_layer = None
        self.target_hidden_layer = None
        # Initialise replay buffer R
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self.loss = None

        # at the beginning, update the target
        self.target_agents.update(self.agents)
        self.target_state_encoder.update(self.state_encoder)
        self.target_mixing.update(self.target_mixing)

        self.target_update_episode = args.update_target_episode

        # initialize the optimizer
        self.param = list(self.agents.parameters())
        self.param += self.state_encoder.parameters()
        self.param += self.mixing.parameters()
        self.optim = torch.optim.RMSprop(self.param, args.lr, alpha=0.99, eps=0.00001)
    def eval(self):
        self.training = False
        self.agents.eval()

    def reset(self, batch_size):
        """
        reset the hidden layer of each rnn when every episode begin
        """
        self.hidden_layer = self.agents.init_hidden().unsqueeze(0).expand(batch_size, self.num_agent, -1)
        self.target_hidden_layer = self.target_agents.init_hidden().unsqueeze(0).expand(batch_size, self.num_agent, -1)

    def epsilon_delay(self, delay_times):
        # update epsilon as paper
        rate = delay_times / DELAY_TIMES
        if self._epsilon > EPSILON_MIN:
            self._epsilon += rate * (EPSILON_MIN - EPSILON_MAX)

    def choose_action(self, obs, evaluate=False):
        """
        Args:
            obs:(B,N,obs_feature) not using one_hot to encode distinct agents
        Returns:
            logits: (B,N) the chosen action using epsilon-greedy
        """
        obs = OneHot(obs)
        B, N, _ = obs.shape
        epsilon = self._epsilon
        obs = torch.FloatTensor([obs]).to(self.device)
        obs = obs.view(B * N, -1)
        self.hidden_layer, out = self.agents(obs, self.hidden_layer)
        out = out.view(B, N, -1)
        action_greedy = np.argmax(out.cpu().detach().numpy(), axis=2)  # B,N
        if evaluate:
            return action_greedy
        else:
            action_random = np.random.randint(self.act_dim, size=self.num_agent, dtype=int)
            mask = (np.random.random((B, N)) > epsilon).astype(int)

            return action_greedy * mask + action_random * (1 - mask)

    def update(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        state_batch, obs_batch, action_batch, reward_batch, done_batch = self.replay_buffer.get_batches()

        T = state_batch.shape[1]  # episode_length

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)

        target_state_batch = state_batch[:, 1:].reshape(self.batch_size*(T-1),1,self.state_height_dim,self.state_width_dim)
        state_batch = state_batch.reshape(self.batch_size*T,1,self.state_height_dim,self.state_width_dim)

        # compute the Q_tot
        out_episode = []
        for t in range(T):
            obs = obs_batch[:, t].reshape(self.batch_size, self.num_agent, -1)
            obs = OneHot(obs)
            obs = torch.Tensor([obs]).to(self.device)
            obs = obs.view(self.batch_size * self.num_agent, -1)
            self.hidden_layer, out = self.agents(obs, self.hidden_layer)
            out = out.view(self.batch_size, self.num_agent, -1)
            out_episode.append(out)
        out_episode = torch.stack(out_episode, dim=1) # (B,T,N,out_feature)
        agent_q = torch.gather(out_episode,3,action_batch).squeeze(3) # (B,T,N)
        state_feature = self.state_encoder(state_batch).reshape(self.batch_size,T,-1)

        q_tot = self.mixing(agent_q,state_feature) # B,T
        # compute the td Q_target
        target_out_episode = []
        for t in range(T):
            obs = obs_batch[:, t].reshape(self.batch_size, self.num_agent, -1)
            obs = OneHot(obs)
            obs = torch.Tensor([obs]).to(self.device)
            obs = obs.view(self.batch_size * self.num_agent, -1)
            self.target_hidden_layer, out = self.target_agents(obs, self.hidden_layer)
            out = out.view(self.batch_size, self.num_agent, -1)
            target_out_episode.append(out)
        out_episode = torch.stack(out_episode[:,1:], dim=1) # (B,T-1,N,out_feature)
        max_next_q = torch.max(out_episode,dim = 3) # (B,T-1,N)
        target_state_feature = self.target_state_encoder(target_state_batch).reshape(self.batch_size,T-1,-1)

        q_tot_target = self.mixing(max_next_q,target_state_feature) # B,(T-1)

        td_error = reward_batch[:-1] + self.gamma * q_tot_target

        gap = td_error.detach() - q_tot[:-1]

        loss = (gap**2 + (reward_batch[-1] - q_tot[-1])**2).sum()/(self.batch_size*T)

        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.param, 10)
        self.optim.step()

        self.loss = loss

    def update_target(self,episode):
        if episode % self.target_update_episode == 0 and self.training:
            self.target_agents.update(self.agents)
            self.target_state_encoder.update(self.state_encoder)
            self.mixing.update(self.target_mixing)


    def save_model(self, filename):
        print("[INFO] save model to %s"%filename)
        torch.save(self.agents.state_dict(), filename)

    def load_model(self, run_dir, episode, continue_training = True):

        if continue_training:
            filename = os.path.join(run_dir,"qmix_ckpt_%d.pth"%episode)
            print("[INFO] load check point %s",filename)
            ckpt = torch.load(filename,map_location=self.device)
            self.agents.load_state_dict(ckpt['agents'])
            self.state_encoder.load_state_dict(ckpt["state_encoder"]),
            self.mixing.load_state_dict(ckpt["mixing"])
            self.target_agents.load_state_dict(ckpt["t_agents"])
            self.target_mixing.load_state_dict(ckpt["t_mixing"])
            self.target_state_encoder.load_state_dict(ckpt["t_state_encoder"])
            self.replay_buffer = ckpt['rb']
        else:
            filename = os.path.join(run_dir,"qmix_agent_%d.pth"%episode)
            print("[INFO] load agent %s",filename)
            self.agents.load_state_dict(torch.load(filename))
            self.agents.eval()

    def save_checkpoint(self,filename):
        # TODO: seed not change
        print("[INFO] save checkpoint to %s"%filename)
        checkpoint = {"agents":self.agents.state_dict(),
                      "state_encoder":self.state_encoder.state_dict(),
                      "mixing":self.mixing.state_dict(),
                      "t_agents": self.target_agents.state_dict(),
                      "t_state_encoder": self.target_state_encoder.state_dict(),
                      "t_mixing": self.target_mixing.state_dict(),
                      "rb":self.replay_buffer
                      }
        torch.save(checkpoint,filename)
