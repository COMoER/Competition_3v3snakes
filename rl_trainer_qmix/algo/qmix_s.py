import os
import torch
import numpy as np
# from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer,ReplayBuffer_s
from common import device
from algo.network_s import MixingNet, Agent, state_encoder

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


class QMIX_s:

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
        self.state_encode_mode = args.mode
        self.tau = args.tau

        self.replay_buffer = ReplayBuffer_s(args.buffer_size, args.batch_size)

        self._epsilon = EPSILON_MAX

        # Initialize the agents
        self.agents = Agent(self.obs_dim, act_dim).to(device)  # all agents share the same AgentNet
        if not args.mode == 4:
            self.state_encoder = state_encoder(state_width_dim, state_height_dim, state_feature_dim,args.mode).to(device)
        else:
            state_feature_dim = 22 # fixed state feature is 22 dimension
            self.state_feature_dim = state_feature_dim
        self.mixing = MixingNet(num_agent, state_feature_dim, MIXING_HIDDEN_SIZE).to(device)

        self.target_agents = Agent(self.obs_dim, act_dim).to(device)  # target network
        if not args.mode == 4:
            self.target_state_encoder = state_encoder(state_width_dim, state_height_dim, state_feature_dim,args.mode).to(device)
        self.target_mixing = MixingNet(num_agent, state_feature_dim, MIXING_HIDDEN_SIZE).to(device)

        self.loss = None

        # at the beginning, update the target
        self.target_agents.update(self.agents)
        if args.mode < 4:
            self.target_state_encoder.update(self.state_encoder)
        self.target_mixing.update(self.mixing)

        self.target_update_episode = args.update_target_episode

        self.isDoubleDQN = args.DoubleDQN
        self.judgeIndependent = args.judgeIndependent

        # initialize the optimizer
        self.param = list(self.agents.parameters())
        if not args.mode == 4:
            self.param += self.state_encoder.parameters()
        self.param += self.mixing.parameters()
        self.optim = torch.optim.RMSprop(self.param, args.lr, alpha=0.99, eps=0.00001)
    def eval(self):
        self.training = False
        self.agents.eval()

    def epsilon_delay(self):
        # update epsilon as paper
        rate = 1 / DELAY_TIMES
        if self._epsilon > EPSILON_MIN:
            self._epsilon += rate * (EPSILON_MIN - EPSILON_MAX)

    def choose_action(self, obs, action_available,evaluate=False):
        """
        Args:
            obs:(N,obs_feature) not using one_hot to encode distinct agents
            action_available: action couldn't take (N,act_dim) or None
        Returns:
            logits: (N,) the chosen action using epsilon-greedy
        """
        N, _ = obs.shape
        obs = obs.reshape(1,N,-1)
        obs = OneHot(obs)

        epsilon = self._epsilon
        obs = torch.FloatTensor([obs]).to(self.device).view(N,-1)
        self.hidden_layer, out = self.agents(obs, self.hidden_layer)
        q_value = out.cpu().detach().numpy() #(N,act_dim)
        if not action_available is None:
            q_value[action_available==0] = -99999 # -inf q_value to avoid choose this action
        action_greedy = np.argmax(q_value, axis=1)  # B,N
        if evaluate:
            return action_greedy
        else:
            if not action_available is None:
                action_random_list = []
                for action in action_available:
                    action = list(action).index(0)
                    available_action = [0,1,2,3]
                    available_action.pop(action)
                    action_random_list.append(available_action[np.random.random_integers(0,self.act_dim-2)])
                action_random = np.array(action_random_list,dtype = int)
            else:

                action_random = np.random.randint(self.act_dim, size=self.num_agent, dtype=int)

            if self.judgeIndependent:
                mask = (np.random.random((N,)) > epsilon).astype(int)
                return action_greedy * mask + action_random * (1 - mask)
            else:
                if np.random.random() > epsilon:
                    return action_greedy
                else:
                    return action_random

    def update(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        state_batch, obs_batch, action_batch, reward_batch, next_state_batch,next_obs_batch,action_available_batch,done_batch = self.replay_buffer.get_batches()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device).view(self.batch_size,self.num_agent,1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        action_available_batch = torch.LongTensor(action_available_batch).to(self.device) # B,T,N,act_dim
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # compute the Q_tot
        obs = obs_batch.reshape(self.batch_size, self.num_agent, -1)
        obs = OneHot(obs)
        obs = torch.FloatTensor([obs]).to(self.device).view(self.batch_size,self.num_agent, -1)
        out = self.agents(obs)

        agent_q = torch.gather(out,2,action_batch).squeeze(2) # (B,T)
        if not self.state_encode_mode == 4:
            state_feature = self.state_encoder(state_batch).reshape(self.batch_size,-1)
            q_tot = self.mixing(agent_q,state_feature) # B,T
        else:
            q_tot = self.mixing(agent_q,state_batch)
        # compute the td Q_target
        target_out_episode = []
        t_obs = next_obs_batch.reshape(self.batch_size, self.num_agent, -1)
        t_obs = OneHot(t_obs)
        t_obs = torch.FloatTensor([t_obs]).to(self.device)
        t_out = self.target_agents(t_obs)
        t_out = t_out.view(self.batch_size, self.num_agent, -1)
        if self.isDoubleDQN:
            # Double DQN
            t_out[action_available_batch==0] = -99999 # not available
            max_next_action = torch.argmax(t_out,dim=2,keepdim=True) # (B,N,1)
            max_next_q = torch.gather(target_out_episode, dim=2, index=max_next_action).squeeze(2)  # B,N
        else:
            t_out[action_available_batch == 0] = -99999  # not available
            max_next_q = torch.max(target_out_episode,dim=2)[0]  # B,N

        if not self.state_encode_mode == 4:
            target_state_feature = self.target_state_encoder(next_state_batch).reshape(self.batch_size, -1)
            q_tot_target = self.target_mixing(max_next_q,target_state_feature) # B
        else:
            q_tot_target = self.target_mixing(max_next_q,next_state_batch) # B

        td_error = reward_batch + self.gamma * q_tot_target

        loss =  F.mse_loss(q_tot,td_error.detach()*done_batch)

        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.param, 10)
        self.optim.step()

        self.loss = loss.detach().cpu().item()

    def update_target(self):
        if self.training:
            self.target_agents.soft_update(self.agents,self.tau)
            if not self.state_encode_mode == 4:
                self.target_state_encoder.soft_update(self.state_encoder,self.tau)
            self.target_mixing.soft_update(self.mixing,self.tau)


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
            filename = run_dir/Path("qmix_agent_%d.pth"%episode)
            print("[INFO] load agent %s"%filename)
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
