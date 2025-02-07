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
        self.state_encode_mode = args.mode

        self._epsilon = EPSILON_MAX
        self.strategy = args.strategy

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

        self.hidden_layer = None
        self.target_hidden_layer = None
        # Initialise replay buffer R
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size,args.strategy)

        self.loss = None
        self.delay_times = args.delay_times

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

    def reset(self, batch_size):
        """
        reset the hidden layer of each rnn when every episode begin
        """
        self.hidden_layer = self.agents.init_hidden().unsqueeze(0).expand(batch_size, self.num_agent, -1)
        self.target_hidden_layer = self.target_agents.init_hidden().unsqueeze(0).expand(batch_size, self.num_agent, -1)

    def epsilon_delay(self, delay_times):
        # update epsilon as paper
        rate = delay_times / self.delay_times
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

        state_batch, obs_batch, action_batch, reward_batch, action_available_batch,done_batch = self.replay_buffer.get_batches()

        T = state_batch.shape[1]  # episode_length

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device).view(self.batch_size,T,self.num_agent,1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        action_available_batch = torch.LongTensor(action_available_batch).to(self.device) # B,T,N,act_dim

        if self.state_encode_mode == 1:
            target_state_batch = state_batch[:, 1:].reshape(self.batch_size * (T - 1), -1, self.state_height_dim,
                                                            self.state_width_dim)
            state_batch = state_batch.reshape(self.batch_size * T, -1, self.state_height_dim, self.state_width_dim)
        elif not self.state_encode_mode == 4:
            target_state_batch = state_batch[:, 1:].reshape(self.batch_size*(T-1),1,self.state_height_dim,self.state_width_dim)
            state_batch = state_batch.reshape(self.batch_size*T,1,self.state_height_dim,self.state_width_dim)
        else:
            state_batch = state_batch.reshape(self.batch_size, T, self.state_feature_dim)
            target_state_batch = state_batch[:, 1:].reshape(self.batch_size ,(T - 1), self.state_feature_dim)

        # compute the Q_tot
        out_episode = []
        for t in range(T):
            obs = obs_batch[:, t].reshape(self.batch_size, self.num_agent, -1)
            obs = OneHot(obs)
            obs = torch.FloatTensor([obs]).to(self.device)
            obs = obs.view(self.batch_size * self.num_agent, -1)
            self.hidden_layer, out = self.agents(obs, self.hidden_layer)
            out = out.view(self.batch_size, self.num_agent, -1)
            out_episode.append(out)
        out_episode = torch.stack(out_episode, dim=1) # (B,T,N,out_feature)
        agent_q = torch.gather(out_episode,3,action_batch).squeeze(3) # (B,T,N)
        if not self.state_encode_mode == 4:
            state_feature = self.state_encoder(state_batch).reshape(self.batch_size,T,-1)
            q_tot = self.mixing(agent_q,state_feature) # B,T
        else:
            q_tot = self.mixing(agent_q,state_batch)
        # compute the td Q_target
        target_out_episode = []
        for t in range(T):
            obs = obs_batch[:, t].reshape(self.batch_size, self.num_agent, -1)
            obs = OneHot(obs)
            obs = torch.FloatTensor([obs]).to(self.device)
            obs = obs.view(self.batch_size * self.num_agent, -1)
            self.target_hidden_layer, out = self.target_agents(obs, self.target_hidden_layer)
            out = out.view(self.batch_size, self.num_agent, -1)
            target_out_episode.append(out)

        target_out_episode = torch.stack(target_out_episode[1:], dim=1) # (B,T-1,N,out_feature)
        if self.isDoubleDQN:
            # Double DQN
            next_state_q_tot = out_episode[:, 1:].clone().detach()  # B,T-1,N,act_dim
            next_state_q_tot[action_available_batch[:,:-1]==0] = -99999 # not available
            max_next_action = torch.argmax(next_state_q_tot,dim=3,keepdim=True) # (B,T-1,N,1)
            max_next_q = torch.gather(target_out_episode, dim=3, index=max_next_action).squeeze(3)  # B,T-1,N
        else:
            target_out_episode[action_available_batch[:, :-1] == 0] = -99999  # not available
            max_next_q = torch.max(target_out_episode,dim=3)[0]  # B,T-1,N

        if not self.state_encode_mode == 4:
            target_state_feature = self.target_state_encoder(target_state_batch).reshape(self.batch_size, T - 1, -1)
            q_tot_target = self.target_mixing(max_next_q,target_state_feature) # B,(T-1)
        else:
            q_tot_target = self.target_mixing(max_next_q,target_state_batch) # B,(T-1)

        td_error = reward_batch[:,:-1] + self.gamma * q_tot_target

        gap = td_error.detach() - q_tot[:,:-1] # B,T-1

        loss = ((gap**2).sum() + ((reward_batch[:,-1] - q_tot[:,-1])**2).sum())/(self.batch_size*T)

        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.param, 10)
        self.optim.step()

        self.loss = loss

    def update_target(self,episode):
        if episode % self.target_update_episode == 0 and self.training:
            print("[INFO] update target..")
            self.target_agents.update(self.agents)
            if not self.state_encode_mode == 4:
                self.target_state_encoder.update(self.state_encoder)
            self.target_mixing.update(self.mixing)


    def save_model(self, filename):
        if not self.state_encode_mode == 4:
            checkpoint = {"agents": self.agents.state_dict(),
                          "state_encoder": self.state_encoder.state_dict(),
                          "mixing": self.mixing.state_dict(),
                          }
        else:
            checkpoint = {"agents": self.agents.state_dict(),
                          "mixing": self.mixing.state_dict(),
                          }
        print("[INFO] save model to %s"%filename)
        torch.save(checkpoint, filename)

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
