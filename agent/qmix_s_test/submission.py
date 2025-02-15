import os
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

device = torch.device("cuda:0")
EPISODE = int(os.environ.get('EPISODE'))

run = int(os.environ.get("RUN"))

def config():
    dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'rl_trainer_qmix','models','snakes_3v3','run%d'%run)
    print("run%d:"%run)
    f = open(os.path.join(dir,"config.yaml"),'r')
    con = yaml.load(f,Loader=yaml.SafeLoader)
    f.close()
    for query,value in con.items():
        print("%s:  %s"%(query,value))
    print()

config()

def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def make_grid_map(board_width, board_height, beans_positions: list, snakes_positions: dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


def get_observations(state, agents_index, obs_dim, height, width):
    """
    - all position should be normalized by width and height
    - agent observation
    - Self position:        0:head_x; 1:head_y
    - Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right (the number of the object inside that pos)
    - Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
    - Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)

    - global state
    - just the grid map

    Args:
        state: global state dict
        agents_index: the ctrl_agents index
        obs_dim: observation dimension for each agents
        height: map height
        width: map width
    Return:
        observations(the obs vector to each agents) state_input(global state map to treat as the mixing net hypernetwork) input
    """
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(np.array(value,dtype = int))
    # create grid_map to store state
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)

    state = np.squeeze(state_, axis=2)

    observations = np.zeros((3, obs_dim))

    beans_position = np.array(beans_positions, dtype=float)
    beans_position[:, 0] /= board_height # y
    beans_position[:, 1] /= board_width # x
    for i,element in enumerate(agents_index):
        # self head position
        head_x = snakes_positions_list[element][0,1]
        head_y = snakes_positions_list[element][0,0]

        observations[i][:2] = head_x/board_width,head_y/board_height
        # head surroundings
        # value [0,1,2,3,4,5,6,7] normalized by 7 which is the max
        head_surrounding = np.array(get_surrounding(state, width, height, head_x, head_y))/7
        observations[i][2:6] = head_surrounding

        # beans positions
        observations[i][6:16] = beans_position.flatten()

        # other snake positions
        snake_heads = np.array([snake[0] for snake in snakes_positions_list])
        snake_heads = np.delete(snake_heads, element, 0)
        snake_heads = snake_heads.astype(np.float)
        snake_heads[:,0] /= board_height
        snake_heads[:,1] /= board_width
        observations[i][16:] = snake_heads.flatten()

    return observations

HIDDEN_SIZE = 64


def OneHot(obs: np.ndarray):
    """
    Args:
        obs:(N,obs_dim)
    Returns:
        obs:(N,obs_dim+N)
    """
    N, _ = obs.shape
    encode = np.eye(N) # one hot encoding
    return np.concatenate([obs, encode], axis=1)

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

class QMIX:
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

s_agent = QMIX(26,4,3)
actor_net = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'rl_trainer_qmix','models','snakes_3v3','run%d'%run)
s_agent.load_model(actor_net,EPISODE)

def to_joint_action(action,index):
    action = action[index]
    joint_action = [0]*4
    joint_action[action] = 1
    return [joint_action]
def my_controller(observation_list, action_space_list, is_act_continuous):
    obs_dim = 26
    obs = observation_list.copy()
    board_width = obs['board_width']
    board_height = obs['board_height']
    o_index = obs['controlled_snake_index']  # 2, 3, 4, 5, 6, 7 -> indexs = [0,1,2,3,4,5]
    action_available = np.array([1, 1, 1, 1])
    if not obs['last_direction'] is None:
        last_direction = ['up','down','left','right'].index(obs['last_direction'][o_index-2])
        action_available[last_direction//2*2+(last_direction+1)%2] = 0
    o_indexs_min = 3 if o_index > 4 else 0
    indexs = [o_indexs_min, o_indexs_min + 1, o_indexs_min + 2]
    observation = get_observations(obs, indexs, obs_dim, height=board_height, width=board_width)
    actions = s_agent.choose_action(observation,action_available,(o_index-2)%3)
    actions = to_joint_action(actions,(o_index-2)%3)
    return actions
