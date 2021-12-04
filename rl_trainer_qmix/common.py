import numpy as np
import torch
import torch.nn as nn
import math
import copy
from typing import Union
from torch.distributions import Categorical
import os
import yaml
import random

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_grid_map(board_width, board_height, beans_positions: list, snakes_positions: dict):
    """
    snake: 2,3,4,5,6,7
    bean: 1
    """
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


def get_min_bean(x, y, beans_position):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_x, min_y, index


def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position)
        beans_position.pop(index)

        next_distances = []
        up_distance = math.inf if head_surrounding[0] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
        next_distances.append(up_distance)
        down_distance = math.inf if head_surrounding[1] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
        next_distances.append(down_distance)
        left_distance = math.inf if head_surrounding[2] > 1 else \
            math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(left_distance)
        right_distance = math.inf if head_surrounding[3] > 1 else \
            math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(right_distance)
        actions.append(next_distances.index(min(next_distances)))
    return actions



def get_observations(state, agents_index, obs_dim, height, width,mode):
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

    if mode == 1: # one hot
        state_index = state.reshape(-1)-1 # (1-7) -> (0,6)
        L = np.max(state_index) + 1
        eye = np.eye(L)
        # One Hot
        state_input = eye[state_index].reshape((state.shape[0],state.shape[1],L))
    elif mode < 4: # normal or fc
        state_input = state.copy() / 7  # normalized
    elif mode == 4:
        pass
    else:
        assert False,"No such mode"

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

    if mode == 4:
        # just the info of each snake
        state_input = np.zeros(10+12)

        # beans positions 5*2
        state_input[:10] = beans_position.flatten()

        # snake positions 2*6 TODO: just snake head?
        snake_heads = np.array([snake[0] for snake in snakes_positions_list])
        snake_heads = snake_heads.astype(np.float)
        snake_heads[:,0] /= board_height
        snake_heads[:,1] /= board_width
        state_input[10:] = snake_heads.flatten()

    return observations,state_input


def get_reward(info, history_reward, ctrl_snake_index, enemy_snake_index, reward, done,args):
    """
    reward function
    global reward for origin qmix

    Args:
        info: the global state information to determine the reward
        history_reward: the historical reward given by the environment which is just the gain length of each agents
        ctrl_snake_index: the index of the controlled snakes
        enemy_snake_index: the index of the enemy snakes
        reward: step reward by environment
        done: whether the episode has been done or not
    Return:
        global_reward: the global reward for Q_tot to learn
    """
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=int).reshape((1,-1,2))
    snake_heads = np.array([snake[0] for snake in snakes_position],dtype = int)
    self_heads:np.ndarray = snake_heads[ctrl_snake_index]

    step_reward = 0
    reward_split = [0,0,0]
    self_length = np.sum(history_reward[ctrl_snake_index])
    enemy_length = np.sum(history_reward[enemy_snake_index])
    # sparse reward
    if self_length > enemy_length:
        # take advantage
        if done:
            # final win
            step_reward += args.win_gain # 30
        else:
            # step win
            step_reward += args.win_gain//2 # 15
    if self_length < enemy_length:
        if done:
            # final lose
            step_reward -= args.lose_gain # 20
        else:
            # step lose
            step_reward -= args.lose_gain//2 # 10
    self_reward = reward[ctrl_snake_index]
    reward_split[0] = step_reward
    # gain reward
    step_reward += np.sum(self_reward)*args.step_radio # the raw env target reward should be signed high weight 20

    reward_split[1] = step_reward - reward_split[0]

    # bean dist reward
    dist = np.sum(np.abs(beans_position - self_heads.reshape((-1,1,2))),axis = 2) # (N,B,2)
    min_bean_dist = np.min(dist,axis = 1) # (N,)
    # step_reward -= np.sum(min_bean_dist[self_reward<=0])*4
    step_reward -= np.sum(min_bean_dist[self_reward <= 0])
    reward_split[2] = step_reward - reward_split[1] - reward_split[0]
    return step_reward,np.array(reward_split)

def action_random(act_dim, actions_ctrl):
    """
    when training, enemy is random policy
    """
    num_agents = len(actions_ctrl)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = actions_ctrl
    return actions

def action_well_trained(state, actions_ctrl, height, width,well_trained_agent):
    """
    when training, enemy is a above well-trained agent TODO
    """
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list

    # greedy_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])
    well_action = np.zeros(3)

    action_list = np.zeros(6)
    action_list[:3] = actions_ctrl
    action_list[3:] = well_action

    return action_list

def action_greedy(state, actions_ctrl, height, width):
    """
    when training, enemy is greedy policy
    """
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list

    greedy_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])

    action_list = np.zeros(6)
    action_list[:3] = actions_ctrl
    action_list[3:] = greedy_action

    return action_list


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding

def get_action_available(action,ctrl_agent_index,act_dim):
    """
    Args:
        action: last action
        ctrl_agent_index: controlled agent index
    """

    ctrl_action = action[ctrl_agent_index]
    action_limited = (ctrl_action//2)*2 + (ctrl_action+1)%2
    action_available = []
    for a in action_limited:
        available = [1]*act_dim
        available[int(a)] = 0
        action_available.append(available)
    return np.array(action_available)



def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()
