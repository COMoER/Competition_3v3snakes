import os
from pathlib import Path
import sys
import numpy as np
import math
import copy

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

    greedy_action = greedy_snake(state, beans, snakes, width, height, actions_ctrl)
    action_list = np.zeros(3)
    action_list[:3] = greedy_action

    return action_list
def to_joint_action(action,index):
    action = action[index]
    joint_action = [0]*4
    joint_action[action] = 1
    return [joint_action]
def my_controller(observation_list, action_space_list, is_act_continuous):
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
    actions = action_greedy(observation_list,indexs,board_height,board_width)
    actions = to_joint_action(actions,(o_index-2)%3)
    return actions
