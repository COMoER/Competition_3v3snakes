import numpy as np
import torch
import os
device = torch.device("cuda:0")
EPISODE = int(os.environ.get('EPISODE'))

run = int(os.environ.get("RUN"))

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