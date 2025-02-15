import numpy as np
import torch
import random


from agent.greedy.submission import action_greedy
from agent.agent_eva.eva import q_agent,QMIX_TYPE
from agent.agent_eva.common import get_observations
from agent.rl_test.submission import rl_agent,rl_get_observations
from env.chooseenv import make
from tabulate import tabulate
import argparse
from torch.distributions import Categorical
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

PATH = os.environ.get('LOG')
RUN = int(os.environ.get('RUN'))
EPISODE = int(os.environ.get('EPISODE'))

def get_actions(state, algo, indexs):

    # random agent
    actions = np.random.randint(4, size=3)

    # rl agent
    if algo == 'rl':
        obs = rl_get_observations(state, indexs, obs_dim=26, height=10, width=20)
        logits = rl_agent.choose_action(obs)
        logits = torch.Tensor(logits)
        actions = np.array([Categorical(out).sample().item() for out in logits])

    # QMIX
    if algo == 'qmix':
        observation = get_observations(state, indexs, 26, height=10, width=20)
        actions = q_agent.choose_action_global(observation)

    if algo=="greedy":
        actions = action_greedy(state,indexs,10,20)

    return actions


def get_join_actions(obs, algo_list):
    obs_2_evaluation = obs[0]
    indexs = [0,1,2,3,4,5]
    first_action = get_actions(obs_2_evaluation, algo_list[0], indexs[:3])
    second_action = get_actions(obs_2_evaluation, algo_list[1], indexs[3:])
    actions = np.zeros(6)
    actions[:3] = first_action[:]
    actions[3:] = second_action[:]
    return actions

def reset():
    if QMIX_TYPE == 1:
        q_agent.reset()

def run_game(env, algo_list, episode, verbose=False):

    total_reward = np.zeros(6)
    num_win = np.zeros(3)

    for i in range(1, episode + 1):
        episode_reward = np.zeros(6)

        state = env.reset()

        reset()

        step = 0

        while True:
            joint_action = get_join_actions(state, algo_list)

            next_state, reward, done, _, info = env.step(env.encode(joint_action))
            reward = np.array(reward)
            episode_reward += reward

            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    num_win[0] += 1
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    num_win[1] += 1
                else:
                    num_win[2] += 1

                if not verbose:
                    print('.', end='')
                    if i % 100 == 0 or i == episode:
                        print()
                break

            state = next_state
            step += 1

        total_reward += episode_reward

    # calculate results
    total_reward /= episode
    print("total_reward: ", total_reward)
    print(f'\nResult base on {episode} ', end='')
    print('episode:') if episode == 1 else print('episodes:')

    header = ['Name', algo_list[0], algo_list[1]]
    data = [['score', np.round(np.sum(total_reward[:3]), 2), np.round(np.sum(total_reward[3:]), 2)],
            ['win', num_win[0], num_win[1]]]
    print(tabulate(data, headers=header, tablefmt='pretty'))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"test_output",PATH),'a') as f:
        f.write(f"run {RUN:d} episode {EPISODE:d}\nscore {np.round(np.sum(total_reward[:3]), 2):.3f} {np.round(np.sum(total_reward[3:]), 2):.3f}\nwin {int(num_win[0]):d} {int(num_win[1]):d}\n\n")



if __name__ == "__main__":
    env_type = 'snakes_3v3'

    game = make(env_type, conf=None)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="rl", help="rl/random")
    parser.add_argument("--opponent", default="random", help="rl/random")
    parser.add_argument("--episode", default=100)
    args = parser.parse_args()

    agent_list = [args.my_ai, args.opponent]
    run_game(game, algo_list=agent_list, episode=args.episode, verbose=False)
