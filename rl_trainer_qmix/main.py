import argparse
import datetime

from tensorboardX import SummaryWriter

from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *
from log_path import *
from env.chooseenv import make
from algo.qmix import QMIX

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main(args):
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    env = make(args.game_name, conf=None)

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')
    ctrl_agent_index = [0, 1, 2]
    enemy_agent_index = [3, 4, 5]
    print(f'Agent control by the actor: {ctrl_agent_index}')
    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    print(f'Game board width: {width}')
    height = env.board_height
    print(f'Game board height: {height}')

    act_dim = env.get_action_dim()
    print(f'action dimension: {act_dim}')
    obs_dim = 26
    print(f'observation dimension: {obs_dim}')

    print(f'replay buffer size: {args.buffer_size}')

    torch.manual_seed(args.seed)

    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    writer = SummaryWriter(str(log_dir))
    save_config(args, log_dir)

    model = QMIX(obs_dim, act_dim,width,height,32, ctrl_agent_num, args)

    if args.load_model:
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
        model.load_model(load_dir, episode=args.load_model_run_episode)

    episode = 0

    while episode < args.max_episodes:

        # Receive initial observation state s1
        state = env.reset()

        model.reset(1) # hidden layer for agents

        # During training, since all agents are given the same obs, we take the state of 1st agent.
        # However, when evaluation in Jidi, each agent get its own state, like state[agent_index]: dict()
        # more details refer to https://github.com/jidiai/Competition_3v3snakes/blob/master/run_log.py#L68
        # state: list() ; state[0]: dict()
        state_to_training = state[0]

        # ======================= feature engineering =======================
        # since all snakes play independently, we choose first three snakes for training.
        # Then, the trained model can apply to other agents. ctrl_agent_index -> [0, 1, 2]
        # Noted, the index is different in obs. please refer to env description.
        obs, state_map = get_observations(state_to_training, ctrl_agent_index, obs_dim, height, width)

        episode += 1
        step = 0

        replay_buffer = []

        episode_reward = np.zeros(6,dtype = int)

        episode_tot_reward = 0
        episode_tot_split_reward = np.zeros(3)

        action_available = None

        # Just run and collect the experience during one episode
        # The environment will be done during every 200 step
        while True:
            # ================================== inference ========================================
            actions_ctrl = model.choose_action(obs,action_available)
            actions_ctrl = actions_ctrl.reshape(-1)
            # ============================== add opponent actions =================================
            # use greedy policy for enemy TODO: both side are QMIX to train
            actions = action_greedy(state_to_training, actions_ctrl, height, width)
            # actions = action_random(act_dim,actions_ctrl)
            # get the limited action in the next state
            action_available = get_action_available(actions,ctrl_agent_index,act_dim)
            # the reward of Env is just the gain of length to each agents
            next_state, reward, done, _, info = env.step(env.encode(actions))
            next_state_to_training = next_state[0]
            next_obs,next_state_map = get_observations(next_state_to_training, ctrl_agent_index, obs_dim, height, width)

            # =========== ======================= reward shaping ========================================
            reward = np.array(reward)
            episode_reward += reward

            step_reward,split_step_reward = get_reward(info, episode_reward, ctrl_agent_index,enemy_agent_index, reward, done)

            episode_tot_reward += step_reward
            # done = np.array([done] * ctrl_agent_num)
            episode_tot_split_reward += split_step_reward

            # ================================== collect data ========================================
            # Store transition in R
            replay_buffer.append([state_map,obs,actions_ctrl,step_reward,action_available,done])

            obs,state_map = next_obs,next_state_map

            step += 1

            if args.episode_length <= step or done:

                print(f'[Episode {episode:05d}] total_reward: {np.sum(episode_reward[0:3]):d}')
                print(f'\t\t\t\tsnake_1: {episode_reward[0]} '
                      f'snake_2: {episode_reward[1]} snake_3: {episode_reward[2]}')
                print(f'\t\t\t\tepisode_win:{np.sum(episode_reward[ctrl_agent_index])>np.sum(episode_reward[enemy_agent_index])}')
                print(f'\t\t\t\tself_length:{np.sum(episode_reward[ctrl_agent_index]):d} enemy_length:{np.sum(episode_reward[enemy_agent_index])}')

                reward_tag = 'reward'

                writer.add_scalars(reward_tag, global_step=episode,
                                   tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                    'snake_3': episode_reward[2], 'total': np.sum(episode_reward[0:3])})

                score_tag = 'score'
                writer.add_scalars(score_tag, global_step=episode,
                                   tag_scalar_dict={'mean_step_reward':episode_tot_reward/step,
                                                    "mean_sparse_reward":episode_tot_split_reward[0]/step,
                                                    'mean_gain_reward':episode_tot_split_reward[1]/step,
                                                    'mean_dist_reward':episode_tot_split_reward[2]/step})

                win_tag = 'win_rate'
                writer.add_scalars(win_tag, global_step=episode,
                                   tag_scalar_dict={'win_rate':int(np.sum(episode_reward[ctrl_agent_index])>np.sum(episode_reward[enemy_agent_index]))})

                env.reset()
                break
        model.reset(args.batch_size)
        model.replay_buffer.push(replay_buffer)
        replay_buffer.clear()
        model.epsilon_delay(step)
        model.update()
        model.update_target(episode)

        loss_tag = 'loss'

        if model.loss:
            writer.add_scalars(loss_tag, global_step=episode,
                               tag_scalar_dict={'loss': model.loss})

        if model.loss:
            print(f'\t\t\t\tloss {model.loss:.3f}')

        if episode % args.save_interval == 0:
            model.save_model(os.path.join(run_dir,"qmix_agent_%d.pth"%episode))
            # model.save_checkpoint(os.path.join(run_dir,"qmix_ckpt_%d.pth"%episode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="qmix", type=str, help="")
    parser.add_argument('--max_episodes', default=100000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)

    parser.add_argument('--buffer_size', default=int(5e3), type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--update_target_episode', default=200, type=int)
    parser.add_argument('--DoubleDQN', action='store_true')
    parser.add_argument('--judgeIndependent', action='store_true')

    parser.add_argument("--save_interval", default=200, type=int)
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--load_model_run", default=2, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)

    args = parser.parse_args()
    main(args)
