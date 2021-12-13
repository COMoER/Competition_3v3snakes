from tensorboardX import SummaryWriter
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *
from log_path import *
from algo.qmix import QMIX
from algo.qmix_s import QMIX_s
from env.chooseenv import make

def step_trainer(args):
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

    setup_seed(args.seed)

    assert (args.compete + args.well_enemy + args.self_compete) < 2, "can't be both true"

    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    writer = SummaryWriter(str(log_dir))
    save_config(args, log_dir)

    model = QMIX_s(obs_dim, act_dim, width, height, 32, ctrl_agent_num, args)

    replay_buffer = []
    if args.compete:
        model_enemy = QMIX_s(obs_dim, act_dim, width, height, 32, ctrl_agent_num, args)
    if args.compete or args.self_compete:
        enemy_replay_buffer = []

    if args.well_enemy:
        model_enemy = QMIX_s(obs_dim, act_dim, width, height, 32, ctrl_agent_num, args)
        model_enemy.load_model(Path(__file__).resolve().parent / Path("well"), 100000, False)
        model_enemy.eval()

    if args.load_model:
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
        model.load_model(load_dir, episode=args.load_model_run_episode)

    episode = 0

    while episode < args.max_episodes:

        # Receive initial observation state s1
        state = env.reset()

        # During training, since all agents are given the same obs, we take the state of 1st agent.
        # However, when evaluation in Jidi, each agent get its own state, like state[agent_index]: dict()
        # more details refer to https://github.com/jidiai/Competition_3v3snakes/blob/master/run_log.py#L68
        # state: list() ; state[0]: dict()
        state_to_training = state[0]

        # ======================= feature engineering =======================
        # since all snakes play independently, we choose first three snakes for training.
        # Then, the trained model can apply to other agents. ctrl_agent_index -> [0, 1, 2]
        # Noted, the index is different in obs. please refer to env description.
        obs, state_map = get_observations(state_to_training, ctrl_agent_index, obs_dim, height, width, args.mode)

        if args.compete or args.well_enemy or args.self_compete:
            obs_e, state_map_e = get_observations(state_to_training, enemy_agent_index, obs_dim, height, width,
                                                  args.mode)

        episode += 1
        step = 0

        episode_reward = np.zeros(6, dtype=int)

        episode_tot_reward = 0
        episode_tot_split_reward = np.zeros(3)

        if args.compete or args.self_compete:
            episode_tot_reward_e = 0
        if args.compete or args.well_enemy or args.self_compete:
            action_available_e = None

        if args.compete:
            loss_e = []
        loss = []

        action_available = None

        # Just run and collect the experience during one episode
        # The environment will be done during every 200 step
        while True:
            # ================================== inference ========================================
            actions_ctrl = model.choose_action(obs, action_available)
            actions_ctrl = actions_ctrl.reshape(-1)

            # ============================== add opponent actions =================================
            # use greedy policy for enemy TODO: both side are QMIX to train
            if args.compete or args.well_enemy or args.self_compete:
                model_e = model_enemy if args.compete or args.well_enemy else model
                actions_e = model_e.choose_action(obs_e, action_available_e)
                actions_e = actions_e.reshape(-1)
                actions = np.concatenate([actions_ctrl, actions_e], axis=0)
            else:
                actions = action_greedy(state_to_training, actions_ctrl, height, width)
            # actions = action_random(act_dim,actions_ctrl)
            # get the limited action in the next state
            action_available = get_action_available(actions, ctrl_agent_index, act_dim)
            if args.compete or args.well_enemy or args.self_compete:
                actions_available_e = get_action_available(actions, enemy_agent_index, act_dim)

            # the reward of Env is just the gain of length to each agents
            next_state, reward, done, _, info = env.step(env.encode(actions))
            next_state_to_training = next_state[0]
            next_obs, next_state_map = get_observations(next_state_to_training, ctrl_agent_index, obs_dim, height,
                                                        width, args.mode)
            if args.compete or args.well_enemy or args.self_compete:
                next_obs_e, next_state_map_e = get_observations(next_state_to_training, enemy_agent_index, obs_dim,
                                                                height,
                                                                width, args.mode)
            # =========== ======================= reward shaping ========================================
            reward = np.array(reward)
            episode_reward += reward

            step_reward, split_step_reward = get_reward(info, episode_reward, ctrl_agent_index, enemy_agent_index,
                                                        reward, done, args)
            if args.compete or args.self_compete:
                step_reward_e, _ = get_reward(info, episode_reward, enemy_agent_index, ctrl_agent_index,
                                              reward, done, args)
                episode_tot_reward_e += step_reward_e

            episode_tot_reward += step_reward
            # done = np.array([done] * ctrl_agent_num)
            episode_tot_split_reward += split_step_reward

            # ================================== collect data ========================================
            # Store transition in R
            if args.self_compete:
                model.replay_buffer.push([state_map_e, obs_e, actions_e, step_reward_e,next_state_map_e,next_obs_e,
                                          actions_available_e, done])


            model.replay_buffer.push([state_map, obs, actions_ctrl, step_reward, next_state_map,next_obs,
                                      action_available, done])
            model.epsilon_delay()
            model.update()
            obs, state_map = next_obs, next_state_map

            state_to_training = next_state_to_training # TODO: a great BUG!!!!
            if args.compete:
                model_enemy.replay_buffer.append([state_map_e, obs_e, actions_e, step_reward_e, next_state_map_e,next_obs_e,
                                            actions_available_e, done])
                model_enemy.epsilon_delay()
                model_enemy.update()
                model_enemy.update_target()
            if args.well_enemy or args.compete or args.self_compete:
                obs_e, state_map_e = next_obs_e, next_state_map_e
            step += 1

            if model.loss:
                loss.append(model.loss)
            if args.compete and model_enemy.loss:
                loss_e.append(model_enemy.loss)

            if args.episode_length <= step or done:

                print(f'[Episode {episode:05d}] total_reward: {np.sum(episode_reward[0:3]):d}')
                print(f'\t\t\t\tsnake_1: {episode_reward[0]} '
                      f'snake_2: {episode_reward[1]} snake_3: {episode_reward[2]}')
                print(f'\t\t\t\tsnake_4: {episode_reward[3]} '
                      f'snake_5: {episode_reward[4]} snake_6: {episode_reward[5]}')
                print(
                    f'\t\t\t\tepisode_win:{np.sum(episode_reward[ctrl_agent_index]) > np.sum(episode_reward[enemy_agent_index])}')
                print(
                    f'\t\t\t\tself_length:{np.sum(episode_reward[ctrl_agent_index]):d} enemy_length:{np.sum(episode_reward[enemy_agent_index])}')

                reward_tag = 'reward'

                if args.compete or args.self_compete:
                    writer.add_scalars(reward_tag, global_step=episode,
                                       tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                        'snake_3': episode_reward[2],
                                                        'total': np.sum(episode_reward[0:3]),
                                                        'snake_4': episode_reward[3], 'snake_5': episode_reward[4],
                                                        'snake_6': episode_reward[5],
                                                        'enemy_total': np.sum(episode_reward[3:])})
                else:
                    writer.add_scalars(reward_tag, global_step=episode,
                                       tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                        'snake_3': episode_reward[2],
                                                        'total': np.sum(episode_reward[0:3])})

                score_tag = 'score'
                if args.compete or args.self_compete:
                    writer.add_scalars(score_tag, global_step=episode,
                                       tag_scalar_dict={'mean_step_reward': episode_tot_reward / step,
                                                        "mean_sparse_reward": episode_tot_split_reward[0] / step,
                                                        'mean_gain_reward': episode_tot_split_reward[1] / step,
                                                        'mean_dist_reward': episode_tot_split_reward[2] / step,
                                                        'enemy_mean_step_reward': episode_tot_reward_e / step})
                else:
                    writer.add_scalars(score_tag, global_step=episode,
                                       tag_scalar_dict={'mean_step_reward': episode_tot_reward / step,
                                                        "mean_sparse_reward": episode_tot_split_reward[0] / step,
                                                        'mean_gain_reward': episode_tot_split_reward[1] / step,
                                                        'mean_dist_reward': episode_tot_split_reward[2] / step})

                win_tag = 'win_rate'
                writer.add_scalars(win_tag, global_step=episode,
                                   tag_scalar_dict={'win_rate': int(np.sum(episode_reward[ctrl_agent_index]) > np.sum(
                                       episode_reward[enemy_agent_index]))})

                loss_tag = 'loss'

                if len(loss) and not args.compete:
                    writer.add_scalars(loss_tag, global_step=episode,
                                           tag_scalar_dict={'loss': np.mean(np.array(loss))})
                else:
                    if len(loss_e):
                        writer.add_scalars(loss_tag, global_step=episode,
                                       tag_scalar_dict={'loss': np.mean(np.array(loss)),
                                                        'enemy': np.mean(np.array(loss_e))})

                if len(loss):
                    if not args.compete:
                        print(f'\t\t\t\tloss {np.mean(np.array(loss)):.3f}')
                    elif len(loss_e):
                        print(f'\t\t\t\tloss {np.mean(np.array(loss)):.3f}')
                        print(f'\t\t\t\tloss {np.mean(np.array(loss_e)):.3f}')


                env.reset()
                break



        if episode % args.save_interval == 0:
            model.save_model(os.path.join(run_dir, "qmix_agent_%d.pth" % episode))
            if args.compete:
                model_enemy.save_model(os.path.join(run_dir, "qmix_enemy_agent_%d.pth" % episode))
            # model.save_checkpoint(os.path.join(run_dir,"qmix_ckpt_%d.pth"%episode))

def rnn_trainer(args):
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

    setup_seed(args.seed)

    assert (args.compete+args.well_enemy+args.self_compete) < 2, "can't be both true"

    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    writer = SummaryWriter(str(log_dir))
    save_config(args, log_dir)

    model = QMIX(obs_dim, act_dim, width, height, 32, ctrl_agent_num, args)

    replay_buffer = []
    if args.compete:
        model_enemy = QMIX(obs_dim, act_dim, width, height, 32, ctrl_agent_num, args)
    if args.compete or args.self_compete:
        enemy_replay_buffer = []

    if args.well_enemy:
        model_enemy = QMIX(obs_dim, act_dim, width, height, 32, ctrl_agent_num, args)
        model_enemy.load_model(Path(__file__).resolve().parent / Path("well"), 100000, False)
        model_enemy.eval()

    if args.load_model:
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
        model.load_model(load_dir, episode=args.load_model_run_episode)

    episode = 0

    while episode < args.max_episodes:

        # Receive initial observation state s1
        state = env.reset()

        model.reset(1)  # hidden layer for agents

        if args.compete or args.well_enemy:
            model_enemy.reset(1)

        # During training, since all agents are given the same obs, we take the state of 1st agent.
        # However, when evaluation in Jidi, each agent get its own state, like state[agent_index]: dict()
        # more details refer to https://github.com/jidiai/Competition_3v3snakes/blob/master/run_log.py#L68
        # state: list() ; state[0]: dict()
        state_to_training = state[0]

        # ======================= feature engineering =======================
        # since all snakes play independently, we choose first three snakes for training.
        # Then, the trained model can apply to other agents. ctrl_agent_index -> [0, 1, 2]
        # Noted, the index is different in obs. please refer to env description.
        obs, state_map = get_observations(state_to_training, ctrl_agent_index, obs_dim, height, width, args.mode)

        if args.compete or args.well_enemy or args.self_compete:
            obs_e, state_map_e = get_observations(state_to_training, enemy_agent_index, obs_dim, height, width,
                                                  args.mode)

        episode += 1
        step = 0

        episode_reward = np.zeros(6, dtype=int)

        episode_tot_reward = 0
        episode_tot_split_reward = np.zeros(3)

        if args.compete or args.self_compete:
            episode_tot_reward_e = 0
        if args.compete or args.well_enemy or args.self_compete:
            action_available_e = None

        action_available = None

        # Just run and collect the experience during one episode
        # The environment will be done during every 200 step
        while True:
            # ================================== inference ========================================
            actions_ctrl = model.choose_action(obs, action_available)
            actions_ctrl = actions_ctrl.reshape(-1)

            # ============================== add opponent actions =================================
            # use greedy policy for enemy TODO: both side are QMIX to train
            if args.compete or args.well_enemy or args.self_compete:
                model_e = model_enemy if args.compete or args.well_enemy else model
                actions_e = model_e.choose_action(obs_e, action_available_e)
                actions_e = actions_e.reshape(-1)
                actions = np.concatenate([actions_ctrl, actions_e], axis=0)
            else:
                actions = action_greedy(state_to_training, actions_ctrl, height, width)
            # actions = action_random(act_dim,actions_ctrl)
            # get the limited action in the next state
            action_available = get_action_available(actions, ctrl_agent_index, act_dim)
            if args.compete or args.well_enemy or args.self_compete:
                actions_available_e = get_action_available(actions, enemy_agent_index, act_dim)

            # the reward of Env is just the gain of length to each agents
            next_state, reward, done, _, info = env.step(env.encode(actions))
            next_state_to_training = next_state[0]
            next_obs, next_state_map = get_observations(next_state_to_training, ctrl_agent_index, obs_dim, height,
                                                        width, args.mode)
            if args.compete or args.well_enemy or args.self_compete:
                next_obs_e, next_state_map_e = get_observations(next_state_to_training, enemy_agent_index, obs_dim,
                                                                height,
                                                                width, args.mode)
            # =========== ======================= reward shaping ========================================
            reward = np.array(reward)
            episode_reward += reward

            step_reward, split_step_reward = get_reward(info, episode_reward, ctrl_agent_index, enemy_agent_index,
                                                        reward, done, args)
            if args.compete or args.self_compete:
                step_reward_e, _ = get_reward(info, episode_reward, enemy_agent_index, ctrl_agent_index,
                                              reward, done,args)
                episode_tot_reward_e += step_reward_e

            episode_tot_reward += step_reward
            # done = np.array([done] * ctrl_agent_num)
            episode_tot_split_reward += split_step_reward

            # ================================== collect data ========================================
            # Store transition in R
            replay_buffer.append([state_map, obs, actions_ctrl, step_reward, action_available, done])

            if args.compete or args.self_compete:
                enemy_replay_buffer.append([state_map_e, obs_e, actions_e, step_reward_e, actions_available_e, done])
                obs_e, state_map_e = next_obs_e, next_state_map_e
            if args.well_enemy:
                obs_e, state_map_e = next_obs_e, next_state_map_e
            obs, state_map = next_obs, next_state_map

            state_to_training = next_state_to_training # TODO: a great BUG!!!!

            step += 1

            if args.episode_length <= step or done:

                print(f'[Episode {episode:05d}] total_reward: {np.sum(episode_reward[0:3]):d}')
                print(f'\t\t\t\tsnake_1: {episode_reward[0]} '
                      f'snake_2: {episode_reward[1]} snake_3: {episode_reward[2]}')
                print(f'\t\t\t\tsnake_4: {episode_reward[3]} '
                      f'snake_5: {episode_reward[4]} snake_6: {episode_reward[5]}')
                print(
                    f'\t\t\t\tepisode_win:{np.sum(episode_reward[ctrl_agent_index]) > np.sum(episode_reward[enemy_agent_index])}')
                print(
                    f'\t\t\t\tself_length:{np.sum(episode_reward[ctrl_agent_index]):d} enemy_length:{np.sum(episode_reward[enemy_agent_index])}')

                reward_tag = 'reward'

                if args.compete or args.self_compete:
                    writer.add_scalars(reward_tag, global_step=episode,
                                       tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                        'snake_3': episode_reward[2],
                                                        'total': np.sum(episode_reward[0:3]),
                                                        'snake_4': episode_reward[3], 'snake_5': episode_reward[4],
                                                        'snake_6': episode_reward[5],
                                                        'enemy_total': np.sum(episode_reward[3:])})
                else:
                    writer.add_scalars(reward_tag, global_step=episode,
                                       tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                        'snake_3': episode_reward[2],
                                                        'total': np.sum(episode_reward[0:3])})

                score_tag = 'score'
                if args.compete or args.self_compete:
                    writer.add_scalars(score_tag, global_step=episode,
                                       tag_scalar_dict={'mean_step_reward': episode_tot_reward / step,
                                                        "mean_sparse_reward": episode_tot_split_reward[0] / step,
                                                        'mean_gain_reward': episode_tot_split_reward[1] / step,
                                                        'mean_dist_reward': episode_tot_split_reward[2] / step,
                                                        'enemy_mean_step_reward': episode_tot_reward_e / step})
                else:
                    writer.add_scalars(score_tag, global_step=episode,
                                       tag_scalar_dict={'mean_step_reward': episode_tot_reward / step,
                                                        "mean_sparse_reward": episode_tot_split_reward[0] / step,
                                                        'mean_gain_reward': episode_tot_split_reward[1] / step,
                                                        'mean_dist_reward': episode_tot_split_reward[2] / step})

                win_tag = 'win_rate'
                writer.add_scalars(win_tag, global_step=episode,
                                   tag_scalar_dict={'win_rate': int(np.sum(episode_reward[ctrl_agent_index]) > np.sum(
                                       episode_reward[enemy_agent_index]))})

                env.reset()
                break
        model.reset(args.batch_size)
        if args.self_compete:
            model.replay_buffer.push(enemy_replay_buffer)
            enemy_replay_buffer.clear()
        model.replay_buffer.push(replay_buffer)
        replay_buffer.clear()
        model.epsilon_delay(step)
        model.update()
        model.update_target(episode)
        if args.compete:
            model_enemy.reset(args.batch_size)
            model_enemy.replay_buffer.push(enemy_replay_buffer)
            enemy_replay_buffer.clear()
            model_enemy.epsilon_delay(step)
            model_enemy.update()
            model_enemy.update_target(episode)


        loss_tag = 'loss'

        if model.loss:
            if not args.compete:
                writer.add_scalars(loss_tag, global_step=episode,
                                   tag_scalar_dict={'loss': float(model.loss)})
            else:
                if model_enemy.loss:
                    writer.add_scalars(loss_tag, global_step=episode,
                                       tag_scalar_dict={'loss': float(model.loss),
                                                        'enemy': float(model_enemy.loss)})

        if model.loss:
            if not args.compete:
                print(f'\t\t\t\tloss {model.loss:.3f}')
            elif model_enemy.loss:
                print(f'\t\t\t\tloss {model.loss:.3f}')
                print(f'\t\t\t\tloss {model_enemy.loss:.3f}')

        if episode % args.save_interval == 0:
            model.save_model(os.path.join(run_dir, "qmix_agent_%d.pth" % episode))
            if args.compete:
                model_enemy.save_model(os.path.join(run_dir, "qmix_enemy_agent_%d.pth" % episode))
            # model.save_checkpoint(os.path.join(run_dir,"qmix_ckpt_%d.pth"%episode))