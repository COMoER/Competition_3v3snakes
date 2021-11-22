import random
import numpy as np


class ReplayBuffer:
    """
    This replay buffer is combined of a set of a whole episode
    """

    def __init__(self, buffer_size, batch_size):
        self.replay_buffer = []
        self.cache_episode_buffer = []
        self.max_size = buffer_size
        self.batch_size = batch_size

    def push(self, replay_buffer):
        """
        Args:
            replay_buffer: list[(state_map,obs,action,step_reward,action_available,done)]:one episode
        """
        state_episode = np.stack([_[0] for _ in replay_buffer], axis=0)  # T,H,W
        obs_episode = np.stack([_[1] for _ in replay_buffer], axis=0)  # T,N,obs_dim
        action_episode = np.stack([_[2] for _ in replay_buffer], axis=0)  # T,N
        reward_episode = np.array([_[3] for _ in replay_buffer])  # T
        action_available_episode = np.stack([_[4] for _ in replay_buffer],axis = 0) # T,N,act_dim
        done_episode = np.array([_[5] for _ in replay_buffer])  # T
        if len(self.replay_buffer) >= self.max_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append([state_episode, obs_episode, action_episode, reward_episode,action_available_episode, done_episode])

    def get_batches(self):
        """
        return B,T,(state,action,reward,next_state,action_available,done)
        """
        sample_batch = random.sample(self.replay_buffer, self.batch_size)

        state_batches = np.stack([_[0] for _ in sample_batch], axis=0)  # B,T,H,W
        obs_batches = np.stack([_[1] for _ in sample_batch], axis=0)  # B,T,N,obs_dim
        action_batches = np.stack([_[2] for _ in sample_batch], axis=0)  # B,T
        reward_batches = np.stack([_[3] for _ in sample_batch], axis=0)  # B,T
        action_available_batches = np.stack([_[4] for _ in sample_batch], axis=0)  # B,T,N,act_dim
        done_batches = np.stack([_[5] for _ in sample_batch], axis=0)  # B,T

        return state_batches, obs_batches, action_batches, reward_batches, action_available_batches,done_batches

    def __len__(self):
        return len(self.replay_buffer)
