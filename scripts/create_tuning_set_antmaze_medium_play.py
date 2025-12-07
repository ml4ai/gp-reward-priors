import d4rl
import gym
import numpy as np
import h5py

def qlearning_ant_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    goal_ = []
    xy_ = []
    done_bef_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])
        goal = dataset["infos/goal"][i].astype(np.float32)
        xy = dataset["infos/qpos"][i][:2].astype(np.float32)

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
            next_final_timestep = dataset["timeouts"][i + 1]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
            next_final_timestep = episode_step == env._max_episode_steps - 2

        done_bef = bool(next_final_timestep)

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        goal_.append(goal)
        xy_.append(xy)
        done_bef_.append(done_bef)
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "goals": np.array(goal_),
        "xys": np.array(xy_),
        "dones_bef": np.array(done_bef_),
    }


env = "antmaze-medium-play-v2"

gym_env = gym.make(env)

dataset = qlearning_ant_dataset(gym_env)

base_path = "./../data/antmaze"


with h5py.File(base_path + "/antmaze-medium-play-v2_tuning_set.hdf5", "a") as f:
    obs = np.concatenate(
            [f["observations"][:], f["actions"][:]],
            axis=-1,
    )

    aux_obs = np.concatenate(
            [f["goals"][:], f["xys"][:]],
            axis=-1,
    )
    f.create_dataset("obs", data=obs)
    f.create_dataset("aux_obs", data=aux_obs)