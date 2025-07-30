import torch
import numpy as np

# These functions represent prior reward functions decomposed into a vector reward functions
# They take an input array (and optionally an auxiliary array)
# and output an array of function outputs


# Prior reward function for pen task, based on task reward function assign by
# Gymnasium Robotics
# X, the set of state-action pairs is not actually considered here. This reward
# function is based on the next state only. But we keep X to conform with how the
# LCFmodel class works
def pen_task_reward_prior(X, aux_X,device):
    if isinstance(aux_X, np.ndarray):
        aux_X = torch.from_numpy(aux_X).to(device)

    intercept = torch.ones(aux_X.size(0)).double().to(device)

    goal_distance = torch.linalg.norm(aux_X[:, 39:42], dim=1).double()
    orien_similarity = torch.einsum(
        "ij,ij->i", aux_X[:, 33:36], aux_X[:, 36:39]
    ).double()

    close = ((goal_distance < 0.075) & (orien_similarity > 0.9)).double()

    closer = ((goal_distance < 0.075) & (orien_similarity > 0.95)).double()

    dropped = (aux_X[:, 26] < 0.075).double()

    return torch.stack(
        [intercept, -goal_distance, orien_similarity, close, closer, -dropped], dim=1
    ).double()


# Same as above, no intercept
def pen_task_reward_prior_no_intercept(X, aux_X,device):
    if isinstance(aux_X, np.ndarray):
        aux_X = torch.from_numpy(aux_X).to(device)

    goal_distance = torch.linalg.norm(aux_X[:, 39:42], dim=1).double()
    orien_similarity = torch.einsum(
        "ij,ij->i", aux_X[:, 33:36], aux_X[:, 36:39]
    ).double()

    close = ((goal_distance < 0.075) & (orien_similarity > 0.9)).double()

    closer = ((goal_distance < 0.075) & (orien_similarity > 0.95)).double()

    dropped = (aux_X[:, 26] < 0.075).double()

    return torch.stack(
        [-goal_distance, orien_similarity, close, closer, -dropped], dim=1
    ).double()


# BB reward function. Again, just aux_X is used here. Assumes the state contains 6 closest obstacles
def bb_reward_prior(X, aux_X,device):
    if isinstance(aux_X, np.ndarray):
        aux_X = torch.from_numpy(aux_X).to(device)

    intercept = torch.ones(aux_X.size(0)).double().to(device)

    goal_distance = torch.sqrt(
        (aux_X[:, 20] - aux_X[:, 0]) ** 2 + (aux_X[:, 21] - aux_X[:, 1]) ** 2
    ).double()

    min_obs_dist = torch.sqrt(
        (aux_X[:, 2] - aux_X[:, 0]) ** 2 + (aux_X[:, 3] - aux_X[:, 1]) ** 2
    ).double()

    return torch.stack(
        [intercept, -goal_distance, min_obs_dist], dim=1
    ).double()
