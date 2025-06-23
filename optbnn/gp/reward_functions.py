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
def pen_task_reward_prior(X, aux_X):
    if isinstance(aux_X, np.ndarray):
        aux_X = torch.from_np(aux_X)
        
    intercept = torch.ones(aux_X.size(0)).float()
    
    goal_distance = torch.linalg.norm(aux_X[:,39:42],dim=1)
    orien_similarity = torch.einsum('ij,ij->i',aux_X[:,33:36],aux_X[:,36:39])
    
    close = ((goal_distance < 0.075) & (orien_similarity > 0.9)).float()
    
    closer = ((goal_distance < 0.075) & (orien_similarity > 0.95)).float()
    
    dropped = (aux_X[:,26] < 0.075).float()

    return torch.stack([intercept,goal_distance,orien_similarity,close,closer,dropped],dim=1)

# Same as above, no intercept
def pen_task_reward_prior_no_intercept(X, aux_X):
    if isinstance(aux_X, np.ndarray):
        aux_X = torch.from_np(aux_X)
        
    goal_distance = torch.linalg.norm(aux_X[:,39:42],dim=1)
    orien_similarity = torch.einsum('ij,ij->i',aux_X[:,33:36],aux_X[:,36:39])
    
    close = ((goal_distance < 0.075) & (orien_similarity > 0.9)).float()
    
    closer = ((goal_distance < 0.075) & (orien_similarity > 0.95)).float()
    
    dropped = (aux_X[:,26] < 0.075).float()

    return torch.stack([goal_distance,orien_similarity,close,closer,dropped],dim=1)