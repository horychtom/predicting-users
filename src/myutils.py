"""This module contains utils."""
import random

import numpy as np
import torch

from config import RANDOM_SEED


def get_class_weights(y, method="ins"):
    """Compute the weights for vector of counts of each label.

    ins = inverse number of samples
    isns = inverse squared number of samples
    esns = effective sampling number of samples
    """
    counts = y.unique(return_counts=True)[1]

    if method == "ins":
        weights = 1.0 / counts
        weights = weights / sum(weights)
    if method == "isns":
        weights = 1.0 / torch.pow(counts, 0.5)
        weights = weights / sum(weights)
    if method == "esns":
        beta = 0.999
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
        weights = weights / sum(weights)

    return weights


def set_random_seed(seed=RANDOM_SEED):
    """Random seed for comparable results."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.cuda.manual_seed_all(seed)
