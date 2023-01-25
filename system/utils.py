import torch
import random
import numpy as np

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Config(object):
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)   