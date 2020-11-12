from itertools import product
import json

from agents.base import Agent
from bandit import Bandit
from experiment import Experiment


def load_experiments(config_path: str):
    exp_args = json.loads(open(config_path).read())
    bandit_configs = exp_args['bandits']
    bandits = [
        Bandit(arms=arm, **config)
        for arm, config in product(exp_args['arms'], bandit_configs)
    ]
    agents = [
        Agent.from_dict(agent_dict, idx)
        for idx, agent_dict in enumerate(exp_args['agents'])
    ]
    start_seed = exp_args['seed']
    exps = []
    for idx, (bandit, agent) in enumerate(product(bandits, agents)):
        name = f'arms={bandit.arms},bandit={bandit.name},agent={agent.name}'
        exp = Experiment(name, bandit, agent, exp_args['rounds'],
                         exp_args['trials'], start_seed, exp_args['alpha'])
        start_seed += exp_args['trials']
        exps.append(exp)
    return exps
