import argparse
import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
from tensorboardX import SummaryWriter
from environment import VoltageCtrl_nonlinear,create_56bus

import os
import gym
import numpy as np
from gym import wrappers

import torch
from ddpg import DDPG
from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition
import matplotlib.pyplot as plt

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                    help='discount factor for model (default: 0.01)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=False)
parser.add_argument('--noise_scale', type=float, default=2, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.5, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=200, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=13, metavar='N',
                    help='random seed (default: 13)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=600, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=100, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
args = parser.parse_args()

writer = SummaryWriter()

pp_net = create_56bus()
injection_bus = np.array([17, 20, 29, 44, 52])
env = VoltageCtrl_nonlinear(pp_net, injection_bus)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent1 = torch.load('./models/dagent1.pt')
agent1.actor.eval()
agent1.actor_target.eval()
agent1.critic.eval()
agent1.critic_target.eval()

agent2 = torch.load('./models/dagent2.pt')
agent3 = torch.load('./models/dagent3.pt')
agent4 = torch.load('./models/dagent4.pt')                                      
agent5 = torch.load('./models/dagent5.pt')

ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, 
    desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05) if args.param_noise else None

rewards = []
total_numsteps = 0
updates = 0
best_val_reward = -100000

for i_episode in range(1):
    state = torch.Tensor([env.reset()])

    if args.ou_noise: 
        ounoise.scale = 0.0
        # ounoise.scale = args.noise_scale
        ounoise.reset()

    # if args.param_noise and args.algo == "DDPG":
    #     agent.perturb_actor_parameters(param_noise)

    episode_reward = 0
    episode_len = 0
    log = []
    while True: #state:[1,1]
        state1 = state[:,0].unsqueeze(-1)
        state2 = state[:,1].unsqueeze(-1)
        state3 = state[:,2].unsqueeze(-1)
        state4 = state[:,3].unsqueeze(-1)
        state5 = state[:,4].unsqueeze(-1)
        action1 = agent1.select_action(state1, ounoise, param_noise)
        action2 = agent2.select_action(state2, ounoise, param_noise)
        action3 = agent3.select_action(state3, ounoise, param_noise)
        action4 = agent4.select_action(state4, ounoise, param_noise)
        action5 = agent5.select_action(state5, ounoise, param_noise)
        action = torch.cat([action1, action2, action3, action4, action5], dim=1)
        log.append(torch.cat([state,action],dim=1).detach().cpu().numpy())
        next_state, reward, done, _ = env.step(action.numpy()[0])
        # print(action)
        total_numsteps += 1
        episode_reward += np.mean(reward)
        episode_len += 1

        action = torch.Tensor(action)
        mask = torch.Tensor([not done])
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])
        state = next_state
        if done or episode_len==60:
            log = np.vstack(log)
            np.savetxt('train_logttd.txt',log, fmt='%1.4e')
            break

    writer.add_scalar('reward/train', episode_reward, i_episode)    
    print("Episode: {}, total numsteps: {}, reward: {}".format(i_episode, total_numsteps, episode_reward))
    
env.close()
# for i_episode in range(1):
#     state = torch.linspace(0.8,1.2,60)

#     if args.ou_noise: 
#         ounoise.scale = 0.0
#         # ounoise.scale = args.noise_scale
#         ounoise.reset()

#     # if args.param_noise and args.algo == "DDPG":
#     #     agent.perturb_actor_parameters(param_noise)

#     episode_reward = 0
#     episode_len = 0
#     log = []
#     for i in range(60): #state:[1,1]
#         state1 = state[i].unsqueeze(-1).unsqueeze(-1)
#         state2 = state[i].unsqueeze(-1).unsqueeze(-1)
#         state3 = state[i].unsqueeze(-1).unsqueeze(-1)
#         state4 = state[i].unsqueeze(-1).unsqueeze(-1)
#         state5 = state[i].unsqueeze(-1).unsqueeze(-1)
#         action1 = agent1.select_action(state1, ounoise, param_noise)
#         action2 = agent2.select_action(state2, ounoise, param_noise)
#         action3 = agent3.select_action(state3, ounoise, param_noise)
#         action4 = agent4.select_action(state4, ounoise, param_noise)
#         action5 = agent5.select_action(state5, ounoise, param_noise)
#         action = torch.cat([action1, action2, action3, action4, action5], dim=1)
#         log.append(torch.cat([state[i].unsqueeze(-1).unsqueeze(-1),action],dim=1).detach().cpu().numpy())
        
#     log = np.vstack(log)
#     np.savetxt('v-u.txt',log, fmt='%1.4e')
#     break
# env.close()